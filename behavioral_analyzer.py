"""Text behavioral analysis layer for audio deepfake detection pipelines.

This module consumes utterance-level ASR/diarization output and enriches it
with text-derived behavioral signals:

* global and utterance-level primary context
* sentiment polarity
* fine-grained emotion collapsed into operational classes
* heuristic behavioral anomaly scores

The implementation is intentionally modular. Hugging Face models are isolated
behind small adapter classes, while the anomaly rules are deterministic and
testable without loading transformer weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence


SentimentLabel = Literal["Positive", "Negative", "Neutral"]
EmotionLabel = Literal["Anger", "Fear", "Sadness", "Joy", "Neutral"]
ContextLabel = str


REQUIRED_COLUMNS = ("start_time", "end_time", "speaker_id", "transcript")


@dataclass(frozen=True)
class LabelScore:
    """Normalized classifier output.

    Attributes:
        label: Human-readable model or canonical label.
        score: Confidence score in the inclusive range [0.0, 1.0].
    """

    label: str
    score: float


@dataclass(frozen=True)
class PrimaryContextSummary:
    """Conversation-level context and intent summary.

    Attributes:
        primary_context: Best global context label, such as
            ``"Financial Wire Transfer"`` or ``"Tech Support"``.
        primary_context_confidence: Confidence for ``primary_context``.
        primary_intent: Best global intent label, such as
            ``"Request Payment"`` or ``"Resolve Account Access"``.
        primary_intent_confidence: Confidence for ``primary_intent``.
        context_scores: Ranked global context candidates.
        intent_scores: Ranked global intent candidates.
        evidence_utterance_count: Number of non-empty utterances used for the
            global classification pass.
    """

    primary_context: str
    primary_context_confidence: float
    primary_intent: str
    primary_intent_confidence: float
    context_scores: list[LabelScore]
    intent_scores: list[LabelScore]
    evidence_utterance_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["context_scores"] = [asdict(score) for score in self.context_scores]
        payload["intent_scores"] = [asdict(score) for score in self.intent_scores]
        return payload


@dataclass(frozen=True)
class BehavioralFlag:
    """A single deterministic behavioral anomaly explanation."""

    code: str
    severity: Literal["low", "medium", "high"]
    message: str
    score_delta: float
    start_time: Any | None = None
    end_time: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class BehavioralAnalyzerConfig:
    """Runtime configuration for ``BehavioralAnalyzer``.

    The default model IDs are permissively licensed at the time this skeleton
    was created, but production deployments should pin exact revisions and keep
    a local model-license manifest.

    Attributes:
        context_model_name: Zero-shot NLI model used for global and utterance
            context classification.
        sentiment_model_name: Sentiment classifier. The default SST-2 model is
            binary, so neutral is inferred when confidence is below
            ``sentiment_neutral_threshold``.
        emotion_model_name: Emotion classifier. The default GoEmotions model
            emits 28 labels that are collapsed into the required operational
            classes by ``EmotionAnalyzer``.
        context_candidate_labels: Candidate labels for global and local topic.
        intent_candidate_labels: Candidate labels for global conversation
            intent.
        batch_size: Batch size used by transformer pipelines.
        max_length: Token truncation length for utterance-level inference.
        global_context_max_chars: Maximum characters retained when building the
            global conversation evidence text.
        global_context_chunk_chars: Maximum characters per global evidence
            chunk. Chunks are classified independently and aggregated, reducing
            truncation bias on long calls.
        sentiment_neutral_threshold: If the top binary sentiment confidence is
            below this value, classify the utterance as neutral.
        emotion_neutral_margin: If neutral is close to the best non-neutral
            emotion, prefer neutral to reduce false urgency.
        min_text_chars: Very short or blank utterances are assigned neutral
            text signals and skipped for global context evidence.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or a Hugging Face pipeline
            integer device encoded as string.
        use_fp16_on_cuda: Pass half precision to models when CUDA is selected.
        high_risk_contexts: Context labels that increase vishing suspicion.
        urgency_keywords: Keyword cues used by the deterministic anomaly
            scorer.
        sensitive_action_keywords: Financial/security action cues used by the
            deterministic anomaly scorer.
    """

    context_model_name: str = "facebook/bart-large-mnli"
    sentiment_model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    emotion_model_name: str = "SamLowe/roberta-base-go_emotions"
    context_candidate_labels: tuple[str, ...] = (
        "Greeting",
        "Tech Support",
        "Financial Wire Transfer",
        "Banking or Account Access",
        "Personal Catch-up",
        "Healthcare Appointment",
        "Job or Workplace",
        "Romance or Relationship",
        "Law Enforcement or Legal Threat",
        "Retail Delivery or Refund",
        "Travel or Logistics",
        "Education",
        "Unknown or Other",
    )
    intent_candidate_labels: tuple[str, ...] = (
        "Request Money Transfer",
        "Collect Sensitive Code or Password",
        "Resolve Technical Problem",
        "Verify Identity",
        "Schedule or Coordinate",
        "Social Conversation",
        "Provide Customer Support",
        "Threaten Consequences",
        "Unknown Intent",
    )
    batch_size: int = 4
    max_length: int = 256
    global_context_max_chars: int = 6000
    global_context_chunk_chars: int = 1600
    sentiment_neutral_threshold: float = 0.68
    emotion_neutral_margin: float = 0.08
    min_text_chars: int = 3
    device: str = "auto"
    use_fp16_on_cuda: bool = True
    high_risk_contexts: Mapping[str, float] = field(
        default_factory=lambda: {
            "Financial Wire Transfer": 36.0,
            "Banking or Account Access": 30.0,
            "Collect Sensitive Code or Password": 28.0,
            "Law Enforcement or Legal Threat": 24.0,
            "Tech Support": 18.0,
        }
    )
    urgency_keywords: tuple[str, ...] = (
        "urgent",
        "immediately",
        "right now",
        "do not tell",
        "keep this confidential",
        "police",
        "warrant",
        "arrest",
        "locked",
        "suspended",
        "final warning",
    )
    sensitive_action_keywords: tuple[str, ...] = (
        "wire",
        "transfer",
        "bank",
        "account",
        "routing",
        "password",
        "passcode",
        "one time code",
        "otp",
        "gift card",
        "crypto",
        "bitcoin",
        "wallet",
        "social security",
    )


class TextClassifier(Protocol):
    """Protocol implemented by inference adapters."""

    def predict(self, texts: Sequence[str]) -> list[LabelScore]:
        """Predict one best label for every input text."""


class HuggingFacePipelineFactory:
    """Factory that lazily constructs optimized Hugging Face pipelines.

    The factory keeps import-time dependencies optional. Environments that only
    unit-test the rule engine can import this module without installing torch or
    transformers.
    """

    def __init__(self, config: BehavioralAnalyzerConfig) -> None:
        self.config = config
        self._device = self._resolve_device(config.device)

    @property
    def device(self) -> int:
        """Return the integer device expected by ``transformers.pipeline``."""

        return self._device

    def build(self, task: str, model_name: str) -> Any:
        """Create a Hugging Face pipeline for a task and model.

        Args:
            task: Hugging Face pipeline task name.
            model_name: Local path or model repository ID.

        Returns:
            A configured ``transformers.Pipeline`` instance.

        Raises:
            RuntimeError: If transformers or torch are not installed.
        """

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and transformers in the active virtual environment "
                "before running transformer inference."
            ) from exc

        model_kwargs: dict[str, Any] = {}
        if self._device >= 0 and self.config.use_fp16_on_cuda:
            model_kwargs["torch_dtype"] = torch.float16

        return pipeline(
            task,
            model=model_name,
            device=self._device,
            model_kwargs=model_kwargs,
        )

    @staticmethod
    def _resolve_device(device: str) -> int:
        """Resolve a config device value into a pipeline device integer."""

        if device == "cpu":
            return -1
        if device == "cuda":
            return 0
        if device.isdigit() or (device.startswith("-") and device[1:].isdigit()):
            return int(device)
        if device != "auto":
            raise ValueError(f"Unsupported device value: {device!r}")

        try:
            import torch
        except ImportError:
            return -1
        return 0 if torch.cuda.is_available() else -1


class PrimaryContextEngine:
    """Global and utterance-level context extractor.

    This engine is intentionally instantiated by ``BehavioralAnalyzer`` during
    initialization so primary context is part of every inference flow, including
    anomaly synthesis.
    """

    def __init__(
        self,
        config: BehavioralAnalyzerConfig,
        pipeline_factory: HuggingFacePipelineFactory,
    ) -> None:
        self.config = config
        self._classifier = pipeline_factory.build(
            "zero-shot-classification",
            config.context_model_name,
        )

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> PrimaryContextSummary:
        """Classify global context and intent from conversation evidence."""

        evidence_chunks = self._build_global_evidence_chunks(rows)
        if not evidence_chunks:
            unknown = LabelScore("Unknown or Other", 1.0)
            unknown_intent = LabelScore("Unknown Intent", 1.0)
            return PrimaryContextSummary(
                primary_context=unknown.label,
                primary_context_confidence=unknown.score,
                primary_intent=unknown_intent.label,
                primary_intent_confidence=unknown_intent.score,
                context_scores=[unknown],
                intent_scores=[unknown_intent],
                evidence_utterance_count=0,
            )

        context_scores = self._zero_shot_aggregate(
            evidence_chunks,
            self.config.context_candidate_labels,
        )
        intent_scores = self._zero_shot_aggregate(
            evidence_chunks,
            self.config.intent_candidate_labels,
        )
        best_context = context_scores[0]
        best_intent = intent_scores[0]

        return PrimaryContextSummary(
            primary_context=best_context.label,
            primary_context_confidence=best_context.score,
            primary_intent=best_intent.label,
            primary_intent_confidence=best_intent.score,
            context_scores=context_scores,
            intent_scores=intent_scores,
            evidence_utterance_count=sum(
                1 for row in rows if _clean_text(row.get("transcript", ""))
            ),
        )

    def classify_utterances(self, texts: Sequence[str]) -> list[LabelScore]:
        """Classify utterance-level context for every transcript."""

        clean_texts = [_clean_text(text) for text in texts]
        if not clean_texts:
            return []
        results = [LabelScore("Unknown or Other", 1.0) for _ in clean_texts]
        valid_items = [
            (index, text)
            for index, text in enumerate(clean_texts)
            if len(text) >= self.config.min_text_chars
        ]
        if not valid_items:
            return results

        outputs = self._classifier(
            [text for _, text in valid_items],
            candidate_labels=list(self.config.context_candidate_labels),
            truncation=True,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
        )
        if isinstance(outputs, Mapping):
            outputs = [outputs]

        for (index, _), output in zip(valid_items, outputs, strict=False):
            results[index] = self._best_score(output)
        return results

    def _zero_shot(self, text: str, labels: Sequence[str]) -> list[LabelScore]:
        output = self._classifier(
            text,
            candidate_labels=list(labels),
            truncation=True,
            max_length=self.config.max_length,
        )
        return self._ranked_scores(output)

    def _zero_shot_aggregate(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> list[LabelScore]:
        outputs = self._classifier(
            list(texts),
            candidate_labels=list(labels),
            truncation=True,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
        )
        if isinstance(outputs, Mapping):
            outputs = [outputs]

        aggregate = {label: 0.0 for label in labels}
        output_count = 0
        for output in outputs:
            output_count += 1
            for label_score in self._ranked_scores(output):
                aggregate[label_score.label] = aggregate.get(label_score.label, 0.0) + label_score.score

        if output_count == 0:
            return []
        return sorted(
            (
                LabelScore(label, _clip_score(score / output_count))
                for label, score in aggregate.items()
            ),
            key=lambda item: item.score,
            reverse=True,
        )

    def _build_global_evidence_chunks(self, rows: Sequence[Mapping[str, Any]]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_chars = 0
        total_chars = 0
        total_budget = self.config.global_context_max_chars
        chunk_budget = self.config.global_context_chunk_chars

        for row in rows:
            text = _clean_text(row.get("transcript", ""))
            if len(text) < self.config.min_text_chars:
                continue
            speaker = row.get("speaker_id", "unknown")
            start = row.get("start_time", "")
            snippet = f"[{start}] {speaker}: {text}"

            if total_chars + len(snippet) > total_budget:
                break
            if current and current_chars + len(snippet) > chunk_budget:
                chunks.append("\n".join(current))
                current = []
                current_chars = 0

            current.append(snippet)
            snippet_chars = len(snippet)
            current_chars += snippet_chars
            total_chars += snippet_chars

        if current:
            chunks.append("\n".join(current))
        return chunks

    @staticmethod
    def _ranked_scores(output: Mapping[str, Any]) -> list[LabelScore]:
        labels = output.get("labels", [])
        scores = output.get("scores", [])
        return [
            LabelScore(str(label), _clip_score(score))
            for label, score in zip(labels, scores, strict=False)
        ]

    @classmethod
    def _best_score(cls, output: Mapping[str, Any]) -> LabelScore:
        ranked = cls._ranked_scores(output)
        return ranked[0] if ranked else LabelScore("Unknown or Other", 1.0)


class SentimentAnalyzer:
    """Positive, negative, neutral sentiment classifier."""

    def __init__(
        self,
        config: BehavioralAnalyzerConfig,
        pipeline_factory: HuggingFacePipelineFactory,
    ) -> None:
        self.config = config
        self._classifier = pipeline_factory.build("text-classification", config.sentiment_model_name)

    def predict(self, texts: Sequence[str]) -> list[LabelScore]:
        """Predict sentiment polarity for every text in a batch."""

        clean_texts = [_clean_text(text) for text in texts]
        if not clean_texts:
            return []
        results = [LabelScore("Neutral", 1.0) for _ in clean_texts]
        valid_items = [
            (index, text)
            for index, text in enumerate(clean_texts)
            if len(text) >= self.config.min_text_chars
        ]
        if not valid_items:
            return results

        outputs = self._classifier(
            [text for _, text in valid_items],
            truncation=True,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
        )
        if isinstance(outputs, Mapping):
            outputs = [outputs]

        for (index, _), output in zip(valid_items, outputs, strict=False):
            results[index] = self._normalize(output)
        return results

    def _normalize(self, output: Mapping[str, Any]) -> LabelScore:
        raw_label = str(output.get("label", "Neutral")).lower()
        score = _clip_score(output.get("score", 0.0))

        if score < self.config.sentiment_neutral_threshold:
            return LabelScore("Neutral", 1.0 - score)
        if "pos" in raw_label or raw_label.endswith("_1") or raw_label == "1":
            return LabelScore("Positive", score)
        if "neg" in raw_label or raw_label.endswith("_0") or raw_label == "0":
            return LabelScore("Negative", score)
        return LabelScore(raw_label.title(), score)


class EmotionAnalyzer:
    """Fine-grained emotion classifier collapsed to operational classes."""

    _EMOTION_GROUPS: Mapping[EmotionLabel, tuple[str, ...]] = {
        "Anger": ("anger", "annoyance", "disapproval", "disgust"),
        "Fear": ("fear", "nervousness"),
        "Sadness": ("sadness", "grief", "remorse", "disappointment"),
        "Joy": (
            "joy",
            "amusement",
            "admiration",
            "excitement",
            "gratitude",
            "love",
            "optimism",
            "relief",
            "pride",
            "caring",
        ),
        "Neutral": ("neutral",),
    }

    def __init__(
        self,
        config: BehavioralAnalyzerConfig,
        pipeline_factory: HuggingFacePipelineFactory,
    ) -> None:
        self.config = config
        self._classifier = pipeline_factory.build("text-classification", config.emotion_model_name)

    def predict(self, texts: Sequence[str]) -> list[LabelScore]:
        """Predict the dominant operational emotion for every text."""

        clean_texts = [_clean_text(text) for text in texts]
        if not clean_texts:
            return []
        results = [LabelScore("Neutral", 1.0) for _ in clean_texts]
        valid_items = [
            (index, text)
            for index, text in enumerate(clean_texts)
            if len(text) >= self.config.min_text_chars
        ]
        if not valid_items:
            return results

        outputs = self._classifier(
            [text for _, text in valid_items],
            top_k=None,
            truncation=True,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
        )
        if outputs and isinstance(outputs[0], Mapping):
            outputs = [outputs]

        for (index, _), output in zip(valid_items, outputs, strict=False):
            results[index] = self._collapse(output)
        return results

    def _collapse(self, output: Sequence[Mapping[str, Any]]) -> LabelScore:
        group_scores: dict[EmotionLabel, float] = {
            "Anger": 0.0,
            "Fear": 0.0,
            "Sadness": 0.0,
            "Joy": 0.0,
            "Neutral": 0.0,
        }
        for item in output:
            raw_label = str(item.get("label", "")).lower()
            score = _clip_score(item.get("score", 0.0))
            for group, source_labels in self._EMOTION_GROUPS.items():
                if raw_label in source_labels:
                    group_scores[group] += score
                    break

        best_label, best_score = max(group_scores.items(), key=lambda item: item[1])
        neutral_score = group_scores["Neutral"]
        if best_label != "Neutral" and neutral_score + self.config.emotion_neutral_margin >= best_score:
            return LabelScore("Neutral", neutral_score)
        return LabelScore(best_label, _clip_score(best_score))


class BehavioralAnomalyScorer:
    """Rule-based synthesizer for behavioral anomaly scoring.

    The scorer deliberately uses transparent heuristics. It correlates the
    model-derived context, sentiment, emotion trajectory, timestamps, and
    high-risk lexical cues into a bounded Vishing Suspicion Index.
    """

    def __init__(self, config: BehavioralAnalyzerConfig) -> None:
        self.config = config

    def score(
        self,
        rows: Sequence[Mapping[str, Any]],
        context_summary: PrimaryContextSummary,
    ) -> tuple[float, list[BehavioralFlag]]:
        """Compute a conversation-level Vishing Suspicion Index.

        Args:
            rows: Enriched utterance rows.
            context_summary: Global context and intent emitted by
                ``PrimaryContextEngine``.

        Returns:
            A tuple containing ``(score_0_to_100, flags)``.
        """

        flags: list[BehavioralFlag] = []
        score = 0.0

        score += self._score_high_risk_context(context_summary, flags)
        score += self._score_emotional_pressure(rows, flags)
        score += self._score_abrupt_emotion_shifts(rows, flags)
        score += self._score_keyword_pressure(rows, flags)

        return min(100.0, round(score, 2)), flags

    def _score_high_risk_context(
        self,
        summary: PrimaryContextSummary,
        flags: list[BehavioralFlag],
    ) -> float:
        context_delta = self.config.high_risk_contexts.get(summary.primary_context, 0.0)
        intent_delta = self.config.high_risk_contexts.get(summary.primary_intent, 0.0)
        delta = max(context_delta, intent_delta)
        if delta <= 0:
            return 0.0

        weighted_delta = delta * max(summary.primary_context_confidence, summary.primary_intent_confidence)
        flags.append(
            BehavioralFlag(
                code="HIGH_RISK_CONTEXT",
                severity="high" if weighted_delta >= 25 else "medium",
                message=(
                    "Conversation context or intent maps to a high-risk social "
                    "engineering category."
                ),
                score_delta=round(weighted_delta, 2),
            )
        )
        return weighted_delta

    @staticmethod
    def _score_emotional_pressure(
        rows: Sequence[Mapping[str, Any]],
        flags: list[BehavioralFlag],
    ) -> float:
        pressure_rows = [
            row
            for row in rows
            if row.get("detected_emotion") in {"Fear", "Anger"}
            or row.get("sentiment_polarity") == "Negative"
        ]
        if not rows or not pressure_rows:
            return 0.0

        ratio = len(pressure_rows) / len(rows)
        delta = min(24.0, ratio * 30.0)
        if delta >= 8.0:
            flags.append(
                BehavioralFlag(
                    code="SUSTAINED_EMOTIONAL_PRESSURE",
                    severity="high" if delta >= 18 else "medium",
                    message="Fear, anger, or negative polarity is sustained across the call.",
                    score_delta=round(delta, 2),
                )
            )
        return delta

    @staticmethod
    def _score_abrupt_emotion_shifts(
        rows: Sequence[Mapping[str, Any]],
        flags: list[BehavioralFlag],
    ) -> float:
        delta = 0.0
        prior: Mapping[str, Any] | None = None
        calm_states = {"Neutral", "Joy", "Positive"}
        pressure_states = {"Fear", "Anger", "Negative"}

        for row in rows:
            if prior is None:
                prior = row
                continue

            prior_state = prior.get("detected_emotion") or prior.get("sentiment_polarity")
            current_state = row.get("detected_emotion") or row.get("sentiment_polarity")
            if prior_state in calm_states and current_state in pressure_states:
                shift_delta = 10.0
                delta += shift_delta
                flags.append(
                    BehavioralFlag(
                        code="ABRUPT_EMOTION_SHIFT",
                        severity="medium",
                        message="Emotion shifted abruptly from calm/positive to pressure.",
                        score_delta=shift_delta,
                        start_time=prior.get("start_time"),
                        end_time=row.get("end_time"),
                    )
                )
            prior = row

        return min(delta, 20.0)

    def _score_keyword_pressure(
        self,
        rows: Sequence[Mapping[str, Any]],
        flags: list[BehavioralFlag],
    ) -> float:
        joined_text = " ".join(_clean_text(row.get("transcript", "")).lower() for row in rows)
        urgency_hits = [keyword for keyword in self.config.urgency_keywords if keyword in joined_text]
        sensitive_hits = [
            keyword for keyword in self.config.sensitive_action_keywords if keyword in joined_text
        ]

        delta = min(10.0, len(urgency_hits) * 2.0) + min(16.0, len(sensitive_hits) * 2.0)
        if delta > 0:
            flags.append(
                BehavioralFlag(
                    code="LEXICAL_PRESSURE_CUES",
                    severity="high" if delta >= 18 else "medium",
                    message="Urgency or sensitive-action language appears in the transcript.",
                    score_delta=round(delta, 2),
                )
            )
        return delta


class BehavioralAnalyzer:
    """Orchestrates the final text-based analysis layer.

    Typical use:

    ```python
    analyzer = BehavioralAnalyzer()
    enriched_df = analyzer.analyze(upstream_dataframe)
    ```

    The constructor initializes the primary context engine, sentiment analyzer,
    emotion analyzer, and anomaly scorer. During ``analyze`` the global primary
    context is computed before local classifiers so the anomaly synthesizer can
    correlate utterance-level emotional shifts with conversation intent.
    """

    def __init__(self, config: BehavioralAnalyzerConfig | None = None) -> None:
        self.config = config or BehavioralAnalyzerConfig()
        self.pipeline_factory = HuggingFacePipelineFactory(self.config)
        self.primary_context_engine = PrimaryContextEngine(self.config, self.pipeline_factory)
        self.sentiment_analyzer = SentimentAnalyzer(self.config, self.pipeline_factory)
        self.emotion_analyzer = EmotionAnalyzer(self.config, self.pipeline_factory)
        self.anomaly_scorer = BehavioralAnomalyScorer(self.config)

    def analyze(self, data: Any) -> Any:
        """Enrich ASR/diarization output with behavioral text signals.

        Args:
            data: A pandas ``DataFrame`` or JSON-compatible sequence of
                dictionaries containing ``start_time``, ``end_time``,
                ``speaker_id``, and ``transcript``.

        Returns:
            A pandas ``DataFrame`` with appended columns:
            ``sentiment_polarity``, ``sentiment_confidence``,
            ``detected_emotion``, ``emotion_confidence``,
            ``utterance_context``, ``utterance_context_confidence``,
            ``primary_context_summary``, ``vishing_suspicion_index``, and
            ``behavioral_flags``.

        Raises:
            ValueError: If required upstream columns are absent.
            RuntimeError: If pandas is unavailable.
        """

        df = self._coerce_dataframe(data)
        self._validate_dataframe(df)

        if df.empty:
            return self._append_empty_outputs(df)

        df = df.copy()
        df["transcript"] = df["transcript"].fillna("").astype(str)
        rows = df[list(REQUIRED_COLUMNS)].to_dict(orient="records")
        transcripts = df["transcript"].tolist()

        context_summary = self.primary_context_engine.summarize(rows)
        utterance_contexts = self.primary_context_engine.classify_utterances(transcripts)
        sentiments = self.sentiment_analyzer.predict(transcripts)
        emotions = self.emotion_analyzer.predict(transcripts)

        df["utterance_context"] = [item.label for item in utterance_contexts]
        df["utterance_context_confidence"] = [item.score for item in utterance_contexts]
        df["sentiment_polarity"] = [item.label for item in sentiments]
        df["sentiment_confidence"] = [item.score for item in sentiments]
        df["detected_emotion"] = [item.label for item in emotions]
        df["emotion_confidence"] = [item.score for item in emotions]
        df["primary_context_summary"] = [context_summary.to_dict()] * len(df)

        enriched_rows = df.to_dict(orient="records")
        vishing_score, flags = self.anomaly_scorer.score(enriched_rows, context_summary)
        df["vishing_suspicion_index"] = vishing_score
        df["behavioral_flags"] = [[flag.to_dict() for flag in flags]] * len(df)
        return df

    @staticmethod
    def _coerce_dataframe(data: Any) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Install pandas in the active virtual environment to use BehavioralAnalyzer."
            ) from exc

        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, str):
            return pd.read_json(data)
        if isinstance(data, Iterable):
            return pd.DataFrame(list(data))
        raise TypeError("data must be a pandas DataFrame, JSON string, or sequence of dicts.")

    @staticmethod
    def _validate_dataframe(df: Any) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required upstream columns: {missing}")

    @staticmethod
    def _append_empty_outputs(df: Any) -> Any:
        df = df.copy()
        output_defaults: Mapping[str, Any] = {
            "utterance_context": [],
            "utterance_context_confidence": [],
            "sentiment_polarity": [],
            "sentiment_confidence": [],
            "detected_emotion": [],
            "emotion_confidence": [],
            "primary_context_summary": [],
            "vishing_suspicion_index": [],
            "behavioral_flags": [],
        }
        for column, value in output_defaults.items():
            df[column] = value
        return df


def _clean_text(value: Any) -> str:
    """Normalize transcript text for model input and keyword matching."""

    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _clip_score(value: Any) -> float:
    """Convert a model score to a stable probability-like float."""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))
