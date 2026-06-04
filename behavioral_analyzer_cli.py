"""Command-line interface for behavioral text analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from behavioral_analyzer import BehavioralAnalyzer, BehavioralAnalyzerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ASR/diarization transcript output and append context, sentiment, "
            "emotion, and anomaly scores."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .json/.jsonl/.csv/.parquet file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output .json/.csv/.parquet file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto", "cpu", "cuda", or device index string.',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for transformer pipeline inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer truncation max_length for per-utterance inference.",
    )
    return parser.parse_args()


def _read_input(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            return rows
        payload = json.loads(text)
        return payload

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for CSV/Parquet input.") from exc

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def _write_output(df: Any, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {suffix}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = BehavioralAnalyzerConfig(
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    analyzer = BehavioralAnalyzer(config=config)
    input_data = _read_input(input_path)
    enriched_df = analyzer.analyze(input_data)
    _write_output(enriched_df, output_path)

    print(
        json.dumps(
            {
                "status": "ok",
                "input": str(input_path),
                "output": str(output_path),
                "rows": int(len(enriched_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
