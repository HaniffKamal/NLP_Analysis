"""
demo_app.py
===========
Interactive Web GUI for Sentiment Polarity & Emotion Classification models.
Built with Streamlit and Hugging Face Transformers.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import torch
from transformers import pipeline
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(
    page_title="NLP Behavioral Analysis Demo",
    page_icon="🧠",
    layout="wide",
)

# 2. Custom CSS Injection for Premium Styling
st.markdown(
    """
    <style>
    /* Custom main header styling with gradient */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 8px;
        margin-left: 0.5rem;
    }
    .status-gpu {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-cpu {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Cache Resource for Model Loading
@st.cache_resource
def load_pipelines():
    # Detect hardware device (0 for GPU, -1 for CPU)
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    
    # Load Sentiment Polarity student model
    sentiment_pipe = pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        top_k=None,
        device=device
    )
    
    # Load Emotion Classification model
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=device
    )
    
    # Load Context / Intent Zero-shot Classification model
    context_pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    
    return sentiment_pipe, emotion_pipe, context_pipe, device_name

# Initialize model loading spinner
with st.spinner("Initializing Hugging Face NLP models. This might take a moment on the first run..."):
    sentiment_pipe, emotion_pipe, context_pipe, device_name = load_pipelines()

# Predefined Context Candidates
CONTEXT_CANDIDATES = [
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
]

# Header Section
st.markdown('<div class="main-title">NLP Behavioral Analysis</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Demonstrating real-time Sentiment Polarity, Emotion and Context Classification models. '
    f'Active Device: <span class="status-badge status-{"gpu" if "GPU" in device_name else "cpu"}">{device_name}</span></div>',
    unsafe_allow_html=True
)

# Layout: Left side for inputs, Right side for live metrics & charts
col_input, col_results = st.columns([1, 1.2], gap="large")

# Predefined Examples
examples = {
    "Select an example sentence to test...": "",
    "Greeting (Good morning)": "Hi, good morning! I hope you are having a wonderful day. How can I help you today?",
    "Very Positive & Joyful": "I am absolutely thrilled and overjoyed with this incredible news! Thank you so much!",
    "Urgent, Angered & Warning (Scam Scenario)": "Your bank account has been compromised! Transfer all your savings immediately to avoid legal consequences!",
    "Worried, Anxious & Scared": "I am really scared and worried that this isn't safe. What if I lose my life savings?",
    "Calm, Formal & Neutral": "The scheduled report has been compiled and is ready for your review at your earliest convenience."
}

# 4. Input Column
with col_input:
    st.subheader("📝 Input Configuration")
    
    # Example selection dropdown
    selected_example = st.selectbox(
        "Choose from predefined examples:",
        options=list(examples.keys()),
        index=0
    )
    
    # Handle text area default value matching selection
    default_text = ""
    if selected_example != "Select an example sentence to test...":
        default_text = examples[selected_example]
        
    user_text = st.text_area(
        "Type or paste custom text below:",
        value=default_text,
        height=150,
        placeholder="Enter your sentence here..."
    )
    
    analyze_button = st.button("🚀 Analyze Behavioral Signals", use_container_width=True)

# 5. Sentiment and Emotion prediction logic & UI
# Initialize session state for history and current prediction
if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# If user clicks analyze
if analyze_button:
    if user_text.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing text..."):
            # Run inference
            sentiment_raw = sentiment_pipe(user_text)[0]
            emotion_raw = emotion_pipe(user_text)[0]
            context_raw = context_pipe(user_text, candidate_labels=CONTEXT_CANDIDATES)
            
            # Format and sort results
            sentiment_scores = sorted(sentiment_raw, key=lambda x: x["score"], reverse=True)
            emotion_scores = sorted(emotion_raw, key=lambda x: x["score"], reverse=True)
            context_scores = sorted(
                [{"label": label, "score": score} for label, score in zip(context_raw["labels"], context_raw["scores"])],
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Get top predictions
            top_sentiment = sentiment_scores[0]
            top_emotion = emotion_scores[0]
            top_context = context_scores[0]
            
            # Create result dictionary
            st.session_state.current_result = {
                "text": user_text,
                "sentiment_label": top_sentiment["label"].upper(),  # standard capitalization style
                "sentiment_score": top_sentiment["score"],
                "emotion_label": top_emotion["label"].capitalize(),
                "emotion_score": top_emotion["score"],
                "context_label": top_context["label"],
                "context_score": top_context["score"],
                "all_sentiment": sentiment_scores,
                "all_emotion": emotion_scores,
                "all_context": context_scores
            }
            
            # Append to session history (avoid duplicate consecutive entries)
            if not st.session_state.history or st.session_state.history[0]["Text Query"] != user_text:
                st.session_state.history.insert(0, {
                    "Text Query": user_text,
                    "Sentiment": f"{top_sentiment['label'].upper()} ({top_sentiment['score']:.1%})",
                    "Emotion": f"{top_emotion['label'].capitalize()} ({top_emotion['score']:.1%})",
                    "Context": f"{top_context['label']} ({top_context['score']:.1%})"
                })

# 6. Results Column
with col_results:
    st.subheader("📊 Model Predictions")
    
    if st.session_state.current_result:
        res = st.session_state.current_result
        
        # Display Metric Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(
                label="Sentiment Polarity",
                value=res["sentiment_label"],
                delta=f"{res['sentiment_score']:.1%} confidence"
            )
        with col_m2:
            st.metric(
                label="Detected Emotion",
                value=res["emotion_label"],
                delta=f"{res['emotion_score']:.1%} confidence"
            )
        with col_m3:
            st.metric(
                label="Context / Intent",
                value=res["context_label"],
                delta=f"{res['context_score']:.1%} confidence"
            )
            
        st.markdown("---")
        
        # Display Probability Distributions using custom clean Plotly charts
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.markdown("##### Sentiment Confidence Scores")
            df_sent = pd.DataFrame(res["all_sentiment"])
            df_sent["label"] = df_sent["label"].str.upper()
            df_sent = df_sent.sort_values(by="score", ascending=True)
            
            # Define specific colors
            colors_map_sent = {"POSITIVE": "#10b981", "NEUTRAL": "#6b7280", "NEGATIVE": "#ef4444"}
            df_sent["color"] = df_sent["label"].map(colors_map_sent)
            
            fig_sent = go.Figure(go.Bar(
                x=df_sent["score"],
                y=df_sent["label"],
                orientation='h',
                marker_color=df_sent["color"],
                text=df_sent["score"].apply(lambda x: f"{x:.1%}"),
                textposition='auto',
                hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>"
            ))
            fig_sent.update_layout(
                margin=dict(l=0, r=0, t=10, b=10),
                height=220,
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", range=[0, 1.05]),
                yaxis=dict(showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sent, use_container_width=True)
            
        with col_c2:
            st.markdown("##### Emotion Confidence Scores")
            df_emot = pd.DataFrame(res["all_emotion"])
            df_emot["label"] = df_emot["label"].str.capitalize()
            df_emot = df_emot.sort_values(by="score", ascending=True)
            
            # Colors for emotion categories
            colors_map_emot = {
                "Joy": "#10b981", "Surprise": "#f59e0b", "Neutral": "#6b7280",
                "Sadness": "#3b82f6", "Anger": "#ef4444", "Fear": "#8b5cf6", "Disgust": "#f97316"
            }
            df_emot["color"] = df_emot["label"].map(lambda x: colors_map_emot.get(x, "#9ca3af"))
            
            fig_emot = go.Figure(go.Bar(
                x=df_emot["score"],
                y=df_emot["label"],
                orientation='h',
                marker_color=df_emot["color"],
                text=df_emot["score"].apply(lambda x: f"{x:.1%}"),
                textposition='auto',
                hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>"
            ))
            fig_emot.update_layout(
                margin=dict(l=0, r=0, t=10, b=10),
                height=220,
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", range=[0, 1.05]),
                yaxis=dict(showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_emot, use_container_width=True)

        with col_c3:
            st.markdown("##### Context Confidence Scores")
            df_cont = pd.DataFrame(res["all_context"])
            # Limit to top 5 context scores for display to avoid overcrowding
            df_cont = df_cont.sort_values(by="score", ascending=True).tail(5)
            
            # Colors for context categories
            colors_map_cont = {
                "Greeting": "#3b82f6",
                "Financial Wire Transfer": "#ef4444",
                "Banking or Account Access": "#f59e0b",
                "Law Enforcement or Legal Threat": "#ec4899",
                "Tech Support": "#8b5cf6",
                "Personal Catch-up": "#10b981",
                "Healthcare Appointment": "#06b6d4",
                "Job or Workplace": "#14b8a6",
                "Romance or Relationship": "#f43f5e",
                "Retail Delivery or Refund": "#f97316",
                "Travel or Logistics": "#84cc16",
                "Education": "#a855f7",
                "Unknown or Other": "#6b7280"
            }
            df_cont["color"] = df_cont["label"].map(lambda x: colors_map_cont.get(x, "#6b7280"))
            
            fig_cont = go.Figure(go.Bar(
                x=df_cont["score"],
                y=df_cont["label"],
                orientation='h',
                marker_color=df_cont["color"],
                text=df_cont["score"].apply(lambda x: f"{x:.1%}"),
                textposition='auto',
                hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>"
            ))
            fig_cont.update_layout(
                margin=dict(l=0, r=0, t=10, b=10),
                height=220,
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", range=[0, 1.05]),
                yaxis=dict(showgrid=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cont, use_container_width=True)
            
    else:
        st.info("👈 Choose an example or write custom text, then click **Analyze Behavioral Signals** to see predictions.")

# 7. Session Query History Section
st.markdown("---")
st.subheader("📜 Session Analysis History")

if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True)
    
    # Button to clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.current_result = None
        st.rerun()
else:
    st.caption("No queries analyzed yet in this session.")
