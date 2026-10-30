import os
import streamlit as st
from openai import OpenAI

from prompts.classifier_prompt import build_ticket_classifier
from structured.parser import parse_and_validate_ticket

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoTicket AI",
    page_icon="🎫",
    layout="centered"
)

# ---------------------------------------------------------------------------
# API Credentials
# ---------------------------------------------------------------------------
api_key = st.secrets.get("LLM_api_key", os.getenv("LLM_api_key", ""))
base_url = "https://api.groq.com/openai/v1"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎫 AutoTicket AI")
st.caption("Intelligent Customer Support Triage & Structured Information Extraction")
st.markdown("---")

# ---------------------------------------------------------------------------
# Model Selection (Allows user to pick active Groq models)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Configuration")
    selected_model = st.selectbox(
        "Select Language Model:",
        [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "llama3-70b-8192",
            "llama3-8b-8192"
        ],
        index=0
    )
    st.info(f"Target API: **Groq Cloud**\nModel: `{selected_model}`")

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------
default_examples = [
    "I was charged twice for the same subscription this month and need an immediate refund!",
    "My laptop screen cracked and I need urgent repair assistance.",
    "I forgot my password and I'm locked out of my account.",
    "My package was supposed to arrive yesterday but still nothing.",
    "Your support agent was incredibly helpful — thank you so much!"
]

selected_example = st.selectbox("💡 Quick Examples:", ["Select an example..."] + default_examples)

initial_text = ""
if selected_example != "Select an example...":
    initial_text = selected_example

ticket_text = st.text_area("📩 Customer Support Ticket:", value=initial_text, height=130, placeholder="Type or paste a customer message here...")

classify_btn = st.button("✨ Classify Ticket", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Inference & Display
# ---------------------------------------------------------------------------
if classify_btn:
    if not ticket_text.strip():
        st.warning("Please enter a ticket message first.")
    elif not api_key:
        st.error("API Key not found. Please set LLM_api_key in Secrets.")
    else:
        with st.spinner("Analyzing and extracting structured data..."):
            try:
                prompt = build_ticket_classifier(ticket_text)
                client = OpenAI(api_key=api_key, base_url=base_url)
                
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You classify support tickets. Return ONLY valid JSON matching the requested schema. No markdown, no explanation."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )
                
                raw_output = response.choices[0].message.content
                result = parse_and_validate_ticket(raw_output)
                
                st.success("✅ Ticket Processed Successfully!")
                
                # Metrics / Highlights
                c1, c2, c3 = st.columns(3)
                
                category_icons = {
                    "technical": "💻 Technical",
                    "account": "🔐 Account",
                    "billing": "💳 Billing",
                    "delivery": "📦 Delivery",
                    "subscription": "🔄 Subscription"
                }
                sentiment_icons = {
                    "positive": "😊 Positive",
                    "negative": "😠 Negative",
                    "neutral": "😐 Neutral"
                }
                urgency_icons = {
                    "high": "🔴 High",
                    "medium": "🟡 Medium",
                    "low": "🟢 Low"
                }
                
                with c1:
                    st.metric(label="Category", value=category_icons.get(result.category, result.category))
                with c2:
                    st.metric(label="Sentiment", value=sentiment_icons.get(result.sentiment, result.sentiment))
                with c3:
                    st.metric(label="Urgency", value=urgency_icons.get(result.urgency, result.urgency))
                
                st.markdown("### 📝 Executive Summary")
                st.info(result.summary)
                
                with st.expander("🔍 View Raw Validated JSON"):
                    st.json({
                        "category": result.category,
                        "sentiment": result.sentiment,
                        "urgency": result.urgency,
                        "summary": result.summary
                    })
                    
            except Exception as e:
                st.error(f"Error during classification: {e}")

st.markdown("---")
st.caption("AutoTicket AI • Developed by Ahmed Yasser • Powered by Groq Cloud")
