"""
AI Ticket Classifier — Gradio App
Deployment entry point for HuggingFace Spaces.

Imports from the existing project modules (prompts + structured)
and calls Groq/OpenAI-compatible API directly for reliability.
"""

import os
import sys
import json

import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path so local packages resolve correctly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from prompts.classifier_prompt import build_ticket_classifier
from structured.parser import parse_and_validate_ticket

# ---------------------------------------------------------------------------
# Configuration — read from environment (HuggingFace Secrets) or .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required on HuggingFace Spaces

API_KEY  = os.getenv("LLM_api_key", "")
BASE_URL = "https://api.groq.com/openai/v1"
MODEL    = "qwen/qwen3.8-27b"

# ---------------------------------------------------------------------------
# Core Classification Logic
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Call the LLM API and return raw text output."""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify support tickets. "
                    "Return ONLY valid JSON matching the requested schema. "
                    "No markdown, no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content


def classify_ticket(ticket_text: str):
    """
    Full pipeline:  ticket_text → prompt → LLM → parse → structured result.
    Returns individual field values for Gradio outputs.
    """
    ticket_text = ticket_text.strip()
    if not ticket_text:
        return "—", "—", "—", "—"

    try:
        prompt     = build_ticket_classifier(ticket_text)
        raw_output = _call_llm(prompt)
        result     = parse_and_validate_ticket(raw_output)

        # Emoji maps for visual polish
        category_emoji = {
            "technical":    "💻 Technical",
            "account":      "🔐 Account",
            "billing":      "💳 Billing",
            "delivery":     "📦 Delivery",
            "subscription": "🔄 Subscription",
        }
        sentiment_emoji = {
            "positive": "😊 Positive",
            "negative": "😠 Negative",
            "neutral":  "😐 Neutral",
        }
        urgency_emoji = {
            "high":   "🔴 High",
            "medium": "🟡 Medium",
            "low":    "🟢 Low",
        }

        return (
            category_emoji.get(result.category, result.category),
            sentiment_emoji.get(result.sentiment, result.sentiment),
            urgency_emoji.get(result.urgency, result.urgency),
            result.summary,
        )

    except Exception as exc:
        return f"Error: {exc}", "—", "—", "—"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["My laptop screen cracked and I need urgent repair assistance."],
    ["I was charged twice for the same subscription this month!"],
    ["I forgot my password and I'm locked out of my account."],
    ["My package was supposed to arrive yesterday but still nothing."],
    ["Your support agent was incredibly helpful — thank you so much!"],
    ["The mobile app crashes every time I try to upload a file."],
    ["I'd like to upgrade my plan to the premium tier."],
]

CSS = """
/* ─── Global ─── */
body { font-family: 'Inter', sans-serif; }

/* ─── Title ─── */
#title-row {
    text-align: center;
    padding: 8px 0 4px;
}
#app-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#app-subtitle {
    color: #6b7280;
    font-size: 0.95rem;
    margin-top: 4px;
}

/* ─── Input box ─── */
#ticket-input textarea {
    font-size: 1rem;
    border-radius: 10px;
    border: 1.5px solid #e5e7eb;
    transition: border-color 0.2s;
}
#ticket-input textarea:focus {
    border-color: #8b5cf6;
    box-shadow: 0 0 0 3px rgba(139,92,246,.15);
}

/* ─── Result cards ─── */
.result-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: #6b7280;
    margin-bottom: 2px;
}
#out-category input, #out-sentiment input,
#out-urgency input,  #out-summary input {
    font-size: 1.05rem;
    font-weight: 600;
    border-radius: 8px;
    border: 1.5px solid #e5e7eb;
    background: #fafafa;
    padding: 10px 14px;
}

/* ─── Classify button ─── */
#classify-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    border-radius: 10px;
    border: none;
    padding: 12px 28px;
    cursor: pointer;
    transition: opacity .2s, transform .1s;
}
#classify-btn:hover { opacity: .9; transform: translateY(-1px); }

/* ─── Examples ─── */
.examples-header { color: #6b7280; font-size: 0.85rem; }
"""

with gr.Blocks(css=CSS, title="AI Ticket Classifier") as demo:

    # ── Header ────────────────────────────────────────────────────────────
    with gr.Row(elem_id="title-row"):
        with gr.Column():
            gr.HTML("""
                <div id="app-title">🎫 AI Ticket Classifier</div>
                <div id="app-subtitle">
                    Instantly classify customer support tickets using LLM&nbsp;—&nbsp;
                    powered by <strong>Groq</strong> &amp; <strong>Llama&nbsp;3.1</strong>
                </div>
            """)

    gr.HTML("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0 0 16px'>")

    # ── Input ─────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            ticket_input = gr.Textbox(
                label="📩 Support Ticket",
                placeholder="Paste or type a customer support ticket here…",
                lines=5,
                elem_id="ticket-input",
            )
            classify_btn = gr.Button(
                "✨ Classify Ticket",
                variant="primary",
                elem_id="classify-btn",
            )

    # ── Outputs ───────────────────────────────────────────────────────────
    with gr.Row():
        out_category = gr.Textbox(
            label="🏷️ Category",
            interactive=False,
            elem_id="out-category",
        )
        out_sentiment = gr.Textbox(
            label="💬 Sentiment",
            interactive=False,
            elem_id="out-sentiment",
        )
        out_urgency = gr.Textbox(
            label="🚨 Urgency",
            interactive=False,
            elem_id="out-urgency",
        )

    out_summary = gr.Textbox(
        label="📝 AI Summary",
        interactive=False,
        elem_id="out-summary",
    )

    # ── Examples ──────────────────────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLES,
        inputs=ticket_input,
        label="💡 Try an example",
        examples_per_page=4,
    )

    # ── How it works ──────────────────────────────────────────────────────
    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown("""
### Pipeline

```
Customer Ticket
      │
      ▼
 Prompt Builder          ← prompts/classifier_prompt.py
      │   (few-shot examples + structured instructions)
      ▼
  Groq LLM API           ← llama-3.1-8b-instant  (free & fast)
      │   (returns raw JSON)
      ▼
  JSON Parser            ← structured/parser.py
      │   (strips markdown, regex fallback, Pydantic validation)
      ▼
  TicketSchema           ← structured/schema.py
   category · sentiment · urgency · summary
```

### Output Fields

| Field | Options |
|---|---|
| **Category** | technical · account · billing · delivery · subscription |
| **Sentiment** | positive · negative · neutral |
| **Urgency** | high · medium · low |
| **Summary** | One-sentence AI-generated description |

### Tech Stack
`Python` · `Pydantic v2` · `Groq API` · `Llama 3.1 8B` · `Gradio` · `Outlines`
        """)

    # ── Footer ────────────────────────────────────────────────────────────
    gr.HTML("""
        <div style="text-align:center;color:#9ca3af;font-size:.8rem;margin-top:24px;padding-bottom:8px;">
            Built with ❤️ · Powered by Groq &amp; Llama 3.1 ·
            <a href="https://github.com" style="color:#8b5cf6">GitHub</a>
        </div>
    """)

    # ── Wiring ────────────────────────────────────────────────────────────
    classify_btn.click(
        fn=classify_ticket,
        inputs=ticket_input,
        outputs=[out_category, out_sentiment, out_urgency, out_summary],
    )
    ticket_input.submit(
        fn=classify_ticket,
        inputs=ticket_input,
        outputs=[out_category, out_sentiment, out_urgency, out_summary],
    )


if __name__ == "__main__":
    demo.launch()
