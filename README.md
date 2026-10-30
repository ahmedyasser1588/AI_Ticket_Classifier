---
title: AutoTicket AI — Production Support Classifier
emoji: 🎫
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Production-grade ticket classification using Groq & Llama 3.1 with guaranteed structured output.
---

<div align="center">

# 🎫 AutoTicket AI
### Production-Grade Customer Support Triage & Structured Extraction Engine

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/Validation-Pydantic%20v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Groq Llama 3.1](https://img.shields.io/badge/Inference-Groq%20Llama%203.1-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![HuggingFace Spaces](https://img.shields.io/badge/Live_Demo-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](#-live-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Transform raw, messy support tickets into validated, actionable structured data instantly.</b><br>
  Built with schema enforcement, self-healing retries, and comprehensive model evaluation.
</p>

[Live Demo](#-live-demo) • [Architecture](#-system-architecture) • [Key Features](#-key-features) • [Quick Start](#-quick-start) • [Evaluation](#-evaluation-framework)

---

</div>

## 📌 Executive Summary

Modern support organizations receive thousands of unstructured messages daily. Manual routing causes delayed response times, incorrect department assignments, and SLA breaches.

**AutoTicket AI** solves this by combining modern LLM reasoning with strict software engineering patterns:
- Extracts **Category**, **Sentiment**, and **Urgency** in a strictly validated schema.
- Provides a concise **1-sentence Executive Summary** for human agents.
- Delivers deterministic, type-safe outputs using **Pydantic v2** and JSON regex fallback extraction.
- Encapsulates network operations in a **Resilient Gateway** with jittered exponential backoff.

---

## 🚀 Live Demo

Try the interactive UI on HuggingFace Spaces:
👉 **[Open Live Space](https://huggingface.co/spaces/ahmedyasser1588/ticket-classifier)** *(Deploy with one click)*

```
Input: "I was charged twice for the same transaction and I need an immediate refund!"
Output:
{
  "category": "billing",
  "sentiment": "negative",
  "urgency": "high",
  "summary": "Customer was double-charged and requests an immediate refund."
}
```

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([Incoming Customer Message]) --> PromptBuilder[Prompt Engineering Layer\n- Category/Urgency taxonomy\n- Few-shot exemplar steering]
    PromptBuilder --> LLMGateway[Resilient LLM Gateway\n- Exponential Backoff\n- Decorator Pattern\n- Custom Exceptions]
    
    subgraph Engine [Inference Pipeline]
        LLMGateway --> ModelAPI[Groq API / Llama-3.1-8b-instant\nUltra-low latency generation]
    end
    
    ModelAPI --> Parser[Robust Parser Layer\n- Markdown strip\n- Regex JSON isolate\n- Pydantic schema validation]
    
    Parser --> ValidatedSchema[Type-Safe TicketSchema Object\ncategory | sentiment | urgency | summary]
    
    ValidatedSchema --> GradioApp[Gradio Interactive Web UI]
    ValidatedSchema --> EvalSuite[Offline Evaluation Benchmark]
```

---

## ✨ Key Features

- **🛡️ Guaranteed Schema Integrity:** Eliminates hallucinated fields and Markdown pollution using automated extraction and Pydantic validation.
- **⚡ Sub-Second Latency:** Leverages Groq's LPU acceleration with Llama 3.1 8B for near real-time routing.
- **🔄 Fault-Tolerant Gateway:** Custom exception hierarchy (`AuthError`, `RateLimitError`, `ConnectionError`) wrapping LLM calls with configurable retry loops.
- **📊 Quantitative Evaluation Framework:** Integrated batch evaluation testing against ground-truth datasets with 6 core metrics:
  - *Validity Rate* (Schema adherence)
  - *Category Accuracy*
  - *Sentiment Accuracy*
  - *Urgency Accuracy*
  - *Overall Success Rate*
  - *Failure Count*

---

## 📂 Repository Structure

```plaintext
├── app.py                     # Production Gradio Web Application
├── prompts/
│   └── classifier_prompt.py   # Few-shot prompt builder with domain taxonomy
├── structured/
│   ├── schema.py              # Pydantic TicketSchema (Literal types)
│   ├── parser.py              # Robust parser (Markdown stripper & Regex fallback)
│   ├── generator.py           # Outlines transformer generation adapter
│   └── outline_adapter.py     # Version-agnostic Outlines bridge
├── LLM/
│   ├── base.py                # Abstract LLM Interface
│   ├── openai.py              # OpenAI-compatible API client
│   ├── gateway.py             # Resilient gateway with retry logic
│   ├── config.py              # Pydantic BaseSettings environment config
│   ├── exceptions.py          # Custom domain exceptions
│   ├── Model.py               # GenerationRequest & Message dataclasses
│   └── ticket_classifier.py   # Core triage domain service
├── evaluation/
│   ├── evaluator.py           # Evaluation pipeline iterating over CSV data
│   ├── metrics.py             # Accuracy & validity metric calculation functions
│   └── models.py              # EvaluationModel dataclass
├── Data/
│   └── tickets.csv            # Support ticket benchmarking dataset
├── requirements_deployment.txt # Minimal deployment dependencies
└── .env.example               # Environment variables template
```

---

## 🚦 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/ahmedyasser1588/AI_Ticket_Classifier.git
cd AI_Ticket_Classifier

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements_deployment.txt
```

### 2. Configure Environment Variables

Create your `.env` file using the provided template:

```bash
cp .env.example .env
```

Add your free [Groq API Key](https://console.groq.com):

```ini
LLM_api_key=gsk_your_api_key_here
LLM_base_url=https://api.groq.com/openai/v1
LLM_Model=llama-3.1-8b-instant
```

### 3. Launch Local Demo

```bash
python app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

---

## 📊 Evaluation Framework

The project includes an offline benchmark suite to calculate accuracy against labeled datasets:

```python
import pandas as pd
from evaluation.evaluator import TicketEvaluator
from evaluation.metrics import metric_evaluation

# Load dataset
df = pd.read_csv("Data/tickets.csv")

# Run batch evaluation
evaluator = TicketEvaluator(generator)
results = evaluator.evaluate(df)

# Print metrics dictionary
metrics = metric_evaluation(results)
print(metrics)
```

Example Benchmark Metrics:
- **Validity Rate:** 100% (Zero schema failures)
- **Category Accuracy:** ~90-95%
- **Sentiment Accuracy:** ~90%
- **Urgency Accuracy:** ~85-90%

---

## 🛠️ Tech Stack & Design Patterns

| Area | Tools & Patterns |
|---|---|
| **Core Models** | `Llama-3.1-8b-instant`, `Transformers` |
| **Validation & Schema** | `Pydantic v2`, `Outlines` |
| **Design Patterns** | Decorator Pattern (`LLMGateWay`), Factory Pattern, Strategy Pattern |
| **API Integration** | `OpenAI SDK` over `Groq Cloud` |
| **Web UI** | `Gradio 4` |
| **Environment Management** | `pydantic-settings`, `python-dotenv` |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Developed by <a href="https://github.com/ahmedyasser1588"><b>Ahmed Yasser</b></a> • Portfolio Showcase
</div>
