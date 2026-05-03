# 🧠 LLM Hallucination & Bias Live-Tracker (V2)

An automated MLOps evaluation pipeline that tracks hallucination rates, bias, and evasiveness of State-of-the-Art LLMs (like GPT-4o-mini and Claude 3.5 Sonnet) on **live, ambiguous breaking news**.

## 🚀 Why This Exists (The "RAG Failure" Problem)
Modern LLMs are excellent at coding and historical facts, but how do they handle the *edge of their knowledge*? When Enterprise RAG systems fail to retrieve context, does the model safely say "I don't know", or does it confidently hallucinate fake metrics, quotes, and policies?

This project simulates that exact edge-case daily. It feeds models just the **headline** of today's breaking news across 5 categories and uses an LLM-as-a-judge to grade their calibration, overconfidence, and bias compared to the full article.

## ✨ V2 Architecture Features
- **Smart News Ingestion:** Uses NewsAPI to fetch daily headlines across Technology, Science, Politics, Business, and Sports.
- **Pre-processing Opinion Filter:** Uses an LLM to automatically categorize and discard opinion pieces/editorials to ensure a clean Ground Truth.
- **Strict Parametric Testing:** Forces models to rely purely on their parametric memory (no search tools enabled, temperature=0.0) to test safety boundaries.
- **LLM-as-a-Judge:** Uses `gpt-4o` with `instructor` and `Pydantic` for strict JSON evaluation rubrics (Factuality, Hallucination, Bias, Evasiveness, Overconfidence).
- **Exact Model Tracking:** Captures specific API model versions (e.g., `gpt-4o-mini-2024-07-18`) to monitor "shadow updates" over time.
- **Live Streamlit Dashboard:** An interactive UI to visualize hallucination drift and bias across different domains.

## 🛠️ Setup & Running Locally

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set your API Keys in `.env`:
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
NEWSAPI_KEY=your_key...
```
*(Note: If API keys are missing, the project will automatically fall back to a Mock Mode to generate dummy data for testing the dashboard!)*

3. Run the evaluation engine:
```bash
python main.py
```

4. Launch the dashboard:
```bash
streamlit run dashboard.py
```

## 📊 Dashboard Capabilities
The `dashboard.py` file provides a local web app that includes:
- Category-specific filtering (Politics vs. Science hallucination rates).
- Overconfidence tracking ("Who is most confidently wrong?").
- Bias detection breakdowns (Political framing, cultural bias, etc.).
