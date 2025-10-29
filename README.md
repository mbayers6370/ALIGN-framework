# ALIGN Framework

***Assessing Language Intent & Guidance Neutrality***

ALIGN is an AI evaluation framework that measures how well an AI-generated response aligns with a user’s intent, tone, and contextual expectations.
It doesn’t just check grammar or coherence, it evaluates understanding.

Built for single-turn or multi-turn evaluation, ALIGN can analyze individual interactions entered manually, or process large-scale JSON datasets containing hundreds of user–AI exchanges. It dissects every conversation across five pillars of conversational quality, combining semantic similarity, zero-shot NLI reasoning, and emotion-aware modeling into one interpretable composite score.

---

## Why it Matters

Modern AI models sound fluent but often miss the point. ALIGN bridges that gap by quantifying how human-centered a response feels, not just if it’s correct.
It’s designed for teams building, tuning, or evaluating conversational agents that need to go beyond accuracy into alignment and empathy.

---

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://align-framework.streamlit.app/)

Try the full ALIGN Framework live here:  
**https://align-framework.streamlit.app/**  

*(First run may take a moment while models download.)*

---

## Features

### Five-Pillar Evaluation System
- **Intent Matching**
- **Relevance**
- **Clarity**
- **Tone Match**
- **Engagement**

### Visual Score Breakdown
- Streamlit-powered GUI with intuitive horizontal bar displays
- Final score averaged and presented out of 10 (with decimal precision)

### Lightweight & Modular
- Single-file scoring logic for rapid iteration
- Easy to integrate or extend into larger evaluation pipelines

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/mbayers6370/ALIGN-framework.git
cd ALIGN-framework

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run the App
```bash
streamlit run align.py
```

Then open the local URL displayed in your terminal (usually http://localhost:8501).

To deploy it on **Streamlit Cloud**:
1. Push this repo to GitHub (which you already have).
2. Visit [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select this repository.
4. Choose `align.py` as the entry point.
5. Streamlit will automatically install dependencies and host the live app.

---

## Scoring Logic

The **final score** is a simple average of five individual scores:
- Each dimension is scored from **0 to 10**
- Final score is **rounded to the nearest tenth**
- No category is weighted: it's honest, balanced evaluation

---

## Vision

ALIGN is a step toward evaluating AI not just by grammar or coherence, but by **human-centered understanding**: how well a model responds with empathy, emotional alignment, and conversational momentum.

We believe the future of AI evaluation is as much about **emotional resonance** as it is about factual relevance.

---

## License

MIT License. Free to use, remix, and build upon.

---

## Author

Developed by **Matthew Bayers**
