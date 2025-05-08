# ALIGN Framework

**ALIGN** is an AI alignment evaluation framework designed to assess how well an AI-generated response aligns with a user's intent, emotional tone, and contextual expectations. Built for single-turn interactions, ALIGN offers a structured breakdown across six key dimensions of conversational quality.

---

## Live Demo

> Coming soon on [Streamlit Cloud](#) – Stay tuned.

---

## Features

- **Six-Pillar Evaluation System**
  - Intent Matching
  - Relevance
  - Completeness
  - Clarity
  - Adaptability
  - Engagement

- **Visual Score Breakdown**
  - Streamlit-powered GUI with intuitive horizontal bar display
  - Final score averaged and presented out of 10 (with decimal precision)

- **Single File Simplicity**
  - Lightweight and easy to integrate
  - Self-contained scoring logic for rapid iteration

---

## ⚙ Installation

```bash
git clone https://github.com/mbayers6370/ALIGN-framework.git
cd ALIGN-framework
python -m venv align-env
source align-env/bin/activate  # On Windows: .\align-env\Scripts\activate
pip install -r requirements.txt
streamlit run align.py

---

Usage
	1.	Launch the Streamlit app:
		streamlit run align.py
	2.	Input a user message and a proposed AI response.
	3.	Click “Evaluate Response” to view alignment scores and the final score.

---

Scoring Logic

Final Score is calculated as a simple average of six individual scores:
	•	Each category is scored on a scale of 0–10
	•	Final score is rounded to the nearest tenth
	•	No weights or penalties—just honest math

⸻

Vision

ALIGN aims to set a new standard for evaluating AI-generated dialogue—not just by fluency, but by empathy, relevance, and intent. It’s a first step toward emotionally intelligent AI evaluation.

⸻

License

MIT License. Free to use, remix, and build upon.

⸻

Author

Developed by Matthew Bayers
