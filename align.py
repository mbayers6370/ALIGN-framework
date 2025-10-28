import streamlit as st
import numpy as np
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

@st.cache_resource(show_spinner="Downloading intent model (first run only)…")
def load_intent_classifier():
    from InstructorEmbedding import INSTRUCTOR
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model = INSTRUCTOR("hkunlp/instructor-base")
    try:
        model.to(device)
    except Exception:
        # Some SentenceTransformers backends ignore .to(); safe to continue.
        pass
    return model


@st.cache_resource(show_spinner="Loading embedding backend…")
def load_embed_model():
    """
    Prefer TF-Hub USE if it's actually available (local dev),
    otherwise fall back to Sentence-Transformers on CPU.
    Returns: encode(List[str]) -> np.ndarray [N, D]
    """
    # Try TF-Hub only if present (not installed on Streamlit Cloud by default)
    try:
        import importlib
        if importlib.util.find_spec("tensorflow_hub") is not None:
            import tensorflow_hub as hub
            use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            def encode(texts):
                import numpy as np
                return np.asarray(use(texts))
            return encode
    except Exception:
        pass  # fall through to Sentence-Transformers fallback

    # Fallback: small, fast, CPU-friendly sentence-transformer
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    def encode(texts):
        import numpy as np
        return np.asarray(
            st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )
    return encode

@st.cache_resource
def load_emotion_model():
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    return tokenizer, model

# === Zero-shot NLI model loader ===
@st.cache_resource(show_spinner="Loading zero-shot NLI model…")
def load_nli_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    return tokenizer, model

# === Cross-encoder for relevance scoring ===
@st.cache_resource(show_spinner="Loading cross-encoder for relevance…")
def load_cross_encoder():
    # Uses SentenceTransformers CrossEncoder trained on MS MARCO passage ranking
    from sentence_transformers import CrossEncoder
    try:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        # Fallback to a slightly larger but compatible variant
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return model

# Five-label zero-shot intent/response categories (no OTHER)
ZERO_SHOT_LABELS = ["FACTOID", "HOW_TO", "PLANNING", "EMOTIONAL", "OPINION"]
USER_HYP = "The user's message is a {} request."
RESP_HYP = "The assistant's reply is written in a {} style."

# ===== Intent taxonomy (keyword-free) =====
INTENT_LABELS = ["FACTOID", "HOW_TO", "PLANNING", "EMOTIONAL", "OPINION", "OTHER"]

USER_PROTOS = {
    "FACTOID": [
        "Represent the user's underlying task category: requesting a specific factual answer.",
        "Characterize the user's goal: obtain one concrete, verifiable fact.",
        "Summarize the user's communicative intent as a fact-seeking request."
    ],
    "HOW_TO": [
        "Represent the user's underlying task category: requesting step-by-step instructions.",
        "Characterize the user's goal: learn actionable steps to accomplish a task.",
        "Summarize the user's communicative intent as a procedure-seeking request."
    ],
    "PLANNING": [
        "Represent the user's underlying task category: seeking a plan with options and tradeoffs.",
        "Characterize the user's goal: explore alternatives and make a decision.",
        "Summarize the user's communicative intent as planning/decision support."
    ],
    "EMOTIONAL": [
        "Represent the user's underlying task category: seeking empathy and emotional support.",
        "Characterize the user's goal: be validated and emotionally supported.",
        "Summarize the user's communicative intent as an emotional-support request."
    ],
    "OPINION": [
        "Represent the user's underlying task category: seeking a reasoned perspective or argument.",
        "Characterize the user's goal: hear a viewpoint with pros and cons.",
        "Summarize the user's communicative intent as opinion/perspective seeking."
    ],
    "OTHER": [
        "Represent the user's underlying task category: small talk, meta-chat, or unclear request.",
        "Characterize the user's goal: casual conversation or ambiguous intent.",
        "Summarize the user's communicative intent as other/unclear."
    ]
}

RESPONSE_PROTOS = {
    "FACTOID": [
        "Represent a concise factual answer to the user's request.",
        "Characterize a reply that states one clear, verifiable fact.",
        "Summarize a response that directly answers with a concrete fact."
    ],
    "HOW_TO": [
        "Represent a step-by-step, actionable set of instructions.",
        "Characterize a reply that teaches a procedure with clear steps.",
        "Summarize a response that guides execution with numbered actions."
    ],
    "PLANNING": [
        "Represent a structured plan exploring options with tradeoffs.",
        "Characterize a reply that lays out alternatives and makes recommendations.",
        "Summarize a response that frames decisions with pros and cons."
    ],
    "EMOTIONAL": [
        "Represent an empathetic, validating, supportive reply.",
        "Characterize a response that acknowledges feelings and offers support.",
        "Summarize a reply that prioritizes empathy and emotional validation."
    ],
    "OPINION": [
        "Represent a reasoned perspective weighing pros and cons.",
        "Characterize a reply that offers a thoughtful viewpoint with rationale.",
        "Summarize a response that argues a position with reasons."
    ],
    "OTHER": [
        "Represent a brief, socially appropriate small-talk reply.",
        "Characterize a response that is meta, chit-chat, or neutral filler.",
        "Summarize a reply that does not attempt to answer the task."
    ]
}

# Averaging encoder helper for prototype stabilization
def _avg_encode_list(inst, prefix, lines):
    """Encode multiple phrasings and return a single averaged, L2-normalized vector."""
    vecs = inst.encode([[prefix, s] for s in lines])
    vecs = np.array(vecs, dtype=float)
    mean = vecs.mean(axis=0)
    norm = np.linalg.norm(mean) + 1e-9
    return mean / norm

# Alignment matrix: rows=user intent, cols=response form → base score 0..10
ALIGNMENT_MATRIX = {
    "FACTOID":  {"FACTOID":10,"HOW_TO":8,"PLANNING":6,"EMOTIONAL":4,"OPINION":2,"OTHER":0},
    "HOW_TO":   {"FACTOID":6,"HOW_TO":10,"PLANNING":8,"EMOTIONAL":2,"OPINION":4,"OTHER":0},
    "PLANNING": {"FACTOID":2,"HOW_TO":8,"PLANNING":10,"EMOTIONAL":4,"OPINION":6,"OTHER":0},
    "EMOTIONAL":{"FACTOID":2,"HOW_TO":4,"PLANNING":8,"EMOTIONAL":10,"OPINION":6,"OTHER":0},
    "OPINION":  {"FACTOID":2,"HOW_TO":6,"PLANNING":8,"EMOTIONAL":4,"OPINION":10,"OTHER":0},
    "OTHER":    {"FACTOID":0,"HOW_TO":4,"PLANNING":2,"EMOTIONAL":8,"OPINION":6,"OTHER":10},
}

@st.cache_resource
def _intent_proto_vectors():
    inst = load_intent_classifier()
    U = []
    R = []
    for label in INTENT_LABELS:
        U.append(_avg_encode_list(inst, "Encode the user intent category", USER_PROTOS[label]))
        R.append(_avg_encode_list(inst, "Encode the response style category", RESPONSE_PROTOS[label]))
    return np.vstack(U), np.vstack(R)

def _softmax(x):
    x = np.array(x, dtype=float)
    x = x - x.max()
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-9)

def _entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

def _cos(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-9) * (np.linalg.norm(b)+1e-9)))

# === Relevance keyword/entity coverage, refusal, and NLI helpers ===
STOPWORDS = {"the","a","an","and","or","but","if","then","than","that","this","those","these","to","of","in","on","for","with","by","from","at","as","it","is","are","be","was","were","i","you","he","she","we","they","them","me","my","your","our","their"}

def _simple_terms(text: str):
    # crude content term extractor: alphanumerics >= 3 chars, stopword filtered, light stemming
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", text.lower())
    terms = []
    for t in toks:
        if t in STOPWORDS:
            continue
        # light stem/normalize endings
        t2 = re.sub(r"(ing|ed|es|s)$", "", t)
        if len(t2) >= 3 and t2 not in STOPWORDS:
            terms.append(t2)
    return terms

def _keyword_coverage(user_text: str, resp_text: str) -> float:
    ut = set(_simple_terms(user_text))
    if not ut:
        return 0.0
    rt = set(_simple_terms(resp_text))
    hit = len(ut & rt)
    return hit / max(1, len(ut))

def _detect_refusal(resp_text: str) -> bool:
    t = resp_text.strip().lower()
    patterns = [
        r"\b(i\s+can't|i\s+cannot|i\s+won't|unable\s+to)\b",
        r"\b(i\s+don't\s+know|not\s+sure)\b",
        r"\b(as\s+an\s+ai|i\s+am\s+an\s+ai)\b",
        r"\b(i\s+cannot\s+provide\s+that|no\s+answer)\b",
    ]
    return any(re.search(p, t) for p in patterns)

def _nli_entails_contra(user_text: str, resp_text: str):
    tokenizer, model = load_nli_model()
    # Entailment: response entails user's request/statement
    inputs = tokenizer(user_text, resp_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1)
    entail = float(probs[2].cpu().numpy())
    contra = float(probs[0].cpu().numpy())
    return entail, contra

# === Zero-shot NLI + heuristics utilities ===
def _nli_entailment_prob(premise: str, hypothesis: str, tokenizer, model) -> float:
    """Return entailment probability P(entailment|premise,hypothesis) for MNLI models."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    # BART/RoBERTa MNLI label order: [contradiction, neutral, entailment]
    probs = F.softmax(logits, dim=-1)
    return float(probs[2].cpu().numpy())

def _heuristic_nudges(text: str, role: str):
    """Small, conservative logit nudges based on obvious surface cues."""
    t = text.lower().strip()
    nudges = {k: 0.0 for k in ZERO_SHOT_LABELS}
    # Question pattern: starts with interrogative and ends with '?'
    if re.match(r"^(how|why|when|where|what|who|which)\b", t) and t.endswith("?"):
        if t.startswith("how"):
            nudges["HOW_TO"] += 0.25
        else:
            nudges["FACTOID"] += 0.20
    # How-to cues
    if re.search(r"\b(step|steps|guide|tutorial|walkthrough|how to)\b", t):
        nudges["HOW_TO"] += 0.15
    # Planning/decision cues
    if re.search(r"\b(pros and cons|vs\.|versus|should i|recommend|plan|itinerary|roadmap|compare)\b", t):
        nudges["PLANNING"] += 0.18
    # Opinion cues
    if re.search(r"\b(opinion|what do you think|your take|argue|debate)\b", t):
        nudges["OPINION"] += 0.15
    # Emotional cues
    if re.search(r"\b(anxious|overwhelmed|sad|lonely|support|comfort|afraid|worried|depressed|hurt|grieving|angry|frustrated|panic)\b", t):
        nudges["EMOTIONAL"] += 0.22
    return nudges

def _nli_label_scores(text: str, role: str):
    """Compute per-label probabilities via MNLI entailment + heuristic logit nudges.
    Returns: (label:str, margin:float, probs:np.ndarray)
    """
    tokenizer, model = load_nli_model()
    hypothesis_tmpl = USER_HYP if role == "user" else RESP_HYP

    # Collect raw entailment logits for each label to combine with nudges
    raw_logits = []
    for label in ZERO_SHOT_LABELS:
        hyp = hypothesis_tmpl.format(label.replace("_", " ").lower())
        inputs = tokenizer(text, hyp, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**inputs).logits[0]
        entail_logit = out[2].item()  # take entailment logit directly
        raw_logits.append(entail_logit)

    # Apply heuristic nudges in logit space (acts like a bias)
    nudges = _heuristic_nudges(text, role)
    logits = [raw_logits[i] + nudges[ZERO_SHOT_LABELS[i]] for i in range(len(ZERO_SHOT_LABELS))]

    # Softmax over adjusted logits to get a normalized distribution
    exps = np.exp(np.array(logits) - np.max(logits))
    probs = exps / (exps.sum() + 1e-9)

    idx = int(np.argmax(probs))
    # Confidence as top1 - top2 margin (more stable than entropy for small K)
    top2 = sorted(probs, reverse=True)[:2]
    margin = float(top2[0] - top2[1]) if len(top2) == 2 else float(probs[idx])

    label = ZERO_SHOT_LABELS[idx]
    return label, margin, probs

# Thin wrappers for clarity
def classify_intent_user_zs(user_text: str):
    return _nli_label_scores(user_text, role="user")  # (label, margin, probs)

def classify_response_form_zs(ai_text: str):
    label, margin, _ = _nli_label_scores(ai_text, role="assistant")
    return label, margin


# Emotion-aware gating helper for user text
def _user_emotion_intensity(text):
    """Return 0..1 intensity using the existing emotion model (no keywords)."""
    tokenizer, model = load_emotion_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    # Treat non-neutral emotions as signal; cap gently
    label_map = model.config.id2label
    neutral_idx = [k for k,v in label_map.items() if v.lower()=="neutral"]
    neutral_p = probs[neutral_idx[0]] if neutral_idx else 0.0
    intensity = float(1.0 - neutral_p)  # higher when not neutral
    return max(0.0, min(1.0, intensity))

def classify_intent_user(user_text):
    """Return (label, confidence, user_task_vec) using prototype similarity (no keywords), with emotion-aware gating."""
    inst = load_intent_classifier()
    U, _ = _intent_proto_vectors()
    uvec = inst.encode([["Represent the user's underlying task category for this text.", user_text]])[0]
    # Cosines to each user prototype
    sims = [ _cos(uvec, U[i]) for i in range(len(INTENT_LABELS)) ]
    # Emotion gating: if the text is near-neutral, softly reduce EMOTIONAL similarity
    intensity = _user_emotion_intensity(user_text)
    emo_idx = INTENT_LABELS.index("EMOTIONAL")
    sims[emo_idx] *= (0.6 + 0.4 * intensity)  # 0.6..1.0 scale

    probs = _softmax(sims)
    conf_raw = 1.0 - _entropy(probs) / np.log(len(INTENT_LABELS))
    idx = int(np.argmax(probs))
    label = INTENT_LABELS[idx]
    # Low-confidence routing to OTHER
    TAU = 0.45
    if conf_raw < TAU:
        label = "OTHER"
    return label, float(conf_raw), uvec

def classify_response_form(ai_text):
    """Return (label, response_vec) for the assistant reply form using averaged prototypes."""
    inst = load_intent_classifier()
    _, R = _intent_proto_vectors()
    avec = inst.encode([["Represent the assistant response meaningfully.", ai_text]])[0]
    sims = [ _cos(avec, R[i]) for i in range(len(INTENT_LABELS)) ]
    idx = int(np.argmax(sims))
    label = INTENT_LABELS[idx]
    return label, avec

def score_intent_alignment(user_text, ai_text):
    """
    Zero-shot NLI version (5 labels, no OTHER exposed).
    - Classify user intent and response style via MNLI entailment with five labels.
    - Use a small set of heuristics as logit biases.
    - Score via a reduced alignment matrix blended with semantic coherence.
    Returns: (score:float, user_label:str|"UNCERTAIN", resp_label:str, user_conf:float)
    """
    # Classify
    user_label, user_margin, _ = _nli_label_scores(user_text, role="user")
    resp_label, _ = classify_response_form_zs(ai_text)

    # Abstain threshold for uncertainty (hide OTHER; treat as UNCERTAIN)
    UNCERTAIN_THRESH = 0.10
    is_uncertain = user_margin < UNCERTAIN_THRESH

    # Coherence from general sentence embeddings (orthogonal to category)
    embed = load_embed_model()
    uvec, avec = embed([user_text, ai_text])
    coh = _cos(uvec, avec)
    to01 = lambda x: 0.5 * (x + 1.0)
    coh01 = to01(coh)

    # Five-label alignment matrix (subset of your original, without OTHER)
    ALIGN5 = {
        "FACTOID":  {"FACTOID":10, "HOW_TO":8, "PLANNING":6, "EMOTIONAL":4, "OPINION":2},
        "HOW_TO":   {"FACTOID":6,  "HOW_TO":10, "PLANNING":8, "EMOTIONAL":2, "OPINION":4},
        "PLANNING": {"FACTOID":2,  "HOW_TO":8,  "PLANNING":10, "EMOTIONAL":4, "OPINION":6},
        "EMOTIONAL":{"FACTOID":2,  "HOW_TO":4,  "PLANNING":8,  "EMOTIONAL":10, "OPINION":6},
        "OPINION":  {"FACTOID":2,  "HOW_TO":6,  "PLANNING":8,  "EMOTIONAL":4,  "OPINION":10},
    }

    if is_uncertain:
        # Neutral baseline when we abstain on user intent; blend with coherence
        base = 6.0
        score = 0.5 * base + 0.5 * (10.0 * coh01)
        final = float(round(max(0.0, min(10.0, score)), 1))
        return final, "UNCERTAIN", resp_label, float(round(user_margin, 3))
    else:
        base = ALIGN5[user_label][resp_label]
        # Blend base with semantic coherence; weight by user-margin confidence
        score = (0.8 * base + 0.2 * (10.0 * coh01)) * (0.7 + 0.3 * min(1.0, user_margin * 12.0))
        final = float(round(max(0.0, min(10.0, score)), 1))
        return final, user_label, resp_label, float(round(user_margin, 3))


def classify_function(message, model, intent_labels):
    inputs = [[f"Represent a {label}", message] for label in intent_labels]
    embeddings = model.encode(inputs)
    sims = sk_cosine_similarity([embeddings[0]], embeddings[1:])[0]
    return intent_labels[int(np.argmax(sims))]

def classify_emotion(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    top_class = torch.argmax(probs, dim=1).item()
    return model.config.id2label[top_class]

def parse_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        raw_json = json.load(uploaded_file)
        if isinstance(raw_json, list) and "user_input" in raw_json[0] and "ai_response" in raw_json[0]:
            df = pd.DataFrame(raw_json)
        elif isinstance(raw_json, list) and all("speaker" in entry and "text" in entry for entry in raw_json):
            rows = []
            for i in range(0, len(raw_json) - 1, 2):
                if raw_json[i]["speaker"] == "user" and raw_json[i+1]["speaker"] == "ai":
                    rows.append({
                        "user_input": raw_json[i]["text"],
                        "ai_response": raw_json[i+1]["text"]
                    })
            df = pd.DataFrame(rows)
        else:
            st.error("Unsupported JSON structure.")
            return None
    else:
        st.error("Unsupported file type.")
        return None
    df = df.dropna(subset=["user_input", "ai_response"])
    return df

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Scoring Functions
def score_intent_matching(user_emb, ai_emb):
    """
    Evaluates whether the AI's response aligns with the intent of the user's input.
    Uses semantic similarity to determine alignment more robustly than punctuation alone.
    """
    
    similarity = cosine_similarity(user_emb, ai_emb)

    # Adjust score based on similarity and whether the AI answered the question
    # Since is_question detection was based on text, we will move that outside or keep heuristic here
    # For now, we will assume user_emb corresponds to a question if the original user_input had a question mark or question keywords
    # But since we removed user_input from parameters, we cannot do that here
    # So we will keep the heuristic outside and call this function only for similarity scoring
    # To keep compatibility, we will assume similarity-based scoring only here

    # For backward compatibility, let's just use similarity to determine scores as before for non-question:
    if similarity > 0.6:
        return 10  # Strong alignment
    elif similarity > 0.4:
        return 7   # Partial alignment
    else:
        return 4   # Poor alignment

def score_relevance(user_input, ai_response, user_emb, ai_emb):
    """
    Simple relevance: calibrated cross-encoder similarity (primary) + keyword coverage (secondary).
    No NLI, no floors. Penalize explicit refusals.
    """
    import math
    ce = load_cross_encoder()
    try:
        ce_raw = float(ce.predict([(user_input, ai_response)])[0])
    except Exception:
        ce_raw = 0.0

    # Sigmoid calibration for CE logits → probability in [0,1]
    ce_prob = 1.0 / (1.0 + math.exp(-ce_raw))
    sem_score = 10.0 * ce_prob  # 0..10

    # Simple keyword coverage (0..1) → 0..10
    cov = _keyword_coverage(user_input, ai_response)
    cov_score = 10.0 * cov

    # Weighted blend (keep it straightforward)
    relevance = 0.8 * sem_score + 0.2 * cov_score

    # Corrective modifier for obvious refusals/deflections
    if _detect_refusal(ai_response):
        relevance *= 0.6

    return float(round(max(0.0, min(10.0, relevance)), 1))

def extract_weighted_keywords(text):
    """
    Extracts and assigns weights to keywords based on heuristic rules.
    - Nouns and named entities are assigned higher weights.
    - Commonly less relevant words (e.g., stop words) are down-weighted.
    """
    # Simple tokenization
    words = re.findall(r'\b\w+\b', text.lower())
    keyword_weights = {}
    
    # Heuristic weights (simulating POS tagging)
    for word in words:
        if word in {"and", "is", "the", "a", "of", "in", "on", "for"}:
            keyword_weights[word] = 0  # Minimal weight for stop words
        elif word.endswith("?"):  # Heuristic for questions
            keyword_weights[word.strip("?")] = 2  # High weight for question drivers
        else:
            keyword_weights[word] = 1  # Default weight

    return Counter(keyword_weights)

def score_flow_and_continuity(previous_response, ai_response, embed):
    """
    Checks whether the AI's response follows logically from the previous response.
    """
    if not previous_response:
        return 10  # First response doesn't need continuity
    embeddings = embed([previous_response, ai_response])
    similarity_score = np.inner(embeddings[0], embeddings[1]) * 10
    return round(max(0, min(10, similarity_score)))

def score_clarity(ai_response):
    """
    Evaluates clarity based on semantic density using response length as a proxy.
    Short responses are considered less clear unless they're meaningfully complete.
    """
    word_count = len(ai_response.split())

    if word_count <= 3:
        return 3  # Very short = vague
    elif word_count <= 6:
        return 5  # Short = potentially underdeveloped
    elif word_count <= 12:
        return 8  # Moderate = generally clear
    else:
        return 10  # Extended = likely more fully formed

def score_tone_match(user_input, ai_response, user_emb, ai_emb):
    """
    Scores tone alignment using emotion classification.
    Compares emotional categories between user and AI response.
    """
    tokenizer, model = load_emotion_model()
    user_emotion = classify_emotion(user_input, tokenizer, model)
    ai_emotion = classify_emotion(ai_response, tokenizer, model)

    if user_emotion == ai_emotion:
        return 10
    elif {user_emotion, ai_emotion} <= {"neutral", "joy", "love"}:
        return 8
    elif {"anger", "fear", "sadness"} & {user_emotion, ai_emotion}:
        return 6 if user_emotion != ai_emotion else 10
    else:
        return 4  # Divergent or mismatched

def score_engagement(user_emb, ai_emb):
    """
    Scores engagement based on semantic continuation and conversational momentum.
    Combines alignment (cosine similarity) and novelty (vector delta).
    Rewards high momentum even at moderate similarity levels.
    """
    similarity = cosine_similarity(user_emb, ai_emb)
    delta = np.linalg.norm(ai_emb - user_emb)

    print(f"[DEBUG] Engagement Similarity: {similarity:.4f}, Delta: {delta:.4f}")

    if similarity > 0.85 and delta > 0.5:
        return 10  # Aligned + forward
    elif similarity > 0.7 and delta > 0.4:
        return 8
    elif similarity > 0.45 and delta > 0.9:
        return 8  # NEW: Mid similarity but strong movement = high engagement
    elif similarity > 0.55:
        return 6
    elif similarity > 0.35 and delta > 0.6:
        return 5
    elif similarity > 0.3:
        return 4
    else:
        return 2

def calculate_final_score(intent_score, relevance_score,
                          clarity_score, tone_match_score, engagement_score):
    """
    Computes a simple average of the five alignment scores.
    """
    total = intent_score + relevance_score + clarity_score + tone_match_score + engagement_score
    average = total / 5
    return round(average, 1)

# Streamlit Interface
st.title("ALIGN - AI Response Evaluator")

user_input = st.text_area("User Input", placeholder="Enter a user message...")
ai_response = st.text_area("AI Response", placeholder="Enter the AI's response...")

if st.button("Evaluate Response"):
    with st.spinner("Loading model and evaluating..."):
        embed = load_embed_model()
        if user_input and ai_response:
            user_emb, ai_emb = embed([user_input, ai_response])
            # Intent (keyword-free)
            intent_score, intent_user_label, intent_resp_label, intent_conf = score_intent_alignment(user_input, ai_response)
            relevance_score = score_relevance(user_input, ai_response, user_emb, ai_emb)
            # completeness_score = score_completeness(user_emb, ai_emb)
            # flow_score = score_flow_and_continuity(None, ai_response, embed)  # Assume no previous response for now
            clarity_score = score_clarity(ai_response)
            tone_match_score = score_tone_match(user_input, ai_response, user_emb, ai_emb)
            engagement_score = score_engagement(user_emb, ai_emb)
            final_score = calculate_final_score(
                intent_score, relevance_score,
                clarity_score, tone_match_score, engagement_score
            )

            st.caption(f"**Intent** → user: `{intent_user_label}` | response: `{intent_resp_label}` | confidence: `{intent_conf}`")
            st.subheader("Score Breakdown")

            def draw_score(label, score):
                st.write(f"**{label}**: {score}/10")
                st.progress(score / 10)

            draw_score("Intent", intent_score)
            draw_score("Relevance", relevance_score)
            # draw_score("Completeness", completeness_score)
            # draw_score("Flow", flow_score)  # flow score rendering
            draw_score("Clarity", clarity_score)
            draw_score("Tone Match", tone_match_score)
            draw_score("Engagement", engagement_score)

            st.markdown(f"### Final Score: {final_score}/10")
        else:
            st.warning("Please fill out both input fields before evaluating.")

st.markdown("---")
st.header("Batch Evaluation")

uploaded_file = st.file_uploader("Upload a .csv or .json file with conversations", type=["csv", "json"])

if uploaded_file:
    df = parse_uploaded_file(uploaded_file)
    if df is not None:
        st.success(f"Loaded {len(df)} conversation pairs.")
        embed = load_embed_model()
        tokenizer, model = load_emotion_model()

        scores = []
        for i, row in df.iterrows():
            user_input = row["user_input"]
            ai_response = row["ai_response"]
            user_emb, ai_emb = embed([user_input, ai_response])

            intent_score, intent_user_label, intent_resp_label, intent_conf = score_intent_alignment(user_input, ai_response)
            relevance_score = score_relevance(user_input, ai_response, user_emb, ai_emb)
            clarity_score = score_clarity(ai_response)
            tone_match_score = score_tone_match(user_input, ai_response, user_emb, ai_emb)
            engagement_score = score_engagement(user_emb, ai_emb)
            final_score = calculate_final_score(intent_score, relevance_score, clarity_score, tone_match_score, engagement_score)

            scores.append({
                "user_input": user_input,
                "ai_response": ai_response,
                "intent": intent_score,
                "relevance": relevance_score,
                "clarity": clarity_score,
                "tone_match": tone_match_score,
                "engagement": engagement_score,
                "final_score": final_score,
                "intent_user_label": intent_user_label,
                "intent_response_label": intent_resp_label,
                "intent_confidence": intent_conf,
            })

        scored_df = pd.DataFrame(scores)
        st.dataframe(scored_df)
        st.subheader("Per-Pair Score Breakdown")
        for i, row in scored_df.iterrows():
            with st.expander(f"Pair {i+1}"):
                st.write(f"**User Input:** {row['user_input']}")
                st.write(f"**AI Response:** {row['ai_response']}")
                st.write(f"- Intent: {row['intent']}/10")
                st.write(f"- Relevance: {row['relevance']}/10")
                st.write(f"- Clarity: {row['clarity']}/10")
                st.write(f"- Tone Match: {row['tone_match']}/10")
                st.write(f"- Engagement: {row['engagement']}/10")
                st.write(f"**Final Score:** {row['final_score']}/10")
        avg_score = scored_df["final_score"].mean().round(2)
        st.markdown(f"### Average Final Score: `{avg_score}/10`")

        st.markdown("### Download Results")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download as CSV",
                data=scored_df.to_csv(index=False).encode("utf-8"),
                file_name="alignment_scores.csv",
                mime="text/csv"
            )

        with col2:
            json_data = scored_df.to_dict(orient="records")
            st.download_button(
                label="Download as JSON",
                data=json.dumps(json_data, indent=2).encode("utf-8"),
                file_name="alignment_scores.json",
                mime="application/json"
            )