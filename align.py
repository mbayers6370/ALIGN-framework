import tensorflow_hub as hub
import streamlit as st
import numpy as np
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

@st.cache_resource
def load_embed_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

@st.cache_resource
def load_emotion_model():
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    return tokenizer, model

def classify_emotion(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    top_class = torch.argmax(probs, dim=1).item()
    return model.config.id2label[top_class]

st.set_page_config(page_title="ALIGN", layout="centered")
#st.title("ALIGN - AI Response Evaluator")
#st.write("Initializing...")

#st.stop()  # Immediately halts the app so we can test UI load

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Scoring Functions
def score_intent_matching(user_emb, ai_emb):
    """
    Evaluates whether the AI's response aligns with the intent of the user's input.
    Uses semantic similarity to determine alignment more robustly than punctuation alone.
    """
    # Basic heuristic: treat anything with a question mark or question keywords as a question
    question_keywords = ["who", "what", "when", "where", "why", "how", "do", "does", "can", "could", "should", "would", "is", "are"]
    # We cannot determine is_question from embeddings, so keep heuristic based on text outside this function
    
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

def score_relevance(user_emb, ai_emb):
    """
    Scores relevance based on semantic similarity between user input and AI response.
    Returns a score between 2 and 10.
    """
    similarity = cosine_similarity(user_emb, ai_emb)
    if similarity > 0.75:
        return 10
    elif similarity > 0.6:
        return 8
    elif similarity > 0.45:
        return 6
    elif similarity > 0.3:
        return 4
    else:
        return 2

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
            # Score individual components
            # Determine if user_input is a question for intent scoring heuristic
            question_keywords = ["who", "what", "when", "where", "why", "how", "do", "does", "can", "could", "should", "would", "is", "are"]
            is_question = "?" in user_input or any(user_input.lower().strip().startswith(q) for q in question_keywords)
            similarity = cosine_similarity(user_emb, ai_emb)
            # Intent logic (more generous thresholds)
            if is_question:
                if similarity > 0.7:
                    intent_score = 10
                elif similarity > 0.55:
                    intent_score = 9
                elif similarity > 0.45:
                    intent_score = 7
                elif similarity > 0.3:
                    intent_score = 5
                else:
                    intent_score = 3
            else:
                intent_score = 10 if similarity > 0.55 else 7

            relevance_score = score_relevance(user_emb, ai_emb)
            # completeness_score = score_completeness(user_emb, ai_emb)
            # flow_score = score_flow_and_continuity(None, ai_response, embed)  # Assume no previous response for now
            clarity_score = score_clarity(ai_response)
            tone_match_score = score_tone_match(user_input, ai_response, user_emb, ai_emb)
            engagement_score = score_engagement(user_emb, ai_emb)
            final_score = calculate_final_score(
                intent_score, relevance_score,
                clarity_score, tone_match_score, engagement_score
            )

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