import tensorflow_hub as hub
import streamlit as st
import numpy as np
import re
from collections import Counter

st.set_page_config(page_title="ALIGN", layout="centered")
#st.title("ALIGN - AI Response Evaluator")
#st.write("Initializing...")

#st.stop()  # Immediately halts the app so we can test UI load

# Scoring Functions
def score_intent_matching(user_input, ai_response, embed):
    """
    Evaluates whether the AI's response aligns with the intent of the user's input.
    Uses semantic similarity to determine alignment more robustly than punctuation alone.
    """
    # Basic heuristic: treat anything with a question mark or question keywords as a question
    question_keywords = ["who", "what", "when", "where", "why", "how", "do", "does", "can", "could", "should", "would", "is", "are"]
    is_question = "?" in user_input or any(user_input.lower().strip().startswith(q) for q in question_keywords)

    # Compute semantic similarity between input and response
    embeddings = embed([user_input, ai_response])
    similarity = np.inner(embeddings[0], embeddings[1])

    # Adjust score based on similarity and whether the AI answered the question
    if is_question:
        if similarity > 0.6:
            return 10  # Strong alignment
        elif similarity > 0.4:
            return 7   # Partial alignment
        else:
            return 4   # Poor alignment
    else:
        # If it's not a question, we assume it's a statement or feeling
        return 10 if similarity > 0.5 else 6

def score_relevance(user_input, ai_response, embed):
    """
    Scores relevance based on semantic similarity between user input and AI response.
    Returns a score between 0 and 10.
    """
    embeddings = embed([user_input, ai_response])
    similarity = np.inner(embeddings[0], embeddings[1])
    if similarity > 0.85:
        return 10
    elif similarity > 0.7:
        return 8
    elif similarity > 0.55:
        return 6
    elif similarity > 0.4:
        return 4
    elif similarity > 0.25:
        return 2
    else:
        return 0

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

def score_completeness(user_input, ai_response, embed):
    """
    Scores completeness based on whether the AI meaningfully addresses the user's concerns
    and contains enough semantic and emotional content. Returns score from 0 to 10.
    """
    embeddings = embed([user_input, ai_response])
    similarity = np.inner(embeddings[0], embeddings[1])  # baseline understanding

    word_count = len(ai_response.split())
    addresses_concern = similarity > 0.5 and word_count > 8

    if similarity > 0.8 and word_count > 15:
        return 10
    elif similarity > 0.7 and word_count > 10:
        return 8
    elif addresses_concern:
        return 6
    elif similarity > 0.4:
        return 4
    elif word_count > 5:
        return 2
    else:
        return 0

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

def score_adaptability(user_input, ai_response):
    """
    Scores adaptability based on how well the AI matches the user's complexity and tone.
    Uses word count ratio as a soft signal of mirroring or mismatch.
    """
    user_len = len(user_input.split())
    ai_len = len(ai_response.split())

    if user_len == 0:
        return 5  # Neutral score if user gave nothing

    ratio = ai_len / user_len

    if 0.8 <= ratio <= 1.2:
        return 10  # Perfectly matched tone and effort
    elif 0.6 <= ratio < 0.8 or 1.2 < ratio <= 1.5:
        return 8  # Close, slight mismatch
    elif 0.4 <= ratio < 0.6 or 1.5 < ratio <= 2:
        return 6  # Noticeable mismatch
    else:
        return 4  # Poor adaptability

def score_engagement(ai_response):
    """
    Scores engagement based on how interactive or open-ended the response is.
    Uses question marks and response length as soft signals.
    """
    word_count = len(ai_response.split())
    has_question = "?" in ai_response

    if has_question and word_count > 10:
        return 10  # Invites a detailed follow-up
    elif has_question:
        return 8   # Invites some form of engagement
    elif word_count > 10:
        return 6   # Long but not engaging
    elif word_count > 5:
        return 5   # Neutral
    else:
        return 3   # Disengaged

def calculate_final_score(intent_score, relevance_score, completeness_score,
                          clarity_score, adaptability_score, engagement_score):
    """
    Computes a simple average of the six alignment scores.
    """
    total = intent_score + relevance_score + completeness_score + clarity_score + adaptability_score + engagement_score
    average = total / 6
    return round(average, 1)

# Streamlit Interface
st.title("ALIGN - AI Response Evaluator")

user_input = st.text_area("User Input", placeholder="Enter a user message...")
ai_response = st.text_area("AI Response", placeholder="Enter the AI's response...")

if st.button("Evaluate Response"):
    with st.spinner("Loading model and evaluating..."):
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        if user_input and ai_response:
            # Score individual components
            intent_score = score_intent_matching(user_input, ai_response, embed)
            relevance_score = score_relevance(user_input, ai_response, embed)
            completeness_score = score_completeness(user_input, ai_response, embed)
            # flow_score = score_flow_and_continuity(None, ai_response, embed)  # Assume no previous response for now
            clarity_score = score_clarity(ai_response)
            adaptability_score = score_adaptability(user_input, ai_response)
            engagement_score = score_engagement(ai_response)
            final_score = calculate_final_score(
                intent_score, relevance_score, completeness_score,
                clarity_score, adaptability_score, engagement_score
            )

            st.subheader("Score Breakdown")

            def draw_score(label, score):
                st.write(f"**{label}**: {score}/10")
                st.progress(score / 10)

            draw_score("Intent", intent_score)
            draw_score("Relevance", relevance_score)
            draw_score("Completeness", completeness_score)
            # draw_score("Flow", flow_score)  # flow score rendering
            draw_score("Clarity", clarity_score)
            draw_score("Adaptability", adaptability_score)
            draw_score("Engagement", engagement_score)

            st.markdown(f"### Final Score: {final_score}/10")
        else:
            st.warning("Please fill out both input fields before evaluating.")