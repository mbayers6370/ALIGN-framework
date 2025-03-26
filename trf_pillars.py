import tensorflow_hub as hub
import numpy as np
import re
from collections import Counter

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Scoring Functions
def score_intent_matching(user_input, ai_response):
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

def score_relevance(user_input, ai_response):
    """
    Scores relevance based on semantic similarity between user input and AI response.
    """
    embeddings = embed([user_input, ai_response])
    similarity_score = np.inner(embeddings[0], embeddings[1]) * 10
    return round(max(0, min(10, similarity_score)))

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
    Evaluates completeness using weighted keyword/topic coverage and semantic similarity.
    Rounds all scores to integers.
    
    Parameters:
    - user_input: The input/query from the user.
    - ai_response: The AI's response to evaluate.
    - embed: Preloaded embedding model (e.g., Universal Sentence Encoder).
    """
    # Step 1: Extract weighted keywords
    user_keywords = extract_weighted_keywords(user_input)
    response_keywords = extract_weighted_keywords(ai_response)
    
    # Step 2: Topic coverage score with weights
    total_user_weight = sum(user_keywords.values())
    matched_weight = sum(
        response_keywords[word] * user_keywords[word]
        for word in user_keywords if word in response_keywords
    )
    topic_coverage = (matched_weight / total_user_weight * 10) if total_user_weight > 0 else 0

    # Step 3: Semantic similarity score
    embeddings = embed([user_input, ai_response])
    semantic_similarity = np.inner(embeddings[0], embeddings[1]) * 10
    semantic_similarity = max(0, min(10, semantic_similarity))  # Clamp score to [0, 10]

    # Step 4: Intent confirmation (rule-based, includes at least one high-weight match)
    intent_confirmed = 10 if any(word in response_keywords for word in user_keywords) else 5

    # Final completeness score: Weighted average with integer rounding
    completeness_score = round(
        (0.5 * round(topic_coverage) + 0.3 * round(semantic_similarity) + 0.2 * intent_confirmed)
    )
    return int(completeness_score)

def score_flow_and_continuity(previous_response, ai_response):
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
                          flow_score, clarity_score, adaptability_score, engagement_score, user_input, ai_response):
    """
    Combines pillar scores into a final weighted score with gentler penalties applied.
    """
    # Weighted scores
    weighted_score = (
        intent_score * 0.30 +
        relevance_score * 0.20 +
        completeness_score * 0.25 +
        flow_score * 0.10 +
        clarity_score * 0.05 +
        adaptability_score * 0.05 +
        engagement_score * 0.20
    )

    # Penalty for low intent or relevance
    if intent_score <= 2:
        weighted_score -= 2
    elif intent_score <= 4:
        weighted_score -= 1  # Softer penalty instead of hard cap
    if relevance_score <= 4:
        weighted_score -= 1  # Subtract for irrelevance (softer penalty)

    # Penalize AI responses that are significantly shorter than user input
    user_len = len(user_input.split())
    ai_len = len(ai_response.split())
    if ai_len < user_len * 0.6 and user_len > 5:
        weighted_score -= 1.5

    # Penalize under-engaged responses (short AI reply vs longer user input)
    response_ratio = len(ai_response.split()) / (len(user_input.split()) + 1e-6)
    if response_ratio < 0.35 and intent_score <= 5:
        weighted_score -= 1.5

    # Adjust for vagueness using semantic similarity
    embeddings = embed([user_input, ai_response])
    similarity = np.inner(embeddings[0], embeddings[1])  # Semantic similarity score

    if similarity < 0.2:  # Loosened threshold
        weighted_score -= 1
    elif 0.2 <= similarity < 0.4:  # Loosened threshold
        weighted_score -= 0.5

    # Penalize passive responses: low intent match + no engagement
    if intent_score <= 4 and engagement_score < 8:
        weighted_score -= 1.5

    # Final cap for passive, generic responses
    if intent_score <= 4 and engagement_score < 8 and response_ratio < 0.5:
        return 3  # Cap the score — it's a passive, generic response

    # Normalize the final score
    return round(max(0, min(10, weighted_score)))

# Update the call to calculate_final_score in the main function
def calculate_contextual_alignment_with_penalties(chat_transcript):
    """
    Evaluates all pillars, applies penalties for misaligned responses, and returns final scores.
    """
    scores = []
    total_score = 0

    for i, exchange in enumerate(chat_transcript):
        user_input = exchange["user"]
        ai_response = exchange["ai"]
        previous_response = chat_transcript[i - 1]["ai"] if i > 0 else None

        # Score each pillar
        intent_score = score_intent_matching(user_input, ai_response)
        relevance_score = score_relevance(user_input, ai_response)
        completeness_score = score_completeness(user_input, ai_response, embed)
        flow_score = score_flow_and_continuity(previous_response, ai_response)
        clarity_score = score_clarity(ai_response)
        adaptability_score = score_adaptability(user_input, ai_response)
        engagement_score = score_engagement(ai_response)

        print(f"\nUSER: {user_input}")
        print(f"AI: {ai_response}")
        print(f"Intent: {intent_score}, Relevance: {relevance_score}, Completeness: {completeness_score}, Flow: {flow_score}, Clarity: {clarity_score}, Adaptability: {adaptability_score}, Engagement: {engagement_score}")

        # Calculate the final score with penalties
        final_score = calculate_final_score(
            intent_score, relevance_score, completeness_score,
            flow_score, clarity_score, adaptability_score, engagement_score,
            user_input, ai_response  # Pass these explicitly
        )

        scores.append({
            "exchange": exchange,
            "intent_score": intent_score,
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "flow_score": flow_score,
            "clarity_score": clarity_score,
            "adaptability_score": adaptability_score,
            "engagement_score": engagement_score,
            "final_score": final_score
        })
        total_score += final_score

    average_score = total_score // len(chat_transcript)  # Integer average
    return scores, average_score

# Example Chat Transcript
chat_transcript = [
    # High-Ranking Responses
    {"user": "I’m feeling really down today. Nothing seems to work out.",
     "ai": "I’m sorry you’re feeling this way. What’s been the hardest part for you lately?"},
    {"user": "Work has been so hard lately. I’m always tired.",
     "ai": "Work can be exhausting. Are there ways to take short breaks or recharge during the day?"},
    {"user": "I just feel like nobody understands me. It’s hard to even talk about it.",
     "ai": "It’s so difficult to feel misunderstood. You’re not alone, and I’m here to listen if you’d like to share more."},

    # Middle-Ranking Responses
    {"user": "I’m so tired of everything. I just want to give up.",
     "ai": "I can sense how drained you feel. It might help to focus on one small thing you can control right now. What would that be?"},
    {"user": "I feel overwhelmed with work. I don't even know where to start.",
     "ai": "That sounds really tough. Maybe start by breaking tasks into smaller steps. What’s one thing you can tackle first?"},
    {"user": "I don’t know what to do anymore. Everything feels so pointless.",
     "ai": "I’m here for you, and I understand it’s hard to feel that way. Sometimes, small steps can help."},

    # Low-Ranking Responses
    {"user": "I feel like nobody cares about me anymore.",
     "ai": "That’s tough. Life can be hard sometimes."},
    {"user": "I don’t know if I can keep going like this.",
     "ai": "It’s hard, I’m sure."},
    {"user": "Everything is just too much. I can’t handle it anymore.",
     "ai": "I understand."}
]

# Process the transcript
scores, average_score = calculate_contextual_alignment_with_penalties(chat_transcript)

# Output Results
print("Contextual Awareness Scores for Each Response:")
for score in scores:
    print(f"User: {score['exchange']['user']}")
    print(f"AI: {score['exchange']['ai']}")
    print(f"Final Score: {score['final_score']}\n")

print(f"Final Average Contextual Alignment Score: {average_score}")