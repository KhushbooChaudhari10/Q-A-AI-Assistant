"""
Plan Node - Determines if retrieval is needed

This node analyzes the user's question and decides whether
RAG retrieval is necessary or if it can be answered directly.

Uses hybrid approach:
1. Fast heuristic filtering for obvious cases
2. LLM-based classification for ambiguous queries (production-ready)
"""

from typing import Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
USE_LLM_PLANNING = os.getenv("USE_LLM_PLANNING", "false").lower() == "true"

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN) if HF_TOKEN else None


def plan_with_llm(question: str) -> tuple[bool, str]:
    """
    Use LLM to classify if question needs retrieval (production approach).

    Args:
        question: User's question

    Returns:
        Tuple of (needs_retrieval: bool, reasoning: str)
    """
    if not client:
        # Fallback to heuristic
        return True, "LLM unavailable, defaulting to retrieval"

    try:
        classification_prompt = f"""Classify if this question requires retrieving information from a knowledge base.

Question: "{question}"

Respond with ONLY one word: YES or NO

YES if: The question asks about facts, concepts, or specific information
NO if: The question is a greeting, opinion, or general conversation

Classification:"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:nebius",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.1,
            max_tokens=10
        )

        answer = response.choices[0].message.content.strip().upper()
        needs_retrieval = "YES" in answer
        reasoning = f"LLM classified as {'knowledge query' if needs_retrieval else 'simple query'}"

        return needs_retrieval, reasoning

    except Exception as e:
        # Fallback to heuristic on error
        print(f"⚠️ LLM planning failed: {e}, using heuristic fallback")
        return True, f"LLM error (fallback to retrieval): {str(e)[:50]}"


def plan_with_heuristic(question: str) -> tuple[bool, str]:
    """
    Fast heuristic-based planning (fallback approach).

    Args:
        question: User's question

    Returns:
        Tuple of (needs_retrieval: bool, reasoning: str)
    """
    question_lower = question.lower().strip()

    # Fast filtering for obvious cases
    greetings = ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"]
    simple_queries = ["how are you", "who are you", "what can you do"]

    # Short greetings
    if any(greeting in question_lower for greeting in greetings):
        if len(question_lower.split()) <= 3:
            return False, "Heuristic: Short greeting detected"

    # Simple conversational queries
    if any(simple in question_lower for simple in simple_queries):
        return False, "Heuristic: Simple conversational query"

    # Default to retrieval for knowledge questions
    return True, "Heuristic: Knowledge question detected"


def plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Plan node: Analyze query and decide if retrieval is needed.

    Production-ready implementation with:
    - LLM-based classification (when enabled)
    - Heuristic fallback for speed/reliability
    - Comprehensive error handling

    Args:
        state: Current agent state

    Returns:
        Updated state with 'needs_retrieval' set
    """
    question = state.get("question", "").strip()

    if not question:
        # Empty question - no retrieval needed
        needs_retrieval = False
        reasoning = "Empty question"
    elif USE_LLM_PLANNING and client:
        # Production: Use LLM for better accuracy
        needs_retrieval, reasoning = plan_with_llm(question)
    else:
        # Fallback: Use fast heuristics
        needs_retrieval, reasoning = plan_with_heuristic(question)

    # Log the decision
    print("=" * 60)
    print("🧠 PLAN NODE")
    print("=" * 60)
    print(f"Question: '{question}'")
    print(f"Needs Retrieval: {needs_retrieval}")
    print(f"Reasoning: {reasoning}")
    print("=" * 60)
    print()

    # Update state
    state["needs_retrieval"] = needs_retrieval
    state["steps_log"].append(
        f"PLAN: {reasoning}. Retrieval={'required' if needs_retrieval else 'not required'}"
    )

    return state


# Test the node
if __name__ == "__main__":
    from agents.workflow_state import create_initial_state

    # Test 1: Knowledge question
    print("Test 1: Knowledge Question")
    state1 = create_initial_state("What is a DDoS attack?")
    result1 = plan(state1)
    assert result1["needs_retrieval"] == True, "Should need retrieval for knowledge question"

    # Test 2: Greeting
    print("\nTest 2: Greeting")
    state2 = create_initial_state("Hello")
    result2 = plan(state2)
    assert result2["needs_retrieval"] == False, "Should not need retrieval for greeting"

    # Test 3: Complex question
    print("\nTest 3: Complex Question")
    state3 = create_initial_state("How does RAG work in LangGraph?")
    result3 = plan(state3)
    assert result3["needs_retrieval"] == True, "Should need retrieval for complex question"

    print("\n✅ All plan node tests passed!")
