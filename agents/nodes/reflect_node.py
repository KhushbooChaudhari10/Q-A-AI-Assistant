"""
Reflect Node - Evaluates answer quality

This node analyzes the generated answer to assess:
- Relevance to the question
- Use of context
- Confidence level
- Overall quality

Production-ready with:
- Multi-metric heuristic evaluation (fast, no API calls)
- Optional LLM-as-judge for higher accuracy (when enabled)
- Comprehensive quality scoring
"""

from typing import Dict, Any
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "false").lower() == "true"

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN) if HF_TOKEN else None


def llm_as_judge(question: str, answer: str, context: str) -> Dict[str, Any]:
    """
    Use LLM to evaluate answer quality (production approach).

    Args:
        question: User's question
        answer: Generated answer
        context: Retrieved context

    Returns:
        Dictionary with quality, confidence, reasoning
    """
    if not client:
        return None  # Fall back to heuristics

    try:
        judge_prompt = f"""Evaluate this Q&A interaction:

QUESTION: {question}

CONTEXT PROVIDED: {context[:500]}...

ANSWER: {answer}

Evaluate on a scale of 1-5:
- Relevance: Does the answer address the question?
- Accuracy: Is the answer grounded in the context?
- Completeness: Is the answer thorough?

Respond in this format:
RELEVANCE: [1-5]
ACCURACY: [1-5]
COMPLETENESS: [1-5]
OVERALL: [poor/fair/good/excellent]
REASONING: [one sentence]"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:nebius",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=150
        )

        judge_response = response.choices[0].message.content.strip()

        # Parse the response
        quality = "medium"
        confidence = "medium"

        if "OVERALL:" in judge_response:
            overall_match = re.search(r'OVERALL:\s*(\w+)', judge_response, re.IGNORECASE)
            if overall_match:
                overall = overall_match.group(1).lower()
                if "excellent" in overall or "good" in overall:
                    quality = "good"
                    confidence = "high"
                elif "poor" in overall:
                    quality = "low"
                    confidence = "low"

        reasoning_match = re.search(r'REASONING:\s*(.+)', judge_response, re.IGNORECASE)
        reasoning = reasoning_match.group(1) if reasoning_match else "LLM judge evaluation"

        return {
            "quality": quality,
            "confidence": confidence,
            "reasoning": reasoning,
            "llm_judge_used": True
        }

    except Exception as e:
        print(f"⚠️ LLM judge failed: {e}, using heuristic fallback")
        return None  # Fall back to heuristics


def reflect(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reflect node: Evaluate answer quality and relevance.

    Evaluation metrics:
    - Answer length (completeness)
    - Context usage (did it say "not enough info"?)
    - Reflection presence (did LLM include self-reflection?)
    - Confidence estimation

    Args:
        state: Current agent state

    Returns:
        Updated state with 'reflection' field populated
    """
    question = state.get("question", "")
    answer = state.get("answer", "")
    context = state.get("retrieved_context", "")

    print("=" * 60)
    print("✅ REFLECT NODE")
    print("=" * 60)
    print(f"Question: '{question}'")
    print(f"Answer length: {len(answer)} chars")

    # Try LLM-as-judge if enabled (production mode)
    if USE_LLM_JUDGE and client:
        print("Using LLM-as-judge for evaluation...")
        llm_eval = llm_as_judge(question, answer, context)
        if llm_eval:
            # Use LLM evaluation
            reflection = {
                "answer_length_chars": len(answer),
                "answer_length_words": len(answer.split()),
                "context_used": len(context) > 0,
                "has_reflection": True,
                "confidence": llm_eval["confidence"],
                "quality": llm_eval["quality"],
                "reasoning": llm_eval["reasoning"],
                "issues": [],
                "llm_judge_used": True
            }

            print(f"\nLLM Judge Results:")
            print(f"  Quality: {reflection['quality']}")
            print(f"  Confidence: {reflection['confidence']}")
            print(f"  Reasoning: {reflection['reasoning']}")
            print("=" * 60)
            print()

            state["reflection"] = reflection
            state["steps_log"].append(
                f"REFLECT: LLM-judge - Quality={reflection['quality']}, Confidence={reflection['confidence']}"
            )
            return state

    # Fall back to heuristic evaluation
    print("Using heuristic evaluation...")

    # Initialize reflection metrics
    reflection = {
        "answer_length_chars": len(answer),
        "answer_length_words": len(answer.split()),
        "context_used": len(context) > 0,
        "has_reflection": False,
        "confidence": "unknown",
        "quality": "unknown",
        "issues": []
    }

    # Check if answer is too short
    if len(answer) < 50:
        reflection["quality"] = "low"
        reflection["issues"].append("Answer is very short")
        reflection["confidence"] = "low"

    # Check for "I don't know" / insufficient information patterns
    insufficient_patterns = [
        "not enough information",
        "doesn't contain enough",
        "cannot answer",
        "don't have enough",
        "insufficient information",
        "i don't know",
        "no information about"
    ]

    answer_lower = answer.lower()
    if any(pattern in answer_lower for pattern in insufficient_patterns):
        reflection["confidence"] = "low"
        reflection["issues"].append("Answer indicates insufficient information")
        print("  ⚠️ Low confidence: Answer indicates lack of information")

    # Check if LLM included its own reflection
    reflection_patterns = [
        r"reflection:",
        r"confidence:",
        r"this answer appears",
        r"i am (confident|uncertain)",
        r"the (context|information) (does|doesn't)"
    ]

    for pattern in reflection_patterns:
        if re.search(pattern, answer_lower):
            reflection["has_reflection"] = True
            print("  ✓ Answer includes self-reflection")
            break

    # Extract confidence from LLM's reflection if present
    if "confidence" in answer_lower:
        if "high" in answer_lower or "confident" in answer_lower:
            reflection["confidence"] = "high"
        elif "low" in answer_lower or "uncertain" in answer_lower:
            reflection["confidence"] = "low"
        else:
            reflection["confidence"] = "medium"

    # Determine quality based on multiple factors
    if reflection["quality"] == "unknown":
        if len(answer.split()) > 20 and reflection["confidence"] != "low":
            reflection["quality"] = "good"
        elif len(answer.split()) > 10:
            reflection["quality"] = "medium"
        else:
            reflection["quality"] = "low"

    # Check if context was available but not used
    if len(context) > 0 and len(answer) < 30:
        reflection["issues"].append("Context available but answer very brief")

    # If no issues found and quality is good
    if not reflection["issues"] and reflection["quality"] == "good":
        reflection["confidence"] = reflection.get("confidence", "high")

    # Final confidence if still unknown
    if reflection["confidence"] == "unknown":
        if reflection["quality"] == "good":
            reflection["confidence"] = "medium"
        else:
            reflection["confidence"] = "low"

    print(f"\nReflection Results:")
    print(f"  Quality: {reflection['quality']}")
    print(f"  Confidence: {reflection['confidence']}")
    print(f"  Has Self-Reflection: {reflection['has_reflection']}")
    print(f"  Answer Words: {reflection['answer_length_words']}")
    if reflection['issues']:
        print(f"  Issues: {', '.join(reflection['issues'])}")
    else:
        print(f"  Issues: None")

    print("=" * 60)
    print()

    # Update state
    state["reflection"] = reflection
    state["steps_log"].append(
        f"REFLECT: Quality={reflection['quality']}, Confidence={reflection['confidence']}"
    )

    return state


# Test the node
if __name__ == "__main__":
    from agents.workflow_state import create_initial_state

    print("Testing Reflect Node\n")

    # Test 1: Good answer with reflection
    print("Test 1: Good Answer")
    state1 = create_initial_state("What is LangGraph?")
    state1["answer"] = """LangGraph is a framework for building stateful multi-actor applications with LLMs.
It enables developers to create complex agent workflows with multiple nodes and edges.

Reflection: This answer appears relevant to the user question."""
    state1["retrieved_context"] = "Some context about LangGraph..."

    result1 = reflect(state1)
    assert result1["reflection"]["quality"] in ["good", "medium"]
    assert result1["reflection"]["has_reflection"] == True

    # Test 2: Insufficient information answer
    print("\nTest 2: Insufficient Information")
    state2 = create_initial_state("What is quantum computing?")
    state2["answer"] = "The context doesn't contain enough information to answer this question."
    state2["retrieved_context"] = "Some unrelated context..."

    result2 = reflect(state2)
    assert result2["reflection"]["confidence"] == "low"
    assert len(result2["reflection"]["issues"]) > 0

    # Test 3: Very short answer
    print("\nTest 3: Short Answer")
    state3 = create_initial_state("What is AI?")
    state3["answer"] = "It's artificial intelligence."
    state3["retrieved_context"] = ""

    result3 = reflect(state3)
    assert result3["reflection"]["quality"] == "low"

    print("\n✅ All reflect node tests passed!")
