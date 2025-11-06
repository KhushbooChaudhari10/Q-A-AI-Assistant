"""
Answer Node - Generates LLM response

This node formats the prompt with context and question,
then calls the LLM to generate the final answer.
"""

from typing import Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client for HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not set in environment variables")

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN) if HF_TOKEN else None


def answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Answer node: Generate LLM response using retrieved context.

    Args:
        state: Current agent state

    Returns:
        Updated state with 'answer' field populated
    """
    question = state.get("question", "")
    context = state.get("retrieved_context", "")
    needs_retrieval = state.get("needs_retrieval", True)

    print("=" * 60)
    print("💬 ANSWER NODE")
    print("=" * 60)
    print(f"Question: '{question}'")
    print(f"Context available: {len(context)} chars")

    if not client:
        error_msg = "Error: HF_TOKEN not configured. Please set it in .env file."
        print(f"❌ {error_msg}")
        state["answer"] = error_msg
        state["steps_log"].append("ANSWER: Failed - HF_TOKEN not set")
        print("=" * 60)
        print()
        return state

    # Load system prompt
    try:
        prompt_file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "static",
            "prompt.md"
        )
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            system_instructions = f.read()
    except Exception as e:
        print(f"⚠️ Could not load prompt.md: {e}")
        system_instructions = """You are a RAG Q&A assistant.
Use ONLY the provided context to answer questions.
If the context doesn't contain enough information, say so clearly.
Include a reflection on your answer's quality."""

    # Format the prompt
    if needs_retrieval and context:
        user_prompt = f"""Use ONLY this context to answer.

CONTEXT:
{context}

QUESTION:
{question}"""
    else:
        # Simple question without context
        user_prompt = question

    # Prepare messages with chat history
    messages = [{"role": "system", "content": system_instructions}]

    # Add chat history (last 5 exchanges to keep context manageable)
    chat_history = state.get("chat_history", [])
    if chat_history:
        recent_history = chat_history[-10:]  # Last 5 exchanges (10 messages)
        messages.extend(recent_history)

    # Add current question
    messages.append({"role": "user", "content": user_prompt})

    # Call LLM with production-ready error handling
    try:
        print("Calling LLM...")
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b:nebius",
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )
        ai_response = completion.choices[0].message.content.strip()
        print(f"Generated response: {len(ai_response)} chars")
        print(f"Preview: {ai_response[:150]}...")

    except ImportError as e:
        # API library not installed
        error_msg = f"OpenAI library not available: {str(e)}"
        print(f"❌ {error_msg}")
        ai_response = "Error: OpenAI library not installed. Please install with: pip install openai"
        state["steps_log"].append("ANSWER: Error - OpenAI library missing")

    except ConnectionError as e:
        # Network issues
        error_msg = f"Network error: {str(e)}"
        print(f"❌ {error_msg}")
        ai_response = "I'm having trouble connecting to the LLM service. Please check your internet connection and try again."
        state["steps_log"].append("ANSWER: Error - Network connection failed")

    except TimeoutError as e:
        # Request timeout
        error_msg = f"Request timeout: {str(e)}"
        print(f"❌ {error_msg}")
        ai_response = "The LLM service took too long to respond. Please try again."
        state["steps_log"].append("ANSWER: Error - Request timeout")

    except ValueError as e:
        # Invalid parameters or response
        error_msg = f"Invalid request/response: {str(e)}"
        print(f"❌ {error_msg}")
        ai_response = f"I encountered an error processing the request: {str(e)[:100]}"
        state["steps_log"].append("ANSWER: Error - Invalid request")

    except Exception as e:
        # Catch-all for API errors (rate limits, auth, etc.)
        error_msg = f"LLM API error: {type(e).__name__}: {str(e)}"
        print(f"❌ {error_msg}")

        # Check for specific error codes in message
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str:
            ai_response = "Authentication error: Please check your HF_TOKEN in the .env file."
            state["steps_log"].append("ANSWER: Error - Invalid API token")
        elif "402" in error_str or "quota" in error_str or "credit" in error_str:
            ai_response = "API quota exceeded. Please check your HuggingFace account credits or try a different model."
            state["steps_log"].append("ANSWER: Error - API quota exceeded")
        elif "429" in error_str or "rate limit" in error_str:
            ai_response = "Rate limit exceeded. Please wait a moment and try again."
            state["steps_log"].append("ANSWER: Error - Rate limit")
        elif "500" in error_str or "503" in error_str:
            ai_response = "The LLM service is temporarily unavailable. Please try again later."
            state["steps_log"].append("ANSWER: Error - Service unavailable")
        else:
            ai_response = f"I apologize, but I encountered an unexpected error: {str(e)[:100]}"
            state["steps_log"].append(f"ANSWER: Error - {type(e).__name__}")

    print("=" * 60)
    print()

    # Update state
    state["answer"] = ai_response
    if "Error" not in state["steps_log"][-1]:  # Only add if not already added in exception
        state["steps_log"].append(
            f"ANSWER: Generated {len(ai_response)} character response using LLM"
        )

    return state


# Test the node
if __name__ == "__main__":
    from agents.workflow_state import create_initial_state

    print("Testing Answer Node\n")

    # Test with context
    state = create_initial_state("What is LangGraph?")
    state["needs_retrieval"] = True
    state["retrieved_context"] = """[Source: docs.pdf, Page: 1]
LangGraph is a framework for building stateful, multi-actor applications with LLMs.
It allows you to create agent workflows with multiple nodes and edges."""

    result = answer(state)

    print("\nAnswer Result:")
    print(f"  Answer length: {len(result['answer'])} chars")
    print(f"  Steps log: {result['steps_log']}")
    print(f"\nGenerated answer:\n{result['answer']}")

    print("\n✅ Answer node test completed!")
