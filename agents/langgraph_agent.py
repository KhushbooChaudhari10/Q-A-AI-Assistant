"""
LangGraph RAG Q&A Agent

Combines 4 nodes into a workflow:
PLAN → RETRIEVE → ANSWER → REFLECT

This is the core agent that will be used by both:
- Jupyter notebook (demo.ipynb) - primary deliverable
- Flask app (app.py) - bonus feature
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from agents.workflow_state import AgentState, create_initial_state
from agents.nodes import plan, retrieve, answer, reflect


def should_retrieve(state: Dict[str, Any]) -> str:
    """
    Conditional edge: Decide whether to retrieve or skip to answer.

    Args:
        state: Current agent state

    Returns:
        "retrieve" if retrieval needed, "answer" if not
    """
    if state.get("needs_retrieval", True):
        return "retrieve"
    else:
        return "answer"


def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow with 4 nodes.

    Workflow structure:
        START → plan → [conditional] → retrieve → answer → reflect → END
                           ↓ (if no retrieval needed)
                         answer

    Returns:
        Compiled LangGraph workflow
    """
    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add the 4 nodes
    workflow.add_node("plan", plan)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", answer)
    workflow.add_node("reflect", reflect)

    # Set entry point
    workflow.set_entry_point("plan")

    # Add conditional edge from plan
    workflow.add_conditional_edges(
        "plan",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "answer": "answer"
        }
    )

    # Add edge from retrieve to answer
    workflow.add_edge("retrieve", "answer")

    # Add edge from answer to reflect
    workflow.add_edge("answer", "reflect")

    # Add edge from reflect to END
    workflow.add_edge("reflect", END)

    # Compile the graph
    return workflow.compile()


# Global workflow instance (singleton)
_workflow = None


def get_workflow():
    """Get or create the compiled workflow."""
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def run_workflow(question: str) -> Dict[str, Any]:
    """
    Run the complete RAG workflow for a question.

    This is the main entry point used by:
    - Jupyter notebook demos
    - Flask web interface

    Args:
        question: User's question

    Returns:
        Final state with answer, reflection, and execution log
    """
    print("\n" + "=" * 60)
    print("🚀 STARTING RAG WORKFLOW")
    print("=" * 60)
    print(f"Question: '{question}'")
    print("=" * 60)
    print()

    # Create initial state
    initial_state = create_initial_state(question)

    # Get workflow
    workflow = get_workflow()

    # Run the workflow
    try:
        final_state = workflow.invoke(initial_state)
    except Exception as e:
        print(f"\n❌ Workflow error: {e}")
        # Return error state
        initial_state["answer"] = f"Error: {str(e)}"
        initial_state["reflection"] = {"quality": "error", "confidence": "none"}
        initial_state["steps_log"].append(f"ERROR: {str(e)}")
        return initial_state

    print("\n" + "=" * 60)
    print("🎯 WORKFLOW COMPLETE")
    print("=" * 60)
    print("\n📋 EXECUTION LOG:")
    for step in final_state.get("steps_log", []):
        print(f"  • {step}")

    print("\n" + "=" * 60)
    print()

    return final_state


def print_result(result: Dict[str, Any]):
    """
    Pretty print workflow results.

    Args:
        result: Final state from run_workflow()
    """
    print("\n" + "🔷" * 30)
    print("\n📊 FINAL RESULT\n")
    print("🔷" * 30)

    print(f"\n❓ QUESTION:")
    print(f"   {result.get('question', 'N/A')}")

    print(f"\n🧠 PLAN:")
    print(f"   Retrieval needed: {result.get('needs_retrieval', 'N/A')}")

    if result.get('retrieved_metadata'):
        print(f"\n🔍 RETRIEVED SOURCES:")
        for i, meta in enumerate(result['retrieved_metadata'], 1):
            print(f"   {i}. {meta.get('source', 'unknown')} (Page {meta.get('page', '?')})")

    print(f"\n💬 ANSWER:")
    answer = result.get('answer', 'N/A')
    print(f"   {answer}")

    reflection = result.get('reflection', {})
    print(f"\n✅ REFLECTION:")
    print(f"   Quality: {reflection.get('quality', 'N/A')}")
    print(f"   Confidence: {reflection.get('confidence', 'N/A')}")
    if reflection.get('issues'):
        print(f"   Issues: {', '.join(reflection['issues'])}")

    print("\n" + "🔷" * 30)
    print()


# Test the complete workflow
if __name__ == "__main__":
    print("Testing LangGraph RAG Workflow\n")

    # Test 1: Knowledge question (should use retrieval)
    print("\n" + "█" * 60)
    print("TEST 1: Knowledge Question (with retrieval)")
    print("█" * 60)

    result1 = run_workflow("What is LangGraph?")
    print_result(result1)

    # Test 2: Simple greeting (should skip retrieval)
    print("\n" + "█" * 60)
    print("TEST 2: Simple Greeting (no retrieval)")
    print("█" * 60)

    result2 = run_workflow("Hello")
    print_result(result2)

    # Test 3: Another knowledge question
    print("\n" + "█" * 60)
    print("TEST 3: Complex Question")
    print("█" * 60)

    result3 = run_workflow("How does RAG work with vector databases?")
    print_result(result3)

    print("\n✅ All workflow tests completed!")
    print("\n📝 Note: The workflow is now ready to be used in:")
    print("   1. Jupyter notebook (demo.ipynb)")
    print("   2. Flask app (app.py)")
