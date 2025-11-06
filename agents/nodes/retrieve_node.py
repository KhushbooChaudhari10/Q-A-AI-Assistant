"""
Retrieve Node - Fetches relevant context from vector store

This node performs semantic search on the ChromaDB vector store
to find the most relevant chunks for answering the question.
"""

from typing import Dict, Any
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.vector_store import ChromaVectorStore


# Initialize vector store (singleton)
_vector_store = None


def get_vector_store() -> ChromaVectorStore:
    """Get or create vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore(data_folder="data")
    return _vector_store


def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve node: Fetch relevant context from vector database.

    Production-ready implementation with:
    - Comprehensive error handling
    - Empty vector store detection
    - Graceful degradation
    - Validation of retrieved results

    Args:
        state: Current agent state

    Returns:
        Updated state with 'retrieved_context' and 'retrieved_metadata'
    """
    question = state.get("question", "")
    needs_retrieval = state.get("needs_retrieval", True)

    print("=" * 60)
    print("🔍 RETRIEVE NODE")
    print("=" * 60)
    print(f"Question: '{question}'")
    print(f"Needs Retrieval: {needs_retrieval}")

    if not needs_retrieval:
        print("Skipping retrieval (not needed)")
        state["retrieved_context"] = ""
        state["retrieved_metadata"] = []
        state["steps_log"].append("RETRIEVE: Skipped (not needed)")
        print("=" * 60)
        print()
        return state

    # Validate question
    if not question or not question.strip():
        print("⚠️ Warning: Empty question, skipping retrieval")
        state["retrieved_context"] = ""
        state["retrieved_metadata"] = []
        state["steps_log"].append("RETRIEVE: Skipped (empty question)")
        print("=" * 60)
        print()
        return state

    # Perform retrieval with error handling
    try:
        vector_store = get_vector_store()

        # Check if vector store is empty
        stats = vector_store.get_stats()
        if stats['total_documents'] == 0:
            print("⚠️ Warning: Vector store is empty (no documents indexed)")
            state["retrieved_context"] = "Knowledge base is empty. Please add documents to the data/ folder."
            state["retrieved_metadata"] = []
            state["steps_log"].append("RETRIEVE: Failed - Empty knowledge base")
            print("=" * 60)
            print()
            return state

        documents, metadatas, distances = vector_store.query(question, top_k=3)

        # Validate retrieval results
        if not documents:
            print("⚠️ No documents retrieved (query returned empty)")
            state["retrieved_context"] = "No relevant information found in knowledge base."
            state["retrieved_metadata"] = []
            state["steps_log"].append("RETRIEVE: No results found")
            print("=" * 60)
            print()
            return state

        print(f"Retrieved {len(documents)} chunks:")

        # Format context with source information
        context_parts = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            source = meta.get('source', 'unknown')
            page = meta.get('page', '?')
            print(f"  {i}. Source: {source}, Page: {page}, Distance: {dist:.4f}")
            print(f"     Preview: {doc[:100]}...")

            context_parts.append(f"[Source: {source}, Page: {page}]\n{doc}")

        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        print("=" * 60)
        print()

        # Update state
        state["retrieved_context"] = context
        state["retrieved_metadata"] = [
            {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "distance": dist
            }
            for meta, dist in zip(metadatas, distances)
        ]
        state["steps_log"].append(
            f"RETRIEVE: Found {len(documents)} relevant chunks from vector store"
        )

        return state

    except FileNotFoundError as e:
        print(f"❌ Error: Data folder not found - {e}")
        state["retrieved_context"] = "Error: Knowledge base data folder not found."
        state["retrieved_metadata"] = []
        state["steps_log"].append(f"RETRIEVE: Error - Data folder missing")
        print("=" * 60)
        print()
        return state

    except Exception as e:
        print(f"❌ Error during retrieval: {type(e).__name__}: {str(e)}")
        state["retrieved_context"] = f"Error retrieving context: {str(e)[:100]}"
        state["retrieved_metadata"] = []
        state["steps_log"].append(f"RETRIEVE: Error - {type(e).__name__}")
        print("=" * 60)
        print()
        return state


# Test the node
if __name__ == "__main__":
    from agents.workflow_state import create_initial_state

    # Test retrieval
    print("Testing Retrieve Node\n")

    state = create_initial_state("What is LangGraph?")
    state["needs_retrieval"] = True

    result = retrieve(state)

    print("\nRetrieve Results:")
    print(f"  Context length: {len(result['retrieved_context'])} chars")
    print(f"  Metadata count: {len(result['retrieved_metadata'])}")
    print(f"  Steps log: {result['steps_log']}")

    if result['retrieved_context']:
        print(f"\nContext preview:")
        print(result['retrieved_context'][:300] + "...")

    print("\n✅ Retrieve node test completed!")
