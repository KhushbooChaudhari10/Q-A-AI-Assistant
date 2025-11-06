"""
Conversational RAG Agent with Memory

Wraps the LangGraph workflow with conversation memory for multi-turn chat.
Used by Flask and Streamlit UIs for stateful conversations.
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
from agents.langgraph_agent import get_workflow
from agents.workflow_state import create_initial_state


class ConversationalAgent:
    """
    Stateful conversational agent that maintains chat history.

    Features:
    - Multi-turn conversation support
    - Chat history persistence
    - Automatic history management (keeps last N exchanges)
    - Export functionality
    """

    def __init__(self, max_history: int = 10, memory_folder: str = "memory"):
        """
        Initialize conversational agent.

        Args:
            max_history: Maximum number of exchanges to keep (user+assistant pairs)
            memory_folder: Folder to save conversation history
        """
        self.max_history = max_history
        self.memory_folder = memory_folder
        self.chat_history: List[Dict[str, str]] = []
        self.workflow = get_workflow()

        # Create memory folder if it doesn't exist
        os.makedirs(self.memory_folder, exist_ok=True)

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message with conversation context.

        Args:
            user_message: The user's message

        Returns:
            Dictionary with answer, reflection, and metadata
        """
        if not user_message or not user_message.strip():
            return {
                "answer": "Please enter a message.",
                "reflection": {"quality": "error", "confidence": "none"},
                "chat_history": self.chat_history
            }

        # Create state with conversation history
        state = create_initial_state(user_message, self.chat_history.copy())

        # Run workflow
        try:
            result = self.workflow.invoke(state)

            # Extract answer
            answer = result.get("answer", "I apologize, but I couldn't generate a response.")

            # Update chat history
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": answer})

            # Trim history if too long (keep last max_history exchanges)
            if len(self.chat_history) > self.max_history * 2:
                self.chat_history = self.chat_history[-(self.max_history * 2):]

            return {
                "answer": answer,
                "reflection": result.get("reflection", {}),
                "retrieved_metadata": result.get("retrieved_metadata", []),
                "steps_log": result.get("steps_log", []),
                "chat_history": self.chat_history
            }

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(f"❌ {error_msg}")

            return {
                "answer": f"I apologize, but an error occurred: {str(e)[:100]}",
                "reflection": {"quality": "error", "confidence": "none"},
                "chat_history": self.chat_history
            }

    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.chat_history.copy()

    def export_conversation(self, filename: str = None) -> str:
        """
        Export conversation to JSON file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        filepath = os.path.join(self.memory_folder, filename)

        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(self.chat_history),
            "messages": self.chat_history
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)

        return filepath


# Test the conversational agent
if __name__ == "__main__":
    print("Testing Conversational Agent\n")

    agent = ConversationalAgent()

    # Test multi-turn conversation
    questions = [
        "What is LangGraph?",
        "How do I use it?",
        "What are the main components?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Turn {i}: {question}")
        print(f"{'='*60}")

        result = agent.chat(question)

        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"History length: {len(result['chat_history'])} messages")

    # Export conversation
    export_path = agent.export_conversation()
    print(f"\n✅ Conversation exported to: {export_path}")
