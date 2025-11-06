"""
Streamlit UI for RAG Q&A Agent

Simple interactive interface for the LangGraph RAG agent with conversation memory.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from agents.conversational_agent import ConversationalAgent
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG Q&A Agent",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 RAG Q&A Agent with LangGraph")
st.markdown("Ask any question - the agent will retrieve relevant information from the knowledge base and answer using LLM.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This agent uses a **4-node LangGraph workflow**:

    1. 🧠 **PLAN** - Decide if retrieval needed
    2. 🔍 **RETRIEVE** - Fetch relevant chunks from ChromaDB
    3. 💬 **ANSWER** - Generate LLM response with context
    4. ✅ **REFLECT** - Evaluate answer quality
    """)

    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.markdown("- LangGraph for workflow")
    st.markdown("- ChromaDB for vector storage")
    st.markdown("- HuggingFace LLM")
    st.markdown("- Sentence Transformers embeddings")

    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("- General purpose Q&A")
    st.markdown("- Context-aware responses")
    st.markdown("- Self-reflection on quality")
    st.markdown("- Source tracking")

# Initialize conversational agent in session state
if "agent" not in st.session_state:
    st.session_state.agent = ConversationalAgent(max_history=10)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar button to clear history
with st.sidebar:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.agent.clear_history()
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response with conversation context
    with st.chat_message("assistant"):
        with st.spinner("🤔 Processing with LangGraph workflow..."):
            try:
                result = st.session_state.agent.chat(prompt)
                answer = result.get('answer', 'No response generated.')

                # Display answer
                st.markdown(answer)

                # Show workflow details in expander
                with st.expander("🔍 View Workflow Details"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📋 Execution Steps:**")
                        for step in result.get('steps_log', []):
                            st.markdown(f"- {step}")

                        st.markdown(f"\n**💬 Conversation Length:**")
                        st.markdown(f"- Messages: {len(result.get('chat_history', []))}")

                    with col2:
                        st.markdown("**✅ Reflection:**")
                        reflection = result.get('reflection', {})
                        st.markdown(f"- Quality: `{reflection.get('quality', 'N/A')}`")
                        st.markdown(f"- Confidence: `{reflection.get('confidence', 'N/A')}`")
                        if reflection.get('issues'):
                            st.markdown(f"- Issues: {', '.join(reflection['issues'])}")

                    # Show retrieved sources
                    if result.get('retrieved_metadata'):
                        st.markdown("**📚 Retrieved Sources:**")
                        for i, meta in enumerate(result['retrieved_metadata'], 1):
                            st.markdown(f"{i}. {meta.get('source', 'unknown')} (Page {meta.get('page', '?')})")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                answer = f"Error occurred: {str(e)}"

    # Add assistant response to display
    st.session_state.messages.append({"role": "assistant", "content": answer})
