from flask import Flask, render_template, request, jsonify, session
from agents.conversational_agent import ConversationalAgent
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-in-production")

# Store agents per session (in-memory for simplicity)
# In production, use Redis or database
agents = {}


def get_agent(session_id: str) -> ConversationalAgent:
    """Get or create agent for this session."""
    if session_id not in agents:
        agents[session_id] = ConversationalAgent(max_history=10)
    return agents[session_id]


@app.route("/")
def index():
    # Create session ID if doesn't exist
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Get session ID
    session_id = session.get('session_id', 'default')

    # Get agent for this session
    agent = get_agent(session_id)

    # Get response with conversation context
    result = agent.chat(user_message)

    return jsonify({
        "response": result['answer'],
        "reflection": result.get('reflection', {}),
        "history_length": len(result.get('chat_history', []))
    })


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear conversation history for current session."""
    session_id = session.get('session_id', 'default')

    if session_id in agents:
        agents[session_id].clear_history()

    return jsonify({"message": "Conversation history cleared."})


@app.route("/export_memory", methods=["GET"])
def export_memory():
    """Export current conversation to file."""
    session_id = session.get('session_id', 'default')

    if session_id in agents:
        file_path = agents[session_id].export_conversation()
        return jsonify({"message": f"Memory exported to {file_path}"})
    else:
        return jsonify({"message": "No conversation to export."})


@app.route("/get_history", methods=["GET"])
def get_history():
    """Get conversation history."""
    session_id = session.get('session_id', 'default')

    if session_id in agents:
        history = agents[session_id].get_history()
        return jsonify({"history": history})
    else:
        return jsonify({"history": []})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
