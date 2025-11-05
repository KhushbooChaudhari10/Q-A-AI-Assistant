from flask import Flask, render_template, request, jsonify
from agents.ai_assistant import MemoryAgentHF
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize the MemoryAgent (persists memory across requests)
agent = MemoryAgentHF()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a message."})
    
    # Get AI response
    bot_response = agent.get_response(user_message)
    
    # Export memory after each message
    try:
        agent.export_memory()
    except Exception as e:
        print(f"⚠️ Failed to export memory: {e}")

    return jsonify({"response": bot_response})

@app.route("/export_memory", methods=["GET"])
def export_memory():
    file_path = agent.export_memory()
    return jsonify({"message": f"Memory exported to {file_path}"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
