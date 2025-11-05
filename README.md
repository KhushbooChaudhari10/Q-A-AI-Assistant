# CyberSec Assistant

An AI assistant designed to provide clear and simple explanations for fundamental cybersecurity concepts. This tool is perfect for individuals new to the field, students, or professionals looking to brush up on their knowledge.

## Screenshots

![Screenshot 1](screenshots/1.png)
![Screenshot 2](screenshots/2.png)
![Screenshot 3](screenshots/3.png)
![Screenshot 4](screenshots/4.png)
![Screenshot 5](screenshots/5.png)
![Screenshot 6](screenshots/6.png)
![Screenshot 7](screenshots/7.png)

## Features

- **Conversational AI:** Engage in a natural conversation to learn about cybersecurity.
- **Beginner-Friendly:** Complex topics are broken down into easy-to-understand explanations.
- **Defensive Security Focus:** The assistant specializes in defensive cybersecurity practices, offering advice on protection and mitigation.
- **Memory:** The assistant can recall previous conversations to provide a more contextual experience.
- **Chat Export:** Export your conversation history for future reference.

## Technologies Used

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **AI:** HuggingFace API with the `meta-llama/Meta-Llama-3.1-8B-Instruct` model
- **Styling:** A clean, modern interface for a smooth user experience.

## Getting Started

### Prerequisites

- Python 3.x
- A HuggingFace account and an API token (`HF_TOKEN`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cyber_security
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   Create a file named `.env` in the root directory and add your HuggingFace API token:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

### Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Access the application:**
   Open your web browser and navigate to `http://127.0.0.1:5001`.

## Project Structure

```
cyber_security/
├── .gitignore
├── alternative_architectures.md  # Documents alternative model deployment strategies
├── app.py                        # Main Flask application
├── model_evidence.md             # Justification for the selected AI model
├── requirements.txt              # Python dependencies
├── task.txt                      # Project requirements
├── agents/
│   └── ai_assistant.py           # Core AI assistant logic
├── memory/                       # Stores chat history and summaries
├── static/
│   ├── prompt.md                 # System prompt for the AI model
│   ├── script.js                 # Frontend JavaScript for interactivity
│   └── style.css                 # CSS for styling the web interface
└── templates/
    └── index.html                # Main HTML template
```

## How It Works

The application uses a Flask backend to serve a simple web interface. When a user sends a message, it's processed by the `MemoryAgentHF` in `ai_assistant.py`. This agent communicates with the HuggingFace API to get a response from the `meta-llama/Meta-Llama-3.1-8B-Instruct` model, then displays it to the user. The conversation history is saved to the `memory` folder.
