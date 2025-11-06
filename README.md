# RAG Q&A Agent with LangGraph

A Retrieval-Augmented Generation (RAG) question-answering agent built with LangGraph that answers questions by retrieving relevant information from a knowledge base and generating contextual responses using an LLM.

## Overview

This project implements a 4-node LangGraph workflow:
- **PLAN**: Determines if retrieval is needed
- **RETRIEVE**: Fetches relevant chunks from ChromaDB vector database
- **ANSWER**: Generates LLM response using retrieved context
- **REFLECT**: Evaluates answer quality and confidence

---

## Prerequisites

- Python 3.8 or higher
- HuggingFace account with API token (free tier works)
- 2GB disk space for dependencies
- Internet connection for LLM API calls

---

## Setup Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Q-A-AI-Assistant
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Required
HF_TOKEN=your_huggingface_token_here

# Optional (for advanced features)
USE_LLM_PLANNING=false
USE_LLM_JUDGE=false
```

**How to get HF_TOKEN:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is sufficient)
3. Copy and paste it into your `.env` file

### 5. Add Documents to Knowledge Base

Place your PDF documents in the `data/` folder:

```bash
mkdir -p data
# Copy your PDF files to data/ folder
cp /path/to/your/documents/*.pdf data/
```

The system will automatically index these documents on first run.

---

## Running the Application

### Understanding Stateless vs Stateful Modes

This project supports **two modes of operation**:

**🔹 Stateless Mode** (Demo & Evaluation)
- **Purpose**: Demonstrate the LangGraph workflow step-by-step
- **Used by**: `demo.ipynb`, `evaluate_rag_advanced.py`
- **Behavior**: Each question is independent, no conversation memory
- **Best for**: Understanding the 4-node workflow, testing, evaluation

**🔹 Stateful Mode** (Interactive UIs)
- **Purpose**: Multi-turn conversations with memory
- **Used by**: Flask app (`app.py`), Streamlit UI (`streamlit_app.py`)
- **Behavior**: Remembers previous messages (last 10 exchanges)
- **Best for**: Real-world usage, follow-up questions, interactive chat

---

### Option 1: Jupyter Notebook (Recommended for Demo)

This is the **primary deliverable** for demonstration (**stateless mode**):

```bash
jupyter notebook demo.ipynb
```

Then click "Run All" to see the complete workflow with test questions.

**Note**: The notebook intentionally processes each question independently to clearly demonstrate the 4-node workflow (PLAN → RETRIEVE → ANSWER → REFLECT).

### Option 2: Flask Web Interface

For interactive web-based usage (**stateful mode with conversation memory**):

```bash
python3 app.py
```

Access at: http://127.0.0.1:5001

**Features**: Session-based chat history, follow-up questions, conversation export

### Option 3: Streamlit Interface

For a modern chat-based interface (**stateful mode with conversation memory**):

```bash
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

**Features**: Chat history display, clear conversation button, workflow details

### Option 4: Direct Python Script

To test the workflow programmatically (**stateless mode**):

```bash
python3 -m agents.langgraph_agent
```

---

## Project Structure

```
Q-A-AI-Assistant/
├── demo.ipynb                    # PRIMARY DELIVERABLE: Jupyter notebook demo
├── DESIGN_REPORT.md              # Design decisions and challenges
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (create this)
│
├── agents/
│   ├── langgraph_agent.py       # Main LangGraph workflow (core)
│   ├── conversational_agent.py  # Conversation memory wrapper
│   ├── vector_store.py          # ChromaDB wrapper
│   ├── workflow_state.py        # State definition (TypedDict)
│   └── nodes/
│       ├── plan_node.py         # Plan: decide if retrieval needed
│       ├── retrieve_node.py     # Retrieve: fetch relevant chunks
│       ├── answer_node.py       # Answer: generate LLM response
│       └── reflect_node.py      # Reflect: evaluate quality
│
├── evaluation/
│   └── evaluate_rag_advanced.py # RAGAS/BERTScore/ROUGE evaluation
│
├── data/                        # Place PDF documents here
│   └── .gitkeep
│
├── chroma_db/                   # ChromaDB persistent storage (auto-created)
├── memory/                      # Exported conversations (auto-created)
│
├── static/
│   ├── prompt.md                # System prompt for LLM
│   ├── script.js                # Frontend JavaScript
│   └── style.css                # CSS styling
│
├── templates/
│   └── index.html               # Flask UI template
│
├── app.py                       # Flask web application (stateful)
└── streamlit_app.py             # Streamlit UI (stateful)
```

---

## Architecture and Approach

### High-Level Workflow

```
User Question
    ↓
[PLAN NODE] ──→ Needs Retrieval?
    ↓                    ↓ No
    ↓ Yes              [ANSWER]
    ↓                    ↓
[RETRIEVE NODE]         ↓
    ↓                    ↓
[ANSWER NODE] ←─────────┘
    ↓
[REFLECT NODE]
    ↓
Final Response
```

### Key Design Decisions

**1. LangGraph for Workflow Management**
- Chosen for its state-based approach to agent workflows
- TypedDict ensures type safety across nodes
- Conditional routing allows skipping retrieval when unnecessary

**2. ChromaDB for Vector Storage**
- Persistent storage (data survives restarts)
- Metadata support (source file, page numbers)
- Simpler API compared to FAISS

**3. 300-Word Chunking Strategy**
- Balances context preservation with retrieval precision
- Tested with 500 words (too broad) and 200 words (too fragmented)
- Includes overlap potential for future enhancement

**4. Hybrid Evaluation Approach**
- Fast heuristic-based reflection by default
- Optional LLM-as-judge for higher accuracy
- Multi-metric scoring (length, confidence, completeness)

**5. Error Handling at Every Level**
- Empty knowledge base detection
- API quota/rate limit handling
- Network error graceful degradation
- Invalid input validation

### Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Workflow Engine | LangGraph | State-based agent orchestration |
| Vector Database | ChromaDB | Persistent embedding storage |
| Embeddings | sentence-transformers | Semantic similarity (all-MiniLM-L6-v2) |
| LLM | HuggingFace Inference API | Answer generation (gpt-oss-20b) |
| Document Processing | PyPDF2 | PDF text extraction |
| Web Framework | Flask | Interactive web interface |
| UI Framework | Streamlit | Modern chat interface |
| Notebook | Jupyter | Demonstration and testing |

---

## Usage Examples

### Example Questions to Try

**In Knowledge Base** (if using task.pdf):
- "What is LangGraph?"
- "What framework should I use for this task?"
- "What vector database is recommended?"
- "What is the submission deadline?"

**Outside Knowledge Base**:
- "What is quantum computing?"
- "How does Bitcoin work?"

The system will detect when information is unavailable and respond accordingly.

---

## Evaluation

Run the evaluation suite to measure performance:

```bash
# Evaluation with RAGAS/BERTScore/ROUGE
python3 -m evaluation.evaluate_rag_advanced
```

**Note**:  Evaluation may take time due to model downloads.

---

## Troubleshooting

### Common Issues

**1. HF_TOKEN not set**
```
Error: HF_TOKEN not set in environment variables
```
**Solution**: Create `.env` file with your HuggingFace token

**2. Empty knowledge base**
```
Warning: Vector store is empty (no documents indexed)
```
**Solution**: Add PDF files to the `data/` folder and restart

**3. API quota exceeded**
```
Error code: 402 - You have exceeded your monthly included credits
```
**Solution**: Wait for quota reset or upgrade HuggingFace plan

**4. ChromaDB initialization error**
```
Error: Cannot connect to ChromaDB
```
**Solution**: Delete `chroma_db/` folder and restart (will re-index)

**5. Import errors**
```
ModuleNotFoundError: No module named 'langgraph'
```
**Solution**: Ensure virtual environment is activated and run `pip install -r requirements.txt`

---

## Advanced Configuration

### Enable LLM-Based Planning

Add to `.env`:
```
USE_LLM_PLANNING=true
```

This uses the LLM to classify query intent more accurately (slower but better).

### Enable LLM-as-Judge Reflection

Add to `.env`:
```
USE_LLM_JUDGE=true
```

This uses the LLM to evaluate answer quality (more accurate than heuristics).

---

## Development

### Running Tests

Test individual nodes:
```bash
python3 -m agents.nodes.plan_node
python3 -m agents.nodes.retrieve_node
python3 -m agents.nodes.answer_node
python3 -m agents.nodes.reflect_node
```

Test complete workflow:
```bash
python3 -m agents.langgraph_agent
```

### Adding New Documents

Simply place new PDF files in the `data/` folder and restart the application:

1. Copy your PDF files to the `data/` folder
2. Restart the application (Flask/Streamlit/notebook)
3. New documents are **automatically detected and indexed**

The system tracks which files are already indexed and only processes new ones. No manual cleanup required!

---

## Submission Checklist

- [x] Jupyter notebook (`demo.ipynb`) demonstrating all 4 nodes
- [x] LangGraph workflow with plan, retrieve, answer, reflect nodes
- [x] ChromaDB vector database with persistent storage
- [x] HuggingFace embeddings (sentence-transformers)
- [x] Logging at each step (plan → retrieve → answer → reflect)
- [x] `requirements.txt` with all dependencies
- [x] README.md with setup steps and approach
- [x] Design report (DESIGN_REPORT.md) explaining challenges
- [x] Evaluation code (RAGAS/BERTScore/ROUGE)
- [x] **Bonus**: Flask web interface
- [x] **Bonus**: Streamlit chat interface

---