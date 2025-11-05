import os
import json
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

from agents.rag_agent import RAGAgent   # <<--- NEW IMPORT

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("⚠️ WARNING: HF_TOKEN not set in environment variables")
    print("Please create a .env file with your HuggingFace token:")
    print("HF_TOKEN=your_huggingface_token_here")
    raise ValueError("HF_TOKEN not set in environment variables")

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)

rag = RAGAgent(data_folder="data")   # <<--- RAG LOADED ONCE


class MemoryAgentHF:
    def __init__(self, model="openai/gpt-oss-20b:nebius",
                 retention=timedelta(days=1),
                 summary_interval: int = 5,
                 recent_keep: int = 4,
                 memory_folder: str = "memory"):
        self.model = model
        self.retention = retention
        self.summary_interval = summary_interval
        self.recent_keep = recent_keep
        self.memory_folder = memory_folder

        os.makedirs(self.memory_folder, exist_ok=True)

        self.chat_history = []
        self.summary = ""

    def _save_summary(self):
        try:
            summary_file = os.path.join(self.memory_folder, "running_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump({"summary": self.summary}, f, indent=2, ensure_ascii=False)
        except:
            pass

    def _summarize_history(self, min_messages: int = None):
        if min_messages is None:
            min_messages = self.summary_interval

        total = len(self.chat_history)
        if total < min_messages:
            return

        cutoff = max(0, total - self.recent_keep)
        old_msgs = self.chat_history[:cutoff]
        history_blob = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in old_msgs]).strip()
        if not history_blob:
            return

        prompt = (
            "Summarize:\n\n"
            f"{history_blob}\n\n"
            "Return only 2-4 sentence summary."
        )

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            summary_text = resp.choices[0].message.content.strip()
        except:
            summary_text = ""

        if summary_text:
            self.summary = summary_text
            self.chat_history = self.chat_history[cutoff:]
            self._save_summary()

    # =====================================
    # UPDATED get_response WITH RAG
    # =====================================
    def get_response(self, user_message: str) -> str:
        if not user_message.strip():
            return "Please enter a message."

        # store user msg into memory chat
        self.chat_history.append({"role": "user", "content": user_message})

        # summarization
        self._summarize_history()

        # ----------- RAG NEW STEP ----------
        context = rag.retrieve_context(user_message)
        rag_prompt = f"""
Use ONLY this context to answer.

CONTEXT:
{context}

QUESTION:
{user_message}
"""
        # -----------------------------------

        # load system prompt
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), "..", "static", "prompt.md")
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                system_instructions = f.read()
        except:
            system_instructions = "You are a RAG agent."

        messages = [{"role": "system", "content": system_instructions}]
        if self.summary:
            messages.append({"role": "system", "content": f"Memory Summary: {self.summary}"})

        messages.append({"role": "user", "content": rag_prompt})

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=700
            )
            ai_response = completion.choices[0].message.content.strip()
            self.chat_history.append({"role": "assistant", "content": ai_response})
            return ai_response

        except Exception as e:
            return "RAG + LLM error: " + str(e)

    def export_memory(self):
        self._summarize_history(min_messages=2)
        filename = f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        file_path = os.path.join(self.memory_folder, filename)
        data = {"summary": self.summary, "messages": self.chat_history}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return file_path
