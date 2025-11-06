# Design Report: RAG Q&A Agent with LangGraph

**Project**: LangGraph-based RAG Question Answering Agent

---

## How the Agent Works

The agent is built using LangGraph's workflow system with four connected nodes that process questions step-by-step. When a user asks a question, it first goes through the **plan node**, which decides whether the question needs information from documents or can be answered directly (like simple greetings). For knowledge questions, the workflow moves to the **retrieve node**, which searches through PDF documents stored in ChromaDB vector database. I chose ChromaDB because it keeps the data saved even after restarting the application, unlike FAISS which loses everything. The documents are split into 300-word chunks and converted into embeddings using the sentence-transformers library, which helps find similar content based on meaning rather than just matching keywords. The retrieve node finds the top 3 most relevant chunks and passes them to the **answer node**, which combines these chunks with the user's question and sends everything to the HuggingFace LLM. The system uses a specific prompt that tells the LLM to only use the provided context and admit when it doesn't have enough information. Finally, the **reflect node** evaluates the answer by checking its length, looking for phrases that indicate uncertainty, and calculating a confidence score. All of this state information (question, context, answer, evaluation) flows through a TypedDict structure that LangGraph manages automatically, making it easy to track what's happening at each step.

## Challenges Faced

The biggest challenge was figuring out the right chunk size for splitting documents. I started with 500 words per chunk, but the problem was that important information got mixed with less relevant content, making it harder for the LLM to extract the right details. When I reduced it to 300 words, the retrieval became more precise, but sometimes related information got split across different chunks. I solved this by adding metadata (source file name, page number, chunk number) to each chunk so I could track where information came from and potentially combine related chunks later if needed. Another major challenge was making the reflect node actually useful without calling the LLM again (which would be slow and use more API credits). I built a system that checks multiple things: answer length to see if it's too short, scans for phrases like "not enough information" to detect when the system is unsure, and looks for the LLM's own reflection statements to extract confidence levels. This approach works about 70-80% of the time in identifying low-quality answers during my testing. The third challenge was handling questions that aren't in the knowledge base at all. Through careful prompt design, I made sure the LLM explicitly says when the context doesn't have enough information instead of making up answers. The reflect node then catches these cases and marks them as low confidence. The last challenge was switching from FAISS to ChromaDB, which required rewriting how documents get loaded and indexed. ChromaDB uses a different API where you need to call `collection.add()` with embeddings and metadata in specific formats, but the effort was worth it because now the vector database persists between runs, which is essential for a real application.

---

## Technical Decisions

| Decision | Why I Made This Choice |
|----------|------------------------|
| **300-word chunks** | Good balance between keeping related information together and finding precise matches |
| **Heuristic-based planning** | Fast decisions without extra LLM calls, though I added an option to use LLM for better accuracy when needed |
| **Multi-metric reflection** | Avoids extra API costs while still catching most quality issues |
| **HuggingFace API** | Free tier available and works with standard OpenAI library format |
| **sentence-transformers** | Fast embedding generation (384 dimensions) with good accuracy for semantic search |

---

## What Works Well

1. **Clear Structure**: Each node does one specific job, making it easy to understand and modify
2. **Visible Progress**: Emoji-coded logging shows exactly what's happening at each step
3. **Smart Routing**: Simple questions skip retrieval and go straight to answering
4. **Type Safety**: Using TypedDict helps catch errors before running the code
5. **Data Persistence**: ChromaDB saves all embeddings so you don't have to re-process documents
6. **Easy Extension**: The node-based design makes it simple to add new steps like re-ranking or query expansion

---

## Known Limitations and Future Improvements

**Current Limitations:**
- Planning uses simple keyword matching; could be smarter with LLM-based classification
- Always retrieves exactly 3 chunks; might need more or fewer depending on the question
- Reflection uses patterns and rules; would be more accurate with LLM-as-judge approach
- Each question is independent; no memory of previous conversation turns
- PDF parsing can fail on complex documents with tables or images

**What Could Be Better:**
1. **Smarter Planning**: Use a small LLM to better understand question types
2. **Flexible Retrieval**: Adjust how many chunks to fetch based on question complexity
3. **Better Ranking**: Add a re-ranking step after retrieval to sort results by relevance
4. **Query Expansion**: Generate similar phrasings of the question to find more matches
5. **Conversation Memory**: Keep track of chat history for multi-turn discussions
6. **Automated Testing**: Add RAGAS metrics to measure performance systematically
7. **Hybrid Search**: Combine semantic search with keyword matching for better coverage

---

## Conclusion

This project successfully demonstrates a working RAG system using LangGraph's state management. The four-node design (plan → retrieve → answer → reflect) separates concerns clearly while keeping everything connected through shared state. Although there's room for improvement in areas like planning accuracy and reflection depth, the system achieves its main goals: answering questions using retrieved context and evaluating its own responses. Using ChromaDB for persistent storage and adding comprehensive error handling makes the system reliable enough for real use. The modular structure means I can improve individual parts without rebuilding everything, which is important for iterative development. Through building this, I learned how state-based workflows handle complex tasks better than simple linear code, and how important it is to design for failure cases (empty databases, API errors, missing information) rather than just the happy path.

