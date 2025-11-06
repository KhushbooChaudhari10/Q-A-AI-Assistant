# # agents/rag_agent.py

# import os
# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer
# from PyPDF2 import PdfReader

# class RAGAgent:
#     def __init__(self, data_folder="data", embed_model="sentence-transformers/all-MiniLM-L6-v2"):
#         self.data_folder = data_folder
#         self.embed_model = SentenceTransformer(embed_model)
#         self.index = None
#         self.chunks = []  # list of text chunks (parallel to index)

#         self._load_and_prepare()

#     def _load_and_prepare(self):
#         pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith(".pdf")]
#         full_chunks = []

#         for pdf_name in pdf_files:
#             pdf_path = os.path.join(self.data_folder, pdf_name)
#             text = self._extract_text_from_pdf(pdf_path)
#             chunks = self._chunk_text(text)
#             full_chunks.extend(chunks)

#         self.chunks = full_chunks

#         # embed and build FAISS index
#         embeds = self.embed_model.encode(full_chunks)
#         dim = embeds.shape[1]

#         self.index = faiss.IndexFlatL2(dim)
#         self.index.add(embeds)

#     def _extract_text_from_pdf(self, pdf_path):
#         text = ""
#         reader = PdfReader(pdf_path)
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#         return text

#     def _chunk_text(self, text, chunk_size=300):
#         words = text.split()
#         chunks = []
#         for i in range(0, len(words), chunk_size):
#             chunk = " ".join(words[i:i+chunk_size])
#             chunks.append(chunk)
#         return chunks

#     def retrieve_context(self, query, top_k=3):
#         q_embed = self.embed_model.encode([query])
#         scores, idx = self.index.search(q_embed, top_k)

#         matched = [self.chunks[i] for i in idx[0]]
#         return "\n".join(matched)
