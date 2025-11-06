"""
ChromaDB Vector Store for RAG Agent

This module handles PDF document ingestion, chunking, embedding,
and semantic search using ChromaDB for persistent vector storage.
"""

import os
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


class ChromaVectorStore:
    """
    Vector store using ChromaDB for persistent document embeddings.

    Features:
    - Persistent storage (survives restarts)
    - Metadata support (source file, page, chunk index)
    - Semantic similarity search
    - Automatic embedding generation
    """

    def __init__(
        self,
        data_folder: str = "data",
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_documents",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            data_folder: Directory containing PDF files
            persist_directory: Where to store ChromaDB data
            collection_name: Name of the ChromaDB collection
            embed_model: Sentence transformer model for embeddings
            chunk_size: Number of words per chunk
        """
        self.data_folder = data_folder
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size

        # Initialize embedding model
        print(f"🔧 Loading embedding model: {embed_model}")
        self.embed_model = SentenceTransformer(embed_model)

        # Initialize ChromaDB client with persistence
        print(f"🔧 Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            doc_count = self.collection.count()
            print(f"✅ Loaded existing collection '{collection_name}' with {doc_count} documents")

            # Check for new documents and auto-index them
            self._check_and_index_new_documents()

        except Exception:
            print(f"📝 Creating new collection '{collection_name}'")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "RAG document chunks with embeddings"}
            )
            # Load documents on first initialization
            self._load_and_index_documents()

    def _extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of (text, page_number) tuples
        """
        pages_text = []
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_text.append((page_text, page_num))
        except Exception as e:
            print(f"⚠️ Error reading {pdf_path}: {e}")

        return pages_text

    def _chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Split text into chunks of approximately chunk_size words.

        Args:
            text: Text to chunk
            chunk_size: Number of words per chunk

        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

        return chunks

    def _get_indexed_sources(self) -> set:
        """
        Get list of PDF files that are already indexed in the collection.

        Returns:
            Set of source filenames that are already indexed
        """
        try:
            # Get all documents from collection
            if self.collection.count() == 0:
                return set()

            # Get all metadata to extract unique sources
            all_data = self.collection.get()
            if all_data and 'metadatas' in all_data:
                indexed_sources = {meta.get('source') for meta in all_data['metadatas'] if meta.get('source')}
                return indexed_sources
            return set()
        except Exception as e:
            print(f"⚠️ Error getting indexed sources: {e}")
            return set()

    def _check_and_index_new_documents(self):
        """
        Check for new PDF files and index them automatically.
        """
        if not os.path.exists(self.data_folder):
            return

        # Get current PDF files
        current_pdf_files = set(f for f in os.listdir(self.data_folder) if f.endswith(".pdf"))

        if not current_pdf_files:
            return

        # Get already indexed sources
        indexed_sources = self._get_indexed_sources()

        # Find new files
        new_files = current_pdf_files - indexed_sources

        if new_files:
            print(f"\n🔍 Detected {len(new_files)} new PDF file(s)")
            self._index_specific_files(list(new_files))
        else:
            print(f"✅ All documents already indexed ({len(current_pdf_files)} file(s))")

    def _index_specific_files(self, pdf_files: List[str]):
        """
        Index specific PDF files.

        Args:
            pdf_files: List of PDF filenames to index
        """
        print(f"\n📚 Processing {len(pdf_files)} new PDF file(s)")

        all_chunks = []
        all_metadatas = []
        all_ids = []
        chunk_counter = 0

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.data_folder, pdf_file)
            print(f"\n📖 Processing: {pdf_file}")

            # Extract text from PDF (with page numbers)
            pages_text = self._extract_text_from_pdf(pdf_path)

            if not pages_text:
                print(f"  ⚠️ No text extracted from {pdf_file}")
                continue

            print(f"  📄 Extracted {len(pages_text)} pages")

            # Chunk each page
            file_chunk_count = 0
            for page_text, page_num in pages_text:
                chunks = self._chunk_text(page_text)

                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "source": pdf_file,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "source_type": "pdf"
                    })
                    all_ids.append(f"{pdf_file}_p{page_num}_c{chunk_idx}")
                    file_chunk_count += 1
                    chunk_counter += 1

            print(f"  ✅ Created {file_chunk_count} chunks from {pdf_file}")

        if not all_chunks:
            print("\n⚠️ No chunks created. No documents to index.")
            return

        # Generate embeddings and add to ChromaDB
        print(f"\n🔄 Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embed_model.encode(all_chunks, show_progress_bar=True)

        print(f"💾 Adding documents to ChromaDB...")
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas,
            ids=all_ids
        )

        print(f"✅ Successfully indexed {len(all_chunks)} chunks from {len(pdf_files)} PDF(s)\n")

    def _load_and_index_documents(self):
        """
        Load all PDFs from data folder and index them in ChromaDB.
        """
        if not os.path.exists(self.data_folder):
            print(f"⚠️ Warning: Data folder '{self.data_folder}' does not exist")
            print(f"📝 Creating empty data folder. Please add PDF files to it.")
            os.makedirs(self.data_folder, exist_ok=True)
            return

        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith(".pdf")]

        if not pdf_files:
            print(f"⚠️ Warning: No PDF files found in '{self.data_folder}'")
            print(f"📝 Please add PDF documents to the data folder.")
            return

        # Use the same indexing logic
        self._index_specific_files(pdf_files)

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Add new documents to the vector store.

        Args:
            documents: List of text chunks to add
            metadatas: Optional list of metadata dicts for each document
        """
        if not documents:
            return

        # Generate embeddings
        embeddings = self.embed_model.encode(documents)

        # Generate IDs
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas if metadatas else [{}] * len(documents),
            ids=ids
        )

        print(f"✅ Added {len(documents)} documents to collection")

    def query(
        self,
        query_text: str,
        top_k: int = 3
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Semantic search for similar documents.

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            Tuple of (documents, metadatas, distances)
        """
        # Check if collection is empty
        if self.collection.count() == 0:
            print("⚠️ Warning: Collection is empty. No documents to search.")
            return [], [], []

        # Generate query embedding
        query_embedding = self.embed_model.encode([query_text])

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, self.collection.count())
        )

        # Extract results
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []

        return documents, metadatas, distances

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve context for RAG (backward compatible with old RAGAgent).

        Args:
            query: Query string
            top_k: Number of chunks to retrieve

        Returns:
            Concatenated context string
        """
        documents, metadatas, distances = self.query(query, top_k)

        if not documents:
            return "No relevant context found in the knowledge base."

        # Format context with metadata
        context_parts = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            source = meta.get('source', 'unknown')
            page = meta.get('page', '?')
            context_parts.append(f"[Source: {source}, Page: {page}]\n{doc}")

        return "\n\n".join(context_parts)

    def get_stats(self) -> Dict:
        """
        Get collection statistics.

        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }

    def reset(self):
        """
        Reset the collection (delete all documents).
        WARNING: This will delete all indexed documents!
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document chunks with embeddings"}
        )
        print(f"🔄 Collection '{self.collection_name}' has been reset")


# Example usage
if __name__ == "__main__":
    # Initialize vector store
    vector_store = ChromaVectorStore(data_folder="data")

    # Get stats
    stats = vector_store.get_stats()
    print(f"\n📊 Vector Store Stats:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Storage: {stats['persist_directory']}")

    # Test query
    if stats['total_documents'] > 0:
        print(f"\n🔍 Testing query...")
        query = "What is cybersecurity?"
        documents, metadatas, distances = vector_store.query(query, top_k=2)

        print(f"\nQuery: '{query}'")
        print(f"Found {len(documents)} results:\n")

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            print(f"Result {i}:")
            print(f"  Source: {meta.get('source', 'unknown')}")
            print(f"  Page: {meta.get('page', '?')}")
            print(f"  Distance: {dist:.4f}")
            print(f"  Text: {doc[:200]}...")
            print()
