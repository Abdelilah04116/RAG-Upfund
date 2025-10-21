import os
from typing import List, Dict, Tuple, Any
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from .utils import EXTENSION_TO_PARSER, sanitize_text, chunk_text
from .models import Source

DATA_DIR = os.getenv("DATA_DIR", "./data")
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw_documents")
COLLECTION_NAME = "rag_docs"
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

class RAGEngine:
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.chroma_client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=Settings(allow_reset=True)
        )
        self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-pro")
    
    def index_documents(self):
        doc_paths = []
        for fname in os.listdir(RAW_DOCS_DIR):
            path = os.path.join(RAW_DOCS_DIR, fname)
            ext = os.path.splitext(fname.lower())[1]
            if ext in EXTENSION_TO_PARSER:
                doc_paths.append((fname, path, ext))
        ids, metadatas, documents = [], [], []
        for fname, path, ext in doc_paths:
            try:
                text = EXTENSION_TO_PARSER[ext](path)
                text = sanitize_text(text)
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    uid = f"{fname}_{i}"
                    ids.append(uid)
                    metadatas.append({"title": fname, "chunk_id": i})
                    documents.append(chunk)
            except Exception as e:
                print(f"[WARN] Failed to index {fname}: {e}")
        if ids:
            self.collection.upsert(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
                embeddings=self.embed_chunks(documents)
            )

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        # Gemini supports embedding via the `embed_content` method
        embeddings = []
        for chunk in chunks:
            try:
                emb = self.model.embed_content(chunk, task_type="retrieval_document")["embedding"]
            except Exception as e:
                print(f"[WARN] Embedding failed: {e}")
                emb = [0.0] * 768
            embeddings.append(emb)
        return embeddings

    def search(self, query: str, k=4) -> List[Dict]:
        try:
            emb = self.model.embed_content(query, task_type="retrieval_query")["embedding"]
            results = self.collection.query(
                query_embeddings=[emb],
                n_results=k,
                include=["documents", "metadatas"]
            )
            hits = []
            for i in range(len(results['documents'][0])):
                hits.append({
                    "title": results["metadatas"][0][i]["title"],
                    "chunk": results["documents"][0][i]
                })
            return hits
        except Exception as e:
            print(f"[WARN] Retrieval failed: {e}")
            return []

    def synthesize_answer(self, query: str, sources: List[Dict]) -> str:
        context = "\n\n".join([f"Source: {src['title']} -> {src['chunk']}" for src in sources])
        prompt = (
            f"Voici des extraits de documents :\n{context}\n\n"
            f"Question : {query}\n"
            "Synthétise une réponse précise, cite tes sources si possible."
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[WARN] Generation failed: {e}")
            return "Une erreur est survenue lors de la génération de la réponse."
