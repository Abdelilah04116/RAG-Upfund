import os
from typing import List, Dict, Tuple, Any
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from utils import EXTENSION_TO_PARSER, sanitize_text, chunk_text
from models import Source

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
        
        # Debug: Lister les modèles disponibles
        try:
            models = genai.list_models()
            print(f"[DEBUG] Modèles disponibles: {[m.name for m in models]}")
        except Exception as e:
            print(f"[DEBUG] Erreur list_models: {e}")
        
        # Configuration simple pour Gemini (modèle le plus récent)
        self.llm_model = genai.GenerativeModel("gemini-2.5-flash")
        # Modèle d'embedding local all-MiniLM-L6-v2
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
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
        try:
            # Utilisation du modèle local all-MiniLM-L6-v2
            embeddings = self.embedding_model.encode(chunks).tolist()
            return embeddings
        except Exception as e:
            print(f"[WARN] Embedding failed: {e}")
            # Fallback: vecteurs zéros (384 dimensions pour all-MiniLM-L6-v2)
            return [[0.0] * 384 for _ in chunks]

    def search(self, query: str, k=4) -> List[Dict]:
        try:
            # Utilisation du modèle local pour la requête
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
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
            print(f"[DEBUG] Tentative de génération avec Gemini...")
            print(f"[DEBUG] Prompt envoyé: {prompt[:200]}...")
            
            # Test simple d'abord
            test_response = self.llm_model.generate_content("Bonjour, peux-tu me dire bonjour ?")
            print(f"[DEBUG] Test Gemini réussi: {test_response.text[:50]}...")
            
            # Maintenant la vraie génération
            response = self.llm_model.generate_content(prompt)
            print(f"[DEBUG] Réponse Gemini reçue: {response.text[:100]}...")
            return response.text
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            print(f"[ERROR] Type d'erreur: {type(e)}")
            print(f"[ERROR] Détails: {str(e)}")
            # Fallback: réponse basique basée sur les sources
            if sources:
                return f"Basé sur les documents trouvés, voici les informations pertinentes :\n\n" + "\n\n".join([f"• {src['chunk'][:200]}..." for src in sources[:2]])
            return "Aucune réponse générée par Gemini, mais des sources ont été trouvées."
