# 🔥 RAG Demo - FastAPI + Gemini + ChromaDB + Streamlit

Ce projet déploie une pipeline RAG « Retrieval-Augmented Generation » utilisable via API et interface utilisateur pour interroger vos documents (PDF, DOCX, TXT) avec l’IA Gemini.

---

## 🚀 Démarrage rapide

1. **Configurer la clé Gemini**  
   Copier `.env.example` → `.env` et renseigner `GEMINI_API_KEY`.

2. **Lancer l’application**  
   ```bash
   docker-compose up --build
   ```
3. **Accéder à l’UI**  
   - Frontend : http://localhost:8501  
   - Backend : http://localhost:8000  
   - ChromaDB : http://localhost:8001

---

## 📦 Structure

```
├── backend/     # FastAPI: RAG, Embeddings, API
├── frontend/    # Streamlit UI
├── data/        # Documents + ChromaDB
│   └── raw_documents/
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🛠 Fonctionnalités

- Upload & indexation de : PDF / DOCX / TXT
- Chunking (300–700 tokens)
- Embeddings Gemini + recherche sémantique ChromaDB
- Génération de réponses contextualisées (Gemini)
- API REST (`/ask`, `/upload`, `/healthcheck`)
- Historique Q/A local (bonus)
- UI full web Streamlit

---

## 🔗 Exemples d’appels API

### Question (POST /ask)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Qu’est-ce que l’intelligence artificielle ?"}'
```

Réponse :
```json
{
  "answer": "...",
  "sources": [
    {"title": "MonDoc.pdf", "chunk": "Définition de l’intelligence..."}
  ]
}
```

### Upload (POST /upload)
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/chemin/vers/mon.pdf"
```

---

## 💡 À propos du pipeline RAG

1. **Parsing** : Extraction texte PDF/DOCX/TXT
2. **Nettoyage** : Retrait des artefacts, normalisation UTF-8
3. **Chunking** : Morceaux de 300–700 tokens (par phrase)
4. **Embeddings** : Via Gemini
5. **Stockage** : Vecteurs dans ChromaDB
6. **Retrieval** : Recherche les passages les plus pertinents
7. **Synthesis** : Génère la réponse finale avec Gemini

---

*Déployez, chargez vos documents, et posez toutes vos questions !*
