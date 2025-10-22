from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, shutil, threading, json

from models import AskRequest, AskResponse, Source, UploadResponse, QAHistoryItem
from rag_engine import RAGEngine, RAW_DOCS_DIR
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

app = FastAPI(title="RAG FastAPI Gemini")

# CORS for Streamlit local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

rag = RAGEngine(GEMINI_API_KEY)
HISTORY_PATH = os.path.join("data", "qa_history.json")

@app.on_event("startup")
def startup_event():
    # Index documents at startup (blocking)
    os.makedirs(RAW_DOCS_DIR, exist_ok=True)
    rag.index_documents()

@app.get("/healthcheck")
async def healthcheck():
    return "OK"

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    query = request.question
    sources = rag.search(query)
    answer = rag.synthesize_answer(query, sources)
    # Save in local history
    save_history(query, answer, sources)
    formatted_sources = [Source(title=src["title"], chunk=src["chunk"]) for src in sources]
    return AskResponse(answer=answer, sources=formatted_sources)

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename.lower())[1]
    dest = os.path.join(RAW_DOCS_DIR, filename)
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail="Format non supporté.")
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Ré-indexation (non bloquante)
    threading.Thread(target=rag.index_documents, daemon=True).start()
    return UploadResponse(status="uploaded", filename=filename)

@app.get("/history")
async def history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(question, answer, sources):
    try:
        item = QAHistoryItem(
            question=question,
            answer=answer,
            sources=[Source(title=s["title"], chunk=s["chunk"]) for s in sources]
        )
        hist = []
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, encoding="utf-8") as f:
                hist = json.load(f)
        hist.append(item.dict())
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save history: {e}")
