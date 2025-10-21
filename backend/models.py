from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    question: str

class Source(BaseModel):
    title: str
    chunk: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]

class UploadResponse(BaseModel):
    status: str
    filename: str

class QAHistoryItem(BaseModel):
    question: str
    answer: str
    sources: List[Source]
