from pydantic import BaseModel
from typing import List, Optional


class DocumentUploadResponse(BaseModel):
    message: str
    chunks_processed: int


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    document_name: Optional[str]
    chunk_index: Optional[int]
    text: Optional[str]
    score: float


class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]


class GenerateResponseRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class GenerateResponse(BaseModel):
    query: str
    response: str
    retrieved_documents: List[SearchResult]