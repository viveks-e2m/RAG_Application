from pydantic import BaseModel
from typing import List, Optional


class VideoUploadRequest(BaseModel):
    video_file_name: str
    description: str
    duration: Optional[float] = None
    tags: Optional[List[str]] = None


class VideoEmbeddingResponse(BaseModel):
    video_file_name: str
    message: str
    embedding_dimensions: int


class VideoQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class VideoSearchResult(BaseModel):
    video_file_name: str
    description: str
    duration: Optional[float] = None
    tags: List[str]
    score: float


class VideoQueryResponse(BaseModel):
    query: str
    results: List[VideoSearchResult]