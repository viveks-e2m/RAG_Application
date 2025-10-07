import uuid
import logging
import tempfile
import os

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.document import (
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    GenerateResponse,
)
from app.schemas.video import (
    VideoUploadRequest,
    VideoEmbeddingResponse,
    VideoQueryRequest,
)
from app.core.qdrant_client_service import QdrantService
from app.core.embedding_service import EmbeddingService
from app.core.response_service import ResponseService
from app.core.video_embedding_service import VideoEmbeddingService
from app.core.video_rag_system import VideoRAGSystemQdrant
from qdrant_client.models import PointStruct

router = APIRouter()
logger = logging.getLogger(__name__)

# Global services
qdrant_service = QdrantService()
embedding_service = EmbeddingService()
response_service = ResponseService()
video_embedding_service = VideoEmbeddingService()
video_rag_system = None  # Will be initialized at startup


@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        qdrant_service.initialize_client()
        embedding_service.initialize_model()
        response_service.initialize_client()
        video_embedding_service.initialize_model()
        
        # Initialize the shared Whisper model
        global video_rag_system
        from app.core.video_rag_system import VideoRAGSystemQdrant
        VideoRAGSystemQdrant.initialize_whisper_model("small")
        
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise


@router.on_event("shutdown")
async def shutdown_event():
    """Release resources on shutdown"""
    try:
        # Release the shared Whisper model
        from app.core.video_rag_system import VideoRAGSystemQdrant
        VideoRAGSystemQdrant.release_whisper_model()
        logger.info("Resources released successfully")
    except Exception as e:
        logger.error(f"Error releasing resources: {e}")


@router.post("/upload-document/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a text file, create embeddings, and store in Qdrant"""
    # Validate file type
    allowed_extensions = [".txt", ".pdf", ".docx", ".pptx", ".html"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(allowed_extensions)} files are allowed",
        )

    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Delegate document processing to embedding service
            chunks = embedding_service.process_uploaded_file(
                file_path=temp_file_path,
                file_extension=file_extension,
                content=content if file_extension == ".txt" else None,
            )

            # Clean up temporary file
            os.unlink(temp_file_path)

            if not chunks:
                raise HTTPException(
                    status_code=400, detail="No valid content found in the document"
                )

            # Process chunks in batches to avoid OpenAI rate limits and token limits
            all_embeddings = []
            batch_size = 10  # Process 10 chunks at a time

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                try:
                    batch_embeddings = embedding_service.encode_text(batch_chunks)
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(
                        f"Error creating embeddings for batch {i//batch_size + 1}: {e}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing document chunk {i//batch_size + 1}: {str(e)}",
                    )

            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                point_id = str(uuid.uuid4())
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "document_name": file.filename,
                            "chunk_index": i,
                            "text": chunk,
                            "content_type": "document",  # Distinguish document from video
                        },
                    )
                )

            # Store points in Qdrant
            qdrant_service.upsert_points(points)

            return DocumentUploadResponse(
                message=f"Document '{file.filename}' processed and stored successfully",
                chunks_processed=len(chunks),
            )

        except Exception as e:
            # Clean up temporary file in case of error
            if "temp_file_path" in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            raise e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while processing document"
        )


@router.post("/upload-video/", response_model=VideoEmbeddingResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file, extract metadata, create embeddings, and store in Qdrant"""
    # Validate video file type
    allowed_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(allowed_extensions)} video files are allowed",
        )

    try:
        # Save video file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process video using VideoRAGSystemQdrant for full processing
            # Use the correct Qdrant host based on environment
            from app.core.config import settings
            qdrant_host = "qdrant" if settings.IS_DOCKER else settings.QDRANT_HOST
            
            video_rag_system = VideoRAGSystemQdrant(
                model_size="small",
                chunk_size=45,
                chunk_overlap=15,
                qdrant_host=qdrant_host,
                qdrant_port=settings.QDRANT_PORT,
            )

            # Process the video
            video_rag_system.process_video(temp_file_path)

            # Also process basic metadata for search
            video_file_name = file.filename
            description = f"Video file: {video_file_name}. This is a video content that can be searched using semantic queries."

            # Process video metadata using the video embedding service
            metadata = video_embedding_service.process_video_metadata(
                video_file_name=video_file_name,
                description=description,
                duration=None,  # Could extract actual duration with video processing libraries
                tags=["video", "uploaded"],  # Default tags
            )

            # Clean up temporary file
            os.unlink(temp_file_path)

            # Create a unique point ID for Qdrant
            point_id = str(uuid.uuid4())

            # Prepare point for Qdrant storage
            point = PointStruct(
                id=point_id,
                vector=metadata["description_embedding"],
                payload={
                    "video_file_name": metadata["video_file_name"],
                    "description": metadata["description"],
                    "duration": metadata["duration"],
                    "tags": metadata["tags"],
                    "content_type": "video",  # Distinguish video from document
                },
            )

            # Store point in Qdrant
            qdrant_service.upsert_points([point])

            return VideoEmbeddingResponse(
                video_file_name=video_file_name,
                message=f"Video '{video_file_name}' processed and stored successfully",
                embedding_dimensions=len(metadata["description_embedding"]),
            )

        except Exception as e:
            # Clean up temporary file in case of error
            if "temp_file_path" in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            raise e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while processing video"
        )


@router.post("/process-video-metadata/", response_model=VideoEmbeddingResponse)
async def process_video_metadata(request: VideoUploadRequest):
    """Process video metadata and create embeddings for video descriptions"""
    try:
        # Process video metadata using the video embedding service
        metadata = video_embedding_service.process_video_metadata(
            video_file_name=request.video_file_name,
            description=request.description,
            duration=request.duration,
            tags=request.tags,
        )

        # Create a unique point ID for Qdrant
        point_id = str(uuid.uuid4())

        # Prepare point for Qdrant storage
        point = PointStruct(
            id=point_id,
            vector=metadata["description_embedding"],
            payload={
                "video_file_name": metadata["video_file_name"],
                "description": metadata["description"],
                "duration": metadata["duration"],
                "tags": metadata["tags"],
                "content_type": "video",  # Distinguish video from document
            },
        )

        # Store point in Qdrant
        qdrant_service.upsert_points([point])

        return VideoEmbeddingResponse(
            video_file_name=request.video_file_name,
            message=f"Video metadata for '{request.video_file_name}' processed and stored successfully",
            embedding_dimensions=len(metadata["description_embedding"]),
        )

    except Exception as e:
        logger.error(f"Error processing video metadata: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing video metadata",
        )

@router.post("/query-video-transcripts/", response_model=GenerateResponse)
async def query_video_transcripts(request: VideoQueryRequest):
    """Query stored video transcript segments using semantic search and generate response"""
    query_text = request.query.strip()

    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Create a temporary VideoRAGSystemQdrant instance to perform the search
        # We need to use the correct Qdrant host based on the environment
        from app.core.config import settings
        
        # Use the appropriate Qdrant host based on environment
        qdrant_host = "qdrant" if settings.IS_DOCKER else settings.QDRANT_HOST
        
        video_rag_system = VideoRAGSystemQdrant(
            model_size="small",
            chunk_size=45,
            chunk_overlap=15,
            qdrant_host=qdrant_host,
            qdrant_port=settings.QDRANT_PORT,
        )

        # Perform search on transcript segments
        search_results = video_rag_system.search(query_text, request.top_k or 5)

        # Format results similar to document results for consistency
        results = []
        for result in search_results:
            results.append(
                {
                    "document_name": "Video Transcript Segment",
                    "chunk_index": 0,
                    "text": result["text"],
                    "score": result["similarity"],
                }
            )

        # Generate response using OpenAI
        generated_response = response_service.generate_response_from_transcripts(
            query=query_text, retrieved_transcripts=results
        )

        return GenerateResponse(
            query=query_text, response=generated_response, retrieved_documents=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video transcript query: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while processing video transcript query"
        )


@router.post("/query-document/", response_model=GenerateResponse)
async def query_document(request: QueryRequest):
    """Query the stored documents and generate a well-formatted response using OpenAI"""
    query_text = request.query.strip()

    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # First, search for relevant documents
        query_embedding = embedding_service.encode_text([query_text])[0]

        search_result = qdrant_service.search_points(
            query_vector=query_embedding, limit=request.top_k or 5
        )

        # Filter results to only include documents
        document_results = [
            result
            for result in search_result
            if result.payload.get("content_type") == "document"
        ]

        # Take only the requested number of results
        document_results = document_results[: request.top_k or 5]

        # Format results
        results = []
        for result in document_results:
            results.append(
                {
                    "document_name": result.payload.get("document_name"),
                    "chunk_index": result.payload.get("chunk_index"),
                    "text": result.payload.get("text"),
                    "score": result.score,
                }
            )

        # Generate response using OpenAI
        generated_response = response_service.generate_response(
            query=query_text, retrieved_documents=results
        )

        return GenerateResponse(
            query=query_text, response=generated_response, retrieved_documents=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while processing query"
        )
