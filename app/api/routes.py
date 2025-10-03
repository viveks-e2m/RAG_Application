from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.document import DocumentUploadResponse, QueryRequest, QueryResponse, GenerateResponse
from app.core.qdrant_client import QdrantService
from app.core.embedding_service import EmbeddingService
from app.core.response_service import ResponseService
from qdrant_client.models import PointStruct
import uuid
import logging
import tempfile
import os

router = APIRouter()
logger = logging.getLogger(__name__)

# Global services
qdrant_service = QdrantService()
embedding_service = EmbeddingService()
response_service = ResponseService()

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        qdrant_service.initialize_client()
        embedding_service.initialize_model()
        response_service.initialize_client()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise

@router.post("/upload-document/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a text file, create embeddings, and store in Qdrant"""
    # Validate file type
    allowed_extensions = ['.txt', '.pdf', '.docx', '.pptx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(allowed_extensions)} files are allowed")
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Delegate document processing to embedding service
            chunks = embedding_service.process_uploaded_file(
                file_path=temp_file_path,
                file_extension=file_extension,
                content=content if file_extension == '.txt' else None
            )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No valid content found in the document")
            
            # Process chunks in batches to avoid OpenAI rate limits and token limits
            all_embeddings = []
            batch_size = 10  # Process 10 chunks at a time
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                try:
                    batch_embeddings = embedding_service.encode_text(batch_chunks)
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                    raise HTTPException(status_code=500, detail=f"Error processing document chunk {i//batch_size + 1}: {str(e)}")
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_name": file.filename,
                        "chunk_index": i,
                        "text": chunk
                    }
                ))
            
            # Store points in Qdrant
            qdrant_service.upsert_points(points)
            
            return DocumentUploadResponse(
                message=f"Document '{file.filename}' processed and stored successfully",
                chunks_processed=len(chunks)
            )
            
        except Exception as e:
            # Clean up temporary file in case of error
            if 'temp_file_path' in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            raise e
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing document")

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
            query_vector=query_embedding,  
            limit=request.top_k or 5
        )
        
        # Format results
        results = []
        for result in search_result:
            results.append({
                "document_name": result.payload.get("document_name"),
                "chunk_index": result.payload.get("chunk_index"),
                "text": result.payload.get("text"),
                "score": result.score
            })
        
        # Generate response using OpenAI
        generated_response = response_service.generate_response(
            query=query_text,
            retrieved_documents=results
        )
        
        return GenerateResponse(
            query=query_text,
            response=generated_response,
            retrieved_documents=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing query")