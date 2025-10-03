# FastAPI RAG Endpoint with Qdrant

A FastAPI application that provides endpoints for uploading text documents, creating vector embeddings using sentence-transformers, and performing similarity search using Qdrant as the vector database.

## Features

- Upload text files via API endpoint
- Automatically create vector embeddings using `all-MiniLM-L6-v2` (free and open-source)
- Store embeddings in Qdrant vector database
- Query documents using similarity search
- Generate well-formatted responses using OpenAI GPT models
- RESTful API with proper error handling and validation
- Streamlit frontend for ChatGPT-like interface
- Automatic document chunking for large files
- **Advanced document parsing and chunking with Docling** (supports PDF, DOCX, PPTX, HTML)
- **Proper service layer architecture** following SOLID principles

## Project Structure

```
.
├── app/                    # Main application package
│   ├── __init__.py
│   ├── main.py            # Application entry point
│   ├── api/               # API routes
│   │   ├── __init__.py
│   │   └── routes.py      # API endpoint definitions (HTTP concerns only)
│   ├── core/              # Core application logic
│   │   ├── __init__.py
│   │   ├── config.py      # Configuration settings
│   │   ├── qdrant_client.py # Qdrant client wrapper
│   │   ├── embedding_service.py # Embedding service (document processing logic)
│   │   └── response_service.py # Response generation service (OpenAI integration)
│   ├── schemas/           # Pydantic models for request/response validation
│   │   ├── __init__.py
│   │   └── document.py
│   └── models/            # Database models (if any)
├── requirements.txt      # Project dependencies
├── streamlit_app.py     # Streamlit frontend application
├── test_api_connection.py # API connectivity test script
├── test_chunking.py     # Document chunking test script
├── test_docling_chunking.py # Docling chunking test script
├── test_service_delegation.py # Service delegation test script
├── test_streamlit_updates.py # Streamlit updates test script
├── test_embedding_model.py # Embedding model change test script
├── test_qdrant_dimension.py # Qdrant dimension test script
├── test_response_service.py # Response service test script
├── test_api_endpoints.py # API endpoints test script
├── verify_routes.py # Route verification script
├── troubleshoot_api.py # API troubleshooting script
├── fix_qdrant_collection.py # Script to fix Qdrant collection dimension issues
├── Dockerfile           # Docker configuration for FastAPI app
├── docker-compose.yml   # Multi-container setup (FastAPI + Qdrant + Streamlit)
├── .dockerignore        # Docker ignore file
├── start.sh             # Application startup script
├── run_docker.sh        # Docker startup script
├── start_streamlit.sh   # Streamlit frontend startup script
├── test_endpoints.py    # Test script
└── README.md            # This file
```

## Architecture

This application follows a clean architecture with proper separation of concerns:

- **Routes Layer**: Handles HTTP request/response concerns only
- **Service Layer**: Contains business logic (document processing, embedding creation, response generation)
- **Data Access Layer**: Handles data persistence (Qdrant vector storage)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FastAPI_Endpoint_For_RAG
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running with Python directly

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   Or run directly:
   ```bash
   python -m app.main
   ```

   Or use the startup script:
   ```bash
   ./start.sh
   ```

2. In a separate terminal, start the Streamlit frontend:
   ```bash
   streamlit run streamlit_app.py
   ```

### Running with Docker

1. Build and start all containers (FastAPI, Streamlit, Qdrant):
   ```bash
   docker-compose up --build
   ```

   Or use the startup script:
   ```bash
   ./run_docker.sh
   ```

2. Access the services:
   - FastAPI backend: `http://localhost:8000`
   - Streamlit frontend: `http://localhost:8501`
   - Qdrant dashboard: `http://localhost:6333/dashboard`

## Troubleshooting

### "API is not accessible" error in Streamlit

If you see "❌ API is not accessible" in the Streamlit frontend:

1. **Check if the FastAPI server is running:**
   ```bash
   # Test connection
   python test_api_connection.py
   ```

2. **For local development:**
   - Make sure you've started the FastAPI server:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port 8000
     ```
   - Check that port 8000 is not blocked by a firewall

3. **For Docker deployment:**
   - Make sure all containers are running:
     ```bash
     docker-compose ps
     ```
   - Check container logs for errors:
     ```bash
     docker-compose logs fastapi-app
     ```

4. **Network connectivity:**
   - In Docker, the Streamlit app should connect to `http://fastapi-app:8000`
   - Locally, it should connect to `http://localhost:8000`
   - The app automatically detects the environment, but you can force it:
     ```bash
     # For Docker
     IS_DOCKER=true streamlit run streamlit_app.py
     
     # For local development
     IS_DOCKER=false streamlit run streamlit_app.py
     ```

### "404 Not Found" error for /query-document endpoint

If you see "Error querying documents: 404 Client Error: Not Found for url: http://fastapi-app:8000/api/v1/query-document/":

1. **Verify the endpoint exists:**
   ```bash
   python verify_routes.py
   ```

2. **Check Docker service status:**
   ```bash
   docker-compose ps
   ```

3. **Restart services with fresh build:**
   ```bash
   docker-compose down
   docker-compose up --build
   ```

4. **Check FastAPI logs for startup errors:**
   ```bash
   docker-compose logs fastapi-app
   ```

5. **Verify OpenAI API key is set:**
   Make sure your `.env` file contains:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

### "Requested too many tokens" error

If you encounter the error "Requested X tokens, max 300000 tokens per request":

1. **This is handled automatically** by the updated chunking implementation
2. The system now automatically splits large documents into smaller chunks
3. Each chunk is processed separately to stay within token limits
4. The maximum chunk size is set to 5000 characters

To test the chunking functionality:
```bash
python test_chunking.py
```

### Vector Dimension Mismatch Error

If you see an error like:
```
Wrong input: Vector inserting error: expected dim: 1536, got 384
```

This happens when the Qdrant collection was created with the wrong vector dimension. To fix this:

1. **Run the fix script:**
   ```bash
   python fix_qdrant_collection.py
   ```

2. **Or manually delete and recreate the collection:**
   ```bash
   # Access Qdrant dashboard at http://localhost:6333/dashboard
   # Delete the existing collection
   # Restart the FastAPI service
   ```

3. **Restart the services:**
   ```bash
   docker-compose down
   docker-compose up --build
   ```

## API Endpoints

### Upload Document
```
POST /api/v1/upload-document/
```
Upload a text file to be processed and stored in Qdrant.

**Parameters:**
- `file`: Text file to upload (required)

**Supported file formats:**
- `.txt` - Plain text files
- `.pdf` - PDF documents (processed with Docling)
- `.docx` - Word documents (processed with Docling)
- `.pptx` - PowerPoint presentations (processed with Docling)
- `.html` - HTML documents (processed with Docling)

**Response:**
```json
{
  "message": "Document 'filename.txt' processed and stored successfully",
  "chunks_processed": 5
}
```

### Query Document
```
POST /api/v1/query-document/
```
Search for relevant document chunks using similarity search and generate a well-formatted response using OpenAI.

**Request Body:**
```json
{
  "query": "Your search query here",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "Your search query here",
  "response": "Well-formatted answer based on the retrieved documents",
  "retrieved_documents": [
    {
      "document_name": "filename.txt",
      "chunk_index": 0,
      "text": "Relevant text chunk",
      "score": 0.85
    }
  ]
}
```

## Streamlit Frontend

The Streamlit frontend provides a ChatGPT-like interface for interacting with your documents:

1. Upload documents using the sidebar file uploader
2. Ask questions in the chat interface
3. View retrieved document chunks and their relevance scores

**Features:**
- Chat history persistence during the session
- Document management in the sidebar
- Visual display of retrieved results with similarity scores
- Real-time API status monitoring
- **Support for multiple document formats** (TXT, PDF, DOCX, PPTX, HTML)
- **Well-formatted responses** generated by OpenAI GPT models

**Supported File Formats:**
The Streamlit interface now supports uploading:
- `.txt` - Plain text files
- `.pdf` - PDF documents (processed with Docling for intelligent parsing)
- `.docx` - Word documents (processed with Docling)
- `.pptx` - PowerPoint presentations (processed with Docling)
- `.html` - HTML documents (processed with Docling)

The frontend automatically detects the file type and sends the appropriate content type to the API.

## Cost Benefits

By switching to the `all-MiniLM-L6-v2` open-source embedding model:
- **Zero cost** for embedding generation (no API fees)
- **Faster processing** (runs locally without network latency)
- **Privacy** (no data sent to external services)
- **No rate limits** (process as many documents as needed)

The `all-MiniLM-L6-v2` model provides good quality embeddings while being completely free to use.

For response generation, the application uses OpenAI GPT models which provide high-quality, well-formatted responses based on the retrieved context.

## Testing

Run the test script to verify the endpoints work correctly:
```bash
python test_endpoints.py
```

Test API connectivity:
```bash
python test_api_connection.py
```

Test document chunking:
```bash
python test_chunking.py
```

Test Docling document chunking:
```bash
python test_docling_chunking.py
```

Test service delegation:
```bash
python test_service_delegation.py
```

Test Streamlit updates:
```bash
python test_streamlit_updates.py
```

Test embedding model change:
```bash
python test_embedding_model.py
```

Test Qdrant dimension handling:
```bash
python test_qdrant_dimension.py
```

Test response service:
```bash
python test_response_service.py
```

Test API endpoints:
```bash
python test_api_endpoints.py
```

Verify routes:
```bash
python verify_routes.py
```

Troubleshoot API issues:
```bash
python troubleshoot_api.py
```

Fix Qdrant collection dimension issues:
```bash
python fix_qdrant_collection.py
```

## Configuration

The application can be configured using environment variables. See [config.py](app/core/config.py) for available settings.

## SOLID Principles Implementation

This project follows SOLID principles:

1. **Single Responsibility Principle**: Each class and module has a single responsibility
2. **Open/Closed Principle**: Classes are open for extension but closed for modification
3. **Liskov Substitution Principle**: Subtypes can substitute their base types
4. **Interface Segregation Principle**: Clients only depend on the interfaces they use
5. **Dependency Inversion Principle**: High-level modules don't depend on low-level modules

## Security Best Practices

- Input validation for all API endpoints
- Proper error handling without exposing sensitive information
- File type validation for uploads
- Secure dependency management
- Docker security best practices
- Environment variable management for configuration

## License

This project is licensed under the MIT License.