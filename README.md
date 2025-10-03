# FastAPI RAG Endpoint with Qdrant

A FastAPI application that provides endpoints for uploading text documents, creating vector embeddings using OpenAI, and performing similarity search using Qdrant as the vector database.

## Features

- Upload text files via API endpoint
- Automatically create vector embeddings using OpenAI Embeddings API
- Store embeddings in Qdrant vector database
- Query documents using similarity search
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
│   │   └── embedding_service.py # Embedding service (document processing logic)
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
- **Service Layer**: Contains business logic (document processing, embedding creation)
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

### "Requested too many tokens" error

If you encounter the error "Requested X tokens, max 300000 tokens per request":

1. **This is handled automatically** by the updated chunking implementation
2. The system now automatically splits large documents into smaller chunks
3. Each chunk is processed separately to stay within OpenAI's token limits
4. The maximum chunk size is set to 5000 characters (well below the token limit)

To test the chunking functionality:
```bash
python test_chunking.py
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
Search for relevant document chunks using similarity search.

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
  "results": [
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

**Supported File Formats:**
The Streamlit interface now supports uploading:
- `.txt` - Plain text files
- `.pdf` - PDF documents (processed with Docling for intelligent parsing)
- `.docx` - Word documents (processed with Docling)
- `.pptx` - PowerPoint presentations (processed with Docling)
- `.html` - HTML documents (processed with Docling)

The frontend automatically detects the file type and sends the appropriate content type to the API.

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
- Environment variable management for API keys

## License

This project is licensed under the MIT License.