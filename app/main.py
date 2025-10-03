from fastapi import FastAPI
from app.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="FastAPI RAG Endpoint with Qdrant",
        description="API for uploading text files, creating embeddings, and querying with RAG",
        version="1.0.0",
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {"message": "FastAPI RAG Endpoint with Qdrant"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
