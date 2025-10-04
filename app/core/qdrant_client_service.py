from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self):
        self.client = None
        self.collection_name = settings.COLLECTION_NAME

    def initialize_client(self):
        """Initialize Qdrant client and create collection if it doesn't exist"""
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
            )

            if settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2":
                vector_dimension = 384  # all-MiniLM-L6-v2 dimension
            else:
                vector_dimension = 1536  # Default for OpenAI embeddings

            try:
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")

                existing_dimension = collection_info.config.params.vectors.size
                if existing_dimension != vector_dimension:
                    logger.warning(
                        f"Collection has incorrect dimension: expected {vector_dimension}, got {existing_dimension}"
                    )
                    logger.warning(
                        "You may need to delete the existing collection and recreate it with the correct dimension"
                    )
            except Exception as e:
                logger.info(
                    f"Collection '{self.collection_name}' does not exist, creating it..."
                )
                # Create collection with the correct vector dimension
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_dimension, distance=Distance.COSINE
                    ),
                )
                logger.info(
                    f"Collection '{self.collection_name}' created successfully with {vector_dimension} dimensions"
                )

        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise

    def upsert_points(self, points):
        """Upsert points to Qdrant collection"""
        if not self.client:
            raise Exception("Qdrant client not initialized")

        return self.client.upsert(collection_name=self.collection_name, points=points)

    def search_points(self, query_vector, limit=10):
        """Search points in Qdrant collection"""
        if not self.client:
            raise Exception("Qdrant client not initialized")

        return self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=limit
        )
