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

            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")
            except (
                Exception
            ) as e:  # Fixed: Catch specific exception instead of bare except
                logger.info(
                    f"Collection '{self.collection_name}' does not exist, creating it..."
                )
                # OpenAI embeddings have 1536 dimensions
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info(
                    f"Collection '{self.collection_name}' created successfully with 1536 dimensions"
                )

        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise

    def upsert_points(self, points):
        """Upsert points to Qdrant collection"""
        if not self.client:
            raise Exception("Qdrant client not initialized")

        return self.client.upsert(collection_name=self.collection_name, points=points)

    def search_points(self, query_vector, limit=5):
        """Search points in Qdrant collection"""
        if not self.client:
            raise Exception("Qdrant client not initialized")

        return self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=limit
        )
