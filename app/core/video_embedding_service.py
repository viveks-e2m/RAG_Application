import logging
from typing import List, Union, Optional

from app.core.config import settings
from app.core.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VideoEmbeddingService:
    """
    Service for handling video embeddings.
    Follows the Single Responsibility Principle by focusing only on video embedding operations.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()

    def initialize_model(self):
        """Initialize the sentence-transformers model through the embedding service"""
        try:
            self.embedding_service.initialize_model()
            logger.info(
                "Video Embedding Service: Model initialized through embedding service"
            )
        except Exception as e:
            logger.error(f"Error initializing model through embedding service: {e}")
            raise

    def encode_video_descriptions(
        self, descriptions: Union[str, List[str]]
    ) -> List[List[float]]:
        """Encode video descriptions into embeddings using the existing embedding service"""
        try:
            # Use the existing embedding service to create embeddings
            embeddings = self.embedding_service.encode_text(descriptions)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating video embeddings: {e}")
            raise

    def process_video_metadata(
        self,
        video_file_name: str,
        description: str,
        duration: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> dict:
        """
        Process video metadata and generate embeddings for the description.

        Args:
            video_file_name (str): Name of the video file
            description (str): Description of the video content
            duration (float, optional): Duration of the video in seconds
            tags (List[str], optional): Tags associated with the video

        Returns:
            dict: Processed video metadata with embeddings
        """
        try:
            # Generate embedding for the description
            description_embedding = self.encode_video_descriptions([description])[0]

            # Prepare metadata
            metadata = {
                "video_file_name": video_file_name,
                "description": description,
                "description_embedding": description_embedding,
                "duration": duration,
                "tags": tags or [],
            }

            logger.info(f"Video metadata processed for '{video_file_name}'")
            return metadata

        except Exception as e:
            logger.error(
                f"Error processing video metadata for '{video_file_name}': {e}"
            )
            raise
