from openai import OpenAI
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.client = None
        self.model_name = settings.OPENAI_EMBEDDING_MODEL

    def initialize_model(self):
        """Initialize the OpenAI client"""
        try:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in the configuration")

            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(f"OpenAI client initialized with model '{self.model_name}'")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise

    def encode_text(self, texts):
        """Encode texts into embeddings using OpenAI"""
        if not self.client:
            raise Exception("OpenAI client not initialized")

        if isinstance(texts, str):
            texts = [texts]

        # Check if any text is too long and split if necessary
        processed_texts = []
        for text in texts:
            if len(text) > 10000:  # Rough estimate for token limit
                # Split long text into smaller chunks
                chunks = self._split_long_text(text)
                processed_texts.extend(chunks)
            else:
                processed_texts.append(text)

        try:
            response = self.client.embeddings.create(
                input=processed_texts, model=self.model_name
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            return embeddings

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def _split_long_text(self, text, max_length=8000):
        """
        Split a long text into smaller chunks.
        This is a simple implementation that splits on sentences.
        """
        if len(text) <= max_length:
            return [text]

        # Split by sentences (looking for sentence endings)
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in ".!?" and len(current_sentence) > max_length // 2:
                sentences.append(current_sentence.strip())
                current_sentence = ""

        # Add the last sentence if it exists
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # If we still have very long sentences, split them
        final_chunks = []
        for sentence in sentences:
            if len(sentence) > max_length:
                # Split long sentence into chunks
                for i in range(0, len(sentence), max_length):
                    final_chunks.append(sentence[i : i + max_length])
            else:
                final_chunks.append(sentence)

        return final_chunks

    def chunk_text(self, text, max_chunk_size=5000):
        """
        Split text into chunks of specified maximum size.
        Tries to split on sentence boundaries when possible.
        """
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Get a chunk of the maximum size
            end_pos = min(current_pos + max_chunk_size, len(text))
            chunk = text[current_pos:end_pos]

            # Try to find a good breaking point (sentence end or paragraph)
            if end_pos < len(text):  # Not the last chunk
                # Look for sentence endings (.!? followed by space)
                sentence_endings = [". ", "! ", "? "]
                best_break = -1

                for ending in sentence_endings:
                    # Search backwards from the end for a sentence ending
                    pos = chunk.rfind(ending)
                    if pos > len(chunk) // 2:  # Make sure it's in the second half
                        best_break = max(best_break, pos + 1)

                # If no sentence ending found, look for paragraph breaks
                if best_break == -1:
                    para_break = chunk.rfind("\n\n")
                    if para_break > len(chunk) // 2:
                        best_break = para_break + 2

                # If still no good break, look for any whitespace
                if best_break == -1:
                    best_break = chunk.rfind(" ")

                # If we found a breaking point, use it
                if best_break > 0:
                    chunk = chunk[:best_break]
                    current_pos += best_break
                else:
                    # Force break at max_chunk_size
                    current_pos += max_chunk_size
            else:
                # Last chunk
                current_pos = end_pos

            chunks.append(chunk)

        return chunks
