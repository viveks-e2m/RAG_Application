import logging
from typing import List, Union

from langchain_docling import DoclingLoader
from docling.chunking import HierarchicalChunker
from langchain_docling.loader import ExportType
from sentence_transformers import SentenceTransformer

from app.core.config import settings


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.model = None
        self.model_name = settings.EMBEDDING_MODEL

    def initialize_model(self):
        """Initialize the sentence-transformers model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"SentenceTransformer model initialized with model '{self.model_name}'")
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer model: {e}")
            raise

    def encode_text(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Encode texts into embeddings using sentence-transformers"""
        if not self.model:
            raise Exception("SentenceTransformer model not initialized")

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
            # Generate embeddings using sentence-transformers
            embeddings = self.model.encode(processed_texts, convert_to_numpy=False)
            # Convert to list format if needed
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
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

    def chunk_text_with_docling(self, file_path):
        """
        Split text into chunks using Docling's HierarchicalChunker.
        This method is specifically for document files (PDF, DOCX, etc.).
        """

        try:
            # Initialize Docling loader with HierarchicalChunker
            loader = DoclingLoader(
                file_path=file_path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HierarchicalChunker(),
            )

            # Load documents
            docs = loader.load()

            # Extract text content from documents
            chunks = [doc.page_content for doc in docs if doc.page_content.strip()]

            logger.info(f"Docling chunking completed. Generated {len(chunks)} chunks.")
            return chunks

        except Exception as e:
            logger.error(f"Error during Docling chunking: {e}")
            raise

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
        
    def process_uploaded_file(self, file_path, file_extension, content=None):
        """
        Process an uploaded file completely - from file to chunks to embeddings.
        This method handles the entire document processing workflow.

        Args:
            file_path (str): Path to the uploaded file
            file_extension (str): File extension (e.g., '.pdf', '.txt')
            content (bytes, optional): File content for text files

        Returns:
            list: List of processed chunks
        """
        # Process document based on file type
        if file_extension in [".pdf", ".docx", ".pptx", ".html"]:
            # Use Docling for advanced document chunking
            chunks = self.chunk_text_with_docling(file_path)
        else:
            # Use existing text chunking for .txt files
            if content is None:
                # Read content from file if not provided
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = content.decode("utf-8")
            chunks = self.chunk_text(text, max_chunk_size=5000)

        # Process chunks (remove empty ones and strip whitespace)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        return chunks