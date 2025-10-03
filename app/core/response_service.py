import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class ResponseService:
    def __init__(self):
        self.client = None

    def initialize_client(self):
        """Initialize the OpenAI client for response generation"""
        try:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in the configuration")

            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for response generation")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise

    def generate_response(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        model: str = "gpt-3.5-turbo",
    ) -> str:
        """
        Generate a well-formatted response using OpenAI based on the query and retrieved documents.

        Args:
            query (str): The user's query
            retrieved_documents (List[Dict[str, Any]]): List of retrieved documents with text content
            model (str): The OpenAI model to use for generation

        Returns:
            str: Generated response
        """
        if not self.client:
            raise Exception("OpenAI client not initialized")

        if not retrieved_documents:
            return "I couldn't find any relevant information to answer your question."

        try:
            # Format the context from retrieved documents
            context = self._format_context(retrieved_documents)

            # Create the prompt
            prompt = self._create_prompt(query, context)

            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Always be accurate and concise.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            generated_response = response.choices[0].message.content
            if generated_response:
                generated_response = generated_response.strip()
            else:
                generated_response = (
                    "I couldn't generate a response based on the provided information."
                )

            logger.info("Response generated successfully using OpenAI")
            return generated_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to a simple response if OpenAI fails
            return self._create_fallback_response(retrieved_documents)

    def _format_context(self, retrieved_documents: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved documents into a context string.

        Args:
            retrieved_documents (List[Dict[str, Any]]): List of retrieved documents

        Returns:
            str: Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(retrieved_documents, 1):
            content = doc.get("text", "")
            source = doc.get("document_name", "Unknown")
            score = doc.get("score", 0)

            context_parts.append(
                f"Document {i} (Source: {source}, Relevance: {score:.2f}):\n{content}"
            )

        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the OpenAI model.

        Args:
            query (str): The user's query
            context (str): Formatted context from retrieved documents

        Returns:
            str: Complete prompt for the model
        """
        prompt = f"""
Answer the question based only on the following context:

{context}

Question: {query}

Instructions:
1. Use only the information from the provided context
2. If the context doesn't contain enough information, say so
3. Provide a clear, concise, and well-structured answer
4. If relevant, mention the source documents
5. Do not make up information not present in the context

Answer:
"""
        return prompt

    def _create_fallback_response(
        self, retrieved_documents: List[Dict[str, Any]]
    ) -> str:
        """
        Create a fallback response when OpenAI generation fails.

        Args:
            retrieved_documents (List[Dict[str, Any]]): List of retrieved documents

        Returns:
            str: Fallback response with raw document information
        """
        if not retrieved_documents:
            return "I couldn't find any relevant information to answer your question."

        response = "Here's what I found in the documents:\n\n"
        for i, doc in enumerate(retrieved_documents, 1):
            content = (
                doc.get("text", "")[:200] + "..."
                if len(doc.get("text", "")) > 200
                else doc.get("text", "")
            )
            source = doc.get("document_name", "Unknown")
            response += f"{i}. From {source}: {content}\n\n"

        return response
