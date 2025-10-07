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

            # Generate response using OpenAI with increased token limit for longer responses
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research assistant that provides comprehensive, detailed, and well-structured answers. Your responses should be thorough, informative, and educational. Always prioritize accuracy and clarity. Structure your responses with clear headings, subheadings, bullet points, and numbered lists where appropriate. Include specific examples, quotes, and references from the provided context. Aim for responses that are 3-5 paragraphs long for complex questions, ensuring you fully address all aspects of the query.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1500,  # Increased from 1000 to 1500 for even more detailed responses
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

    def generate_response_from_transcripts(
        self,
        query: str,
        retrieved_transcripts: List[Dict[str, Any]],
        model: str = "gpt-3.5-turbo",
    ) -> str:
        """
        Generate a well-formatted response using OpenAI based on the query and retrieved video transcripts.

        Args:
            query (str): The user's query
            retrieved_transcripts (List[Dict[str, Any]]): List of retrieved video transcripts with text content
            model (str): The OpenAI model to use for generation

        Returns:
            str: Generated response
        """
        if not self.client:
            raise Exception("OpenAI client not initialized")

        if not retrieved_transcripts:
            return "I couldn't find any relevant video transcript information to answer your question."

        try:
            # Format the context from retrieved transcripts
            context = self._format_transcript_context(retrieved_transcripts)

            # Create the prompt
            prompt = self._create_transcript_prompt(query, context)

            # Generate response using OpenAI with increased token limit for longer responses
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research assistant that provides comprehensive, detailed, and well-structured answers based on video transcript content. Your responses should be thorough, informative, and educational. Always prioritize accuracy and clarity. Structure your responses with clear headings, subheadings, bullet points, and numbered lists where appropriate. Include specific examples, quotes, and references from the provided video transcript segments. Aim for responses that are 3-5 paragraphs long for complex questions, ensuring you fully address all aspects of the query.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1500,
            )

            generated_response = response.choices[0].message.content
            if generated_response:
                generated_response = generated_response.strip()
            else:
                generated_response = (
                    "I couldn't generate a response based on the provided video transcript information."
                )

            logger.info("Response generated successfully from video transcripts using OpenAI")
            return generated_response

        except Exception as e:
            logger.error(f"Error generating response from transcripts: {e}")
            # Fallback to a simple response if OpenAI fails
            return self._create_transcript_fallback_response(retrieved_transcripts)

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

    def _format_transcript_context(self, retrieved_transcripts: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved video transcripts into a context string.

        Args:
            retrieved_transcripts (List[Dict[str, Any]]): List of retrieved video transcripts

        Returns:
            str: Formatted context string
        """
        context_parts = []
        for i, transcript in enumerate(retrieved_transcripts, 1):
            content = transcript.get("description", "")  # Video transcripts use 'description' instead of 'text'
            score = transcript.get("score", 0)

            context_parts.append(
                f"Video Transcript Segment {i} (Relevance: {score:.2f}):\n{content}"
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
        # Create the prompt
        prompt = f"""
                        You are answering a question based only on the following retrived documents:

                        {context}

                        Question: {query}

                        Guidelines for your answer:
                        - Use ONLY the information from the retrived documents provided. Do not add outside knowledge or make up details.  
                        - If the retrived documents does not provide enough information to fully answer, clearly mention what is missing.  
                        - Write in a natural, conversational, and helpful tone—avoid rigid sections or forced formatting.  
                        - Provide a clear, detailed, and coherent response that flows naturally, as if you are explaining directly to someone.  
                        - Highlight key details from the retrived documents when relevant, but weave them smoothly into your explanation.  
                        - Aim for completeness: address all aspects of the question as much as the retrived documents allows.  
                        - If specific retrived documents are particularly relevant, you may refer to them casually.  

                        Answer:
                        """
        return prompt

    def _create_transcript_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the OpenAI model based on video transcript context.

        Args:
            query (str): The user's query
            context (str): Formatted context from retrieved video transcripts

        Returns:
            str: Complete prompt for the model
        """
        # Create the prompt
        prompt = f"""
                        You are answering a question based only on the following video transcript segments:

                        {context}

                        Question: {query}

                        Guidelines for your answer:
                        - Use ONLY the information from the transcript segments provided. Do not add outside knowledge or make up details.  
                        - If the transcript does not provide enough information to fully answer, clearly mention what is missing.  
                        - Write in a natural, conversational, and helpful tone—avoid rigid sections or forced formatting.  
                        - Provide a clear, detailed, and coherent response that flows naturally, as if you are explaining directly to someone.  
                        - Highlight key details from the transcript when relevant, but weave them smoothly into your explanation.  
                        - Aim for completeness: address all aspects of the question as much as the transcript allows.  
                        - If specific transcript segments are particularly relevant, you may refer to them casually (e.g., "In one part of the video, it was mentioned that...").  

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
            content = doc.get("text", "")
            source = doc.get("document_name", "Unknown")
            score = doc.get("score", 0)

            # Provide more detailed information in the fallback response
            response += (
                f"Document {i} (Source: {source}, Relevance Score: {score:.4f}):\n"
            )
            response += f"Content: {content}\n\n"

        response += "Please note: This is a fallback response. The AI-generated response would provide a more structured and detailed answer based on this information."
        return response

    def _create_transcript_fallback_response(
        self, retrieved_transcripts: List[Dict[str, Any]]
    ) -> str:
        """
        Create a fallback response when OpenAI generation fails for video transcripts.

        Args:
            retrieved_transcripts (List[Dict[str, Any]]): List of retrieved video transcripts

        Returns:
            str: Fallback response with raw transcript information
        """
        if not retrieved_transcripts:
            return "I couldn't find any relevant video transcript information to answer your question."

        response = "Here's what I found in the video transcripts:\n\n"
        for i, transcript in enumerate(retrieved_transcripts, 1):
            content = transcript.get("description", "")  # Video transcripts use 'description' instead of 'text'
            score = transcript.get("score", 0)

            # Provide more detailed information in the fallback response
            response += (
                f"Video Transcript Segment {i} (Relevance Score: {score:.4f}):\n"
            )
            response += f"Content: {content}\n\n"

        response += "Please note: This is a fallback response. The AI-generated response would provide a more structured and detailed answer based on this video transcript information."
        return response
