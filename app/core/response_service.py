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
        prompt = f"""
Answer the question comprehensively based only on the following context:

{context}

Question: {query}

Instructions for providing a detailed response:
1. Use ONLY the information from the provided context - do not make up information
2. If the context doesn't contain enough information to fully answer, clearly state what is missing
3. Provide a comprehensive and detailed answer with thorough explanation
4. Structure your response with clear headings, subheadings, bullet points, and numbered lists where appropriate
5. Include specific details, examples, direct quotes, and references from the context when relevant
6. If relevant, mention the source documents and their relevance to your answer
7. Aim for a response of at least 4-6 substantial paragraphs for complex questions
8. Use a professional, educational, and helpful tone
9. Organize information logically with proper flow between ideas
10. Highlight key points and important concepts
11. Address all aspects of the question thoroughly
12. Conclude with a summary of the main points if appropriate

Format your response with:
- A clear introduction that addresses the main question
- Well-organized body paragraphs with supporting details
- Bullet points or numbered lists for enumerating items or steps
- Direct quotes from the context when they add value
- A conclusion that summarizes key findings

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
        prompt = f"""
Answer the question comprehensively based only on the following video transcript segments:

{context}

Question: {query}

Instructions for providing a detailed response:
1. Use ONLY the information from the provided video transcript segments - do not make up information
2. If the transcript segments don't contain enough information to fully answer, clearly state what is missing
3. Provide a comprehensive and detailed answer with thorough explanation
4. Structure your response with clear headings, subheadings, bullet points, and numbered lists where appropriate
5. Include specific details, examples, direct quotes, and references from the transcript segments when relevant
6. Mention which transcript segments were most relevant to your answer
7. Aim for a response of at least 3-5 substantial paragraphs for complex questions
8. Use a professional, educational, and helpful tone
9. Organize information logically with proper flow between ideas
10. Highlight key points and important concepts from the video content
11. Address all aspects of the question thoroughly
12. Conclude with a summary of the main points if appropriate

Format your response with:
- A clear introduction that addresses the main question
- Well-organized body paragraphs with supporting details from the video
- Bullet points or numbered lists for enumerating items or steps
- Direct quotes from the transcript when they add value
- A conclusion that summarizes key findings

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
