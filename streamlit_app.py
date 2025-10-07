import streamlit as st
import requests
import json
from typing import List, Dict, Optional, Any
import time
import os
import logging
from openai import OpenAI

# Configuration - Handle both Docker and local environments
# Use environment variable to determine if running in Docker
IS_DOCKER = os.getenv("IS_DOCKER", "false").lower() == "true"

if IS_DOCKER:
    # Internal Docker network communication
    API_BASE_URL = "http://fastapi-app:8000/api/v1"
else:
    # External/local access
    API_BASE_URL = "http://localhost:8000/api/v1"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"


def check_api_status():
    """Check if the API is accessible and update session state"""
    try:
        response = requests.get(API_BASE_URL.replace("/api/v1", "/"), timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = "connected"
            return True
        else:
            st.session_state.api_status = "error"
            return False
    except requests.exceptions.ConnectionError:
        st.session_state.api_status = "disconnected"
        return False
    except requests.exceptions.Timeout:
        st.session_state.api_status = "timeout"
        return False
    except Exception as e:
        st.session_state.api_status = "error"
        return False


def upload_document(file) -> Optional[Dict[str, Any]]:
    """Upload a document to the RAG system"""
    if st.session_state.api_status != "connected":
        if not check_api_status():
            st.error(
                "API is not accessible. Please make sure the FastAPI server is running."
            )
            return None

    try:
        # Determine content type based on file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".pdf":
            content_type = "application/pdf"
        elif file_extension == ".docx":
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif file_extension == ".pptx":
            content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        elif file_extension == ".html":
            content_type = "text/html"
        else:
            content_type = "text/plain"

        files = {"file": (file.name, file, content_type)}
        response = requests.post(
            f"{API_BASE_URL}/upload-document/", files=files, timeout=600
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. Please make sure the FastAPI server is running."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading document: {str(e)}")
        return None


def upload_video(file) -> Optional[Dict[str, Any]]:
    """Upload a video to the RAG system"""
    if st.session_state.api_status != "connected":
        if not check_api_status():
            st.error(
                "API is not accessible. Please make sure the FastAPI server is running."
            )
            return None

    try:
        # Determine content type based on file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".mp4":
            content_type = "video/mp4"
        elif file_extension == ".avi":
            content_type = "video/x-msvideo"
        elif file_extension == ".mov":
            content_type = "video/quicktime"
        elif file_extension == ".mkv":
            content_type = "video/x-matroska"
        else:
            content_type = "video/unknown"

        files = {"file": (file.name, file, content_type)}
        response = requests.post(
            f"{API_BASE_URL}/upload-video/", files=files, timeout=600
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. Please make sure the FastAPI server is running."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading video: {str(e)}")
        return None


def query_documents(query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Query documents in the RAG system and generate response"""
    if st.session_state.api_status != "connected":
        if not check_api_status():
            st.error(
                "API is not accessible. Please make sure the FastAPI server is running."
            )
            return None

    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(
            f"{API_BASE_URL}/query-document/", json=payload, timeout=600
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. Please make sure the FastAPI server is running."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying documents: {str(e)}")
        return None


def query_video_transcripts(query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Query video transcripts in the RAG system and generate response"""
    if st.session_state.api_status != "connected":
        if not check_api_status():
            st.error(
                "API is not accessible. Please make sure the FastAPI server is running."
            )
            return None

    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(
            f"{API_BASE_URL}/query-video-transcripts/", json=payload, timeout=600
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. Please make sure the FastAPI server is running."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying video transcripts: {str(e)}")
        return None


def format_document_preview(content: str, max_length: int = 300) -> str:
    """Format document content for preview display"""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def display_document_results(documents: List[Dict[str, Any]], expanded: bool = False):
    """Display document results in a structured format"""
    st.markdown("### 📚 Retrieved Documents")

    # Show summary statistics
    st.markdown(
        f"**Found {len(documents)} relevant document{'s' if len(documents) != 1 else ''}**"
    )

    # Create tabs for better organization if there are multiple documents
    if len(documents) > 1:
        tabs = st.tabs([f"Document {i+1}" for i in range(len(documents))])
        for i, (tab, doc) in enumerate(zip(tabs, documents)):
            with tab:
                display_single_document(doc, i + 1, expanded)
    else:
        # Single document view
        for i, doc in enumerate(documents):
            display_single_document(doc, i + 1, expanded)


def display_video_results(videos: List[Dict[str, Any]], expanded: bool = False):
    """Display video results in a structured format"""
    st.markdown("### 🎥 Retrieved Videos")

    # Show summary statistics
    st.markdown(
        f"**Found {len(videos)} relevant video{'s' if len(videos) != 1 else ''}**"
    )

    # Create tabs for better organization if there are multiple videos
    if len(videos) > 1:
        tabs = st.tabs([f"Video {i+1}" for i in range(len(videos))])
        for i, (tab, video) in enumerate(zip(tabs, videos)):
            with tab:
                display_single_video(video, i + 1, expanded)
    else:
        # Single video view
        for i, video in enumerate(videos):
            display_single_video(video, i + 1, expanded)


def display_video_transcript_results(
    video_transcripts: List[Dict[str, Any]], expanded: bool = False
):
    """Display video transcript results in a structured format"""
    st.markdown("### 🎬 Retrieved Video Transcripts")

    # Show summary statistics
    st.markdown(
        f"**Found {len(video_transcripts)} relevant transcript segment{'s' if len(video_transcripts) != 1 else ''}**"
    )

    # Create tabs for better organization if there are multiple transcript segments
    if len(video_transcripts) > 1:
        tabs = st.tabs([f"Segment {i+1}" for i in range(len(video_transcripts))])
        for i, (tab, transcript) in enumerate(zip(tabs, video_transcripts)):
            with tab:
                display_single_video_transcript(transcript, i + 1, expanded)
    else:
        # Single transcript view
        for i, transcript in enumerate(video_transcripts):
            display_single_video_transcript(transcript, i + 1, expanded)


def display_single_document(doc: Dict[str, Any], index: int, expanded: bool = False):
    """Display a single document in a structured format"""
    # Document header with score
    score_percentage = min(100, max(0, int(doc["score"] * 100)))
    st.markdown(f"**📄 Document:** `{doc['document_name']}`")
    st.progress(
        score_percentage / 100,
        text=f"Relevance Score: {doc['score']:.4f} ({score_percentage}%)",
    )

    # Document content
    with st.expander("📄 **Document Content**", expanded=expanded):
        # Format content for better readability
        content = doc["text"]
        if len(content) > 500:
            # For longer content, show a preview with option to expand
            st.markdown("**Preview:**")
            st.markdown(format_document_preview(content, 500))
            st.markdown("---")
            st.markdown("**Full Content:**")
            st.markdown(content)
        else:
            st.markdown(content)

    # Document metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Chunk Index:** {doc['chunk_index']}")
    with col2:
        st.markdown(f"**Characters:** {len(doc['text'])}")


def display_single_video(video: Dict[str, Any], index: int, expanded: bool = False):
    """Display a single video in a structured format"""
    # Video header with score
    score_percentage = min(100, max(0, int(video["score"] * 100)))
    # st.markdown(f"**🎥 Video:** `{video['video_file_name']}`")
    st.progress(
        score_percentage / 100,
        text=f"Relevance Score: {video['score']:.4f} ({score_percentage}%)",
    )

    # Video description
    with st.expander("🎥 **Video Description**", expanded=expanded):
        st.markdown(video["description"])

    # Video metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Tags:** {', '.join(video['tags'])}")
    with col2:
        if video["duration"]:
            st.markdown(f"**Duration:** {video['duration']} seconds")
        else:
            st.markdown("**Duration:** Not specified")
    with col3:
        st.markdown(f"**Characters:** {len(video['description'])}")


def display_single_video_transcript(
    transcript: Dict[str, Any], index: int, expanded: bool = False
):
    """Display a single video transcript segment in a structured format"""
    # Transcript header with score
    score_percentage = min(100, max(0, int(transcript["score"] * 100)))
    st.markdown(
        f"**🎬 Transcript Segment:** `{transcript.get('document_name', 'Video Transcript Segment')}`"
    )
    st.progress(
        score_percentage / 100,
        text=f"Relevance Score: {transcript['score']:.4f} ({score_percentage}%)",
    )

    # Transcript content
    with st.expander("🎬 **Transcript Content**", expanded=expanded):
        st.markdown(transcript["text"])  # Changed from 'description' to 'text'

    # Transcript metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Chunk Index:** {transcript.get('chunk_index', 'N/A')}")
    with col2:
        st.markdown(
            f"**Characters:** {len(transcript['text'])}"
        )  # Changed from 'description' to 'text'


def main():
    st.set_page_config(page_title="RAG Chat Interface", page_icon="💬", layout="wide")

    st.title("💬 RAG Chat Interface")
    st.caption("🚀 Intelligent document and video search with AI-powered responses")

    # Check API status on app start or refresh
    if st.session_state.api_status == "unknown":
        check_api_status()

    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Content Management")

        # Display API status with more details
        st.subheader("📡 API Status")
        if st.session_state.api_status == "connected":
            st.success("✅ Connected to API")
            st.caption(f"Endpoint: {API_BASE_URL}")
        elif st.session_state.api_status == "disconnected":
            st.error("❌ Disconnected")
            st.caption(f"Trying to reach: {API_BASE_URL}")
            if st.button("Retry Connection"):
                check_api_status()
        elif st.session_state.api_status == "timeout":
            st.warning("⏰ Connection Timeout")
            st.caption(f"Endpoint: {API_BASE_URL}")
        else:
            st.warning("❓ Unknown Status")
            st.caption(f"Configured endpoint: {API_BASE_URL}")

        st.divider()

        # Updated file uploader to support multiple document formats
        st.subheader("📤 Upload Documents")
        st.caption("Supported formats: TXT, PDF, DOCX, PPTX, HTML")
        uploaded_document = st.file_uploader(
            "Choose a document",
            type=["txt", "pdf", "docx", "pptx", "html"],
            key="document_uploader",
        )

        if (
            uploaded_document is not None
            and st.session_state.uploaded_file != uploaded_document.name
        ):
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_document)
                if result:
                    st.success(f"✅ {result['message']}")
                    st.session_state.uploaded_file = uploaded_document.name
                else:
                    st.error("Failed to upload document")

        st.divider()

        # Video uploader
        st.subheader("📹 Upload Videos")
        st.caption("Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WEBM")
        uploaded_video = st.file_uploader(
            "Choose a video",
            type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
            key="video_uploader",
        )

        if (
            uploaded_video is not None
            and st.session_state.uploaded_file != uploaded_video.name
        ):
            with st.spinner("Processing video..."):
                result = upload_video(uploaded_video)
                if result:
                    st.success(f"✅ {result['message']}")
                    st.session_state.uploaded_file = uploaded_video.name
                else:
                    st.error("Failed to upload video")

        st.divider()

        st.subheader("⚙️ Setup Instructions")
        if IS_DOCKER:
            st.info("Running in Docker mode")
            st.markdown(
                """
            1. Make sure all services are running:
               ```bash
               docker-compose up
               ```
            2. The API should be accessible at:
               `http://fastapi-app:8000`
            """
            )
        else:
            st.info("Running in local mode")
            st.markdown(
                """
            1. Start the FastAPI server:
               ```bash
               uvicorn app.main:app --host 0.0.0.0 --port 8000
               ```
            2. The API should be accessible at:
               `http://localhost:8000`
            """
            )

    # Main chat interface
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # For assistant messages, display the response and content separately
                st.markdown("### 🤖 AI Response")
                st.markdown(message["content"])

                # Display retrieved documents if available
                if "documents" in message and message["documents"]:
                    st.markdown("---")
                    display_document_results(message["documents"])

                # Display retrieved videos if available
                if "videos" in message and message["videos"]:
                    st.markdown("---")
                    display_video_results(message["videos"])

                # Display retrieved video transcripts if available
                if "video_transcripts" in message and message["video_transcripts"]:
                    st.markdown("---")
                    display_video_transcript_results(message["video_transcripts"])
            else:
                # For user messages, display normally
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents or videos..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                # Query both documents and videos
                document_response = query_documents(prompt)
                # video_response = query_videos(prompt)  # Removed since we're not using the query-videos endpoint
                video_transcript_response = query_video_transcripts(prompt)

                # Combine results
                response_text = ""
                documents = []
                # videos = []  # Removed since we're not using the query-videos endpoint
                video_transcripts = []

                if document_response:
                    response_text = document_response["response"]
                    documents = document_response["retrieved_documents"]

                # if video_response:  # Removed since we're not using the query-videos endpoint
                #     videos = video_response["results"]

                if video_transcript_response:
                    # For video transcripts, we now get a generated response
                    response_text = video_transcript_response["response"]
                    video_transcripts = video_transcript_response["retrieved_documents"]

                if response_text or documents or video_transcripts:
                    # Display the generated response in a structured format
                    if response_text:
                        st.markdown("### 🤖 AI Response")
                        st.markdown(response_text)

                    # Display retrieved documents in a structured format
                    if documents:
                        st.markdown("---")
                        display_document_results(documents)

                    # Display retrieved videos in a structured format
                    # if videos:  # Removed since we're not using the query-videos endpoint
                    #     st.markdown("---")
                    #     display_video_results(videos)

                    # Display retrieved video transcripts in a structured format
                    if video_transcripts:
                        st.markdown("---")
                        display_video_transcript_results(video_transcripts)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                response_text
                                if response_text
                                else "No specific response generated."
                            ),
                            "documents": documents,
                            # "videos": videos,  # Removed since we're not using the query-videos endpoint
                            "video_transcripts": video_transcripts,
                        }
                    )
                else:
                    st.error(
                        "Failed to get response from the API. Please check the API status in the sidebar."
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Sorry, I encountered an error while processing your request.",
                        }
                    )


if __name__ == "__main__":
    main()
