import streamlit as st
import requests
import json
from typing import List, Dict, Optional, Any
import time
import os

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


def query_documents(query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Query documents in the RAG system"""
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


def main():
    st.set_page_config(page_title="RAG Chat Interface", page_icon="💬", layout="wide")

    st.title("💬 RAG Chat Interface")
    st.caption("🚀 Interface for your document search system")

    # Check API status on app start or refresh
    if st.session_state.api_status == "unknown":
        check_api_status()

    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Management")

        # Display API status with more details
        st.subheader("API Status")
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
        st.subheader("Upload Documents")
        st.caption("Supported formats: TXT, PDF, DOCX, PPTX, HTML")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["txt", "pdf", "docx", "pptx", "html"], 
            key="file_uploader"
        )

        if (
            uploaded_file is not None
            and st.session_state.uploaded_file != uploaded_file.name
        ):
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file)
                if result:
                    st.success(f"✅ {result['message']}")
                    st.session_state.uploaded_file = uploaded_file.name
                else:
                    st.error("Failed to upload document")

        st.divider()

        st.subheader("Setup Instructions")
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
            st.markdown(message["content"])
            # If it's an assistant message with results, display them
            if "results" in message:
                st.markdown("---")
                st.markdown("**Retrieved Documents:**")
                for i, result in enumerate(message["results"], 1):
                    with st.expander(f"Document {i} (Score: {result['score']:.4f})"):
                        st.markdown(f"**File:** {result['document_name']}")
                        st.markdown(f"**Content:** {result['text']}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_documents(prompt)

                if response:
                    # Extract the most relevant text from results
                    if response["results"]:
                        # Get the highest scoring result
                        best_result = max(response["results"], key=lambda x: x["score"])
                        answer = best_result["text"]
                    else:
                        answer = (
                            "I couldn't find any relevant information in the documents."
                        )

                    # Display the answer
                    st.markdown(answer)

                    # Display all results in an expander
                    if response["results"]:
                        st.markdown("---")
                        st.markdown("**Retrieved Documents:**")
                        for i, result in enumerate(response["results"], 1):
                            with st.expander(
                                f"Document {i} (Score: {result['score']:.4f})"
                            ):
                                st.markdown(f"**File:** {result['document_name']}")
                                st.markdown(f"**Content:** {result['text']}")

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "results": response["results"],
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