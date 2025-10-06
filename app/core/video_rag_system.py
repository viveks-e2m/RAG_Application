import os
import json
import subprocess
import tempfile
import logging
from typing import List, Dict, Optional
import whisper
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VideoRAGSystemQdrant:
    def __init__(
        self,
        model_size="medium",
        chunk_size=30,
        chunk_overlap=10,
        qdrant_host="qdrant",
        qdrant_port=6333,
    ):
        """Initialize the Video RAG System with Qdrant

        Args:
            model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large)
            chunk_size (int): Size of chunks in seconds for transcript segmentation
            chunk_overlap (int): Overlap between chunks in seconds
            qdrant_host (str): Host for Qdrant service
            qdrant_port (int): Port for Qdrant service
        """
        logger.info(f"Loading Whisper {model_size} model...")
        # Load the whisper model
        self.whisper_model = whisper.load_model(model_size)

        # Use existing embedding service
        self.embedding_service = EmbeddingService()
        self.embedding_service.initialize_model()

        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Collection name for storing video segments
        self.collection_name = "video_segments"

        # Chunking parameters
        self.chunk_size = chunk_size  # in seconds
        self.chunk_overlap = chunk_overlap  # in seconds

        self.transcript_segments = []
        self.video_path = None

    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file using ffmpeg

        Args:
            video_path (str): Path to the video file

        Returns:
            str: Path to extracted audio file
        """
        audio_path = tempfile.mktemp(suffix=".wav")
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
            "-y",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to extract audio: {e}")
        except FileNotFoundError:
            raise Exception(
                "ffmpeg not found. Please install ffmpeg to extract audio from videos."
            )

    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """
        Transcribe audio with timestamps using whisper

        Args:
            audio_path (str): Path to audio file

        Returns:
            List[Dict]: List of transcript segments with timestamps
        """
        # Transcribe audio
        result = whisper.transcribe(self.whisper_model, audio_path, language="en")

        # Extract segments with timestamps
        segments = []
        for segment in result.get("segments", []):
            segments.append(
                {
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                    "words": segment.get("words", []),
                }
            )

        return segments

    def chunk_transcript(self, segments: List[Dict]) -> List[Dict]:
        """
        Chunk transcript segments into larger overlapping chunks for better context

        Args:
            segments (List[Dict]): Original transcript segments from Whisper

        Returns:
            List[Dict]: Chunked segments with configurable size and overlap
        """
        if not segments:
            return []

        # Simple approach: group segments by time windows
        chunked_segments = []
        current_chunk = {
            "text": "",
            "start": segments[0]["start"],
            "end": segments[0]["start"]
        }
        
        for segment in segments:
            segment_start = segment["start"]
            segment_end = segment["end"]
            segment_text = segment["text"].strip()
            
            # Skip empty segments
            if not segment_text:
                continue
                
            # If current chunk would exceed the desired size, save it and start a new one
            if (segment_end - current_chunk["start"]) > self.chunk_size and current_chunk["text"]:
                # Save current chunk
                chunked_segments.append({
                    "text": current_chunk["text"].strip(),
                    "start": current_chunk["start"],
                    "end": current_chunk["end"]
                })
                
                # Start new chunk with overlap
                # Find segments that fall within the overlap window
                overlap_start = max(segment_start - self.chunk_overlap, current_chunk["start"])
                new_chunk_text = ""
                new_chunk_start = segment_start
                
                # Collect text from segments in the overlap window
                for prev_segment in reversed(segments):
                    if prev_segment["start"] >= overlap_start and prev_segment["start"] < segment_start:
                        if new_chunk_text:
                            new_chunk_text = prev_segment["text"].strip() + " " + new_chunk_text
                        else:
                            new_chunk_text = prev_segment["text"].strip()
                        new_chunk_start = min(new_chunk_start, prev_segment["start"])
                    elif prev_segment["start"] < overlap_start:
                        break
                
                # Add current segment to the new chunk
                if new_chunk_text:
                    current_chunk = {
                        "text": new_chunk_text + " " + segment_text,
                        "start": new_chunk_start,
                        "end": segment_end
                    }
                else:
                    current_chunk = {
                        "text": segment_text,
                        "start": segment_start,
                        "end": segment_end
                    }
            else:
                # Add segment to current chunk
                if current_chunk["text"]:
                    current_chunk["text"] += " " + segment_text
                else:
                    current_chunk["text"] = segment_text
                    current_chunk["start"] = segment_start
                current_chunk["end"] = segment_end

        # Don't forget the last chunk
        if current_chunk["text"].strip():
            chunked_segments.append({
                "text": current_chunk["text"].strip(),
                "start": current_chunk["start"],
                "end": current_chunk["end"]
            })

        # If chunking didn't work well, fall back to simple grouping
        if len(chunked_segments) < 2:
            chunked_segments = self._simple_chunk_grouping(segments)
            
        return chunked_segments

    def _simple_chunk_grouping(self, segments: List[Dict]) -> List[Dict]:
        """
        Simple grouping approach when complex chunking fails
        
        Args:
            segments (List[Dict]): Original transcript segments
            
        Returns:
            List[Dict]: Grouped segments
        """
        if not segments:
            return []
            
        chunked_segments = []
        group_size = max(3, len(segments) // 10)  # Group into roughly 10 chunks
        
        for i in range(0, len(segments), group_size):
            group = segments[i:i + group_size]
            if group:
                combined_text = " ".join([seg["text"].strip() for seg in group if seg["text"].strip()])
                if combined_text:
                    chunked_segments.append({
                        "text": combined_text,
                        "start": group[0]["start"],
                        "end": group[-1]["end"]
                    })
        
        return chunked_segments if chunked_segments else segments

    def save_transcript_to_file(self, segments: List[Dict], output_file: str):
        """
        Save transcript segments to a text file with timestamps

        Args:
            segments (List[Dict]): Transcript segments to save
            output_file (str): Path to output text file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Video RAG System - Transcript Output\n")
            f.write(
                f"Chunk Size: {self.chunk_size} seconds, Overlap: {self.chunk_overlap} seconds\n"
            )
            f.write("=" * 50 + "\n\n")

            for i, segment in enumerate(segments, 1):
                start_formatted = self.format_time(segment["start"])
                end_formatted = self.format_time(segment["end"])
                f.write(f"[{start_formatted} -> {end_formatted}] Chunk {i}:\n")
                f.write(f"{segment['text']}\n")
                f.write("-" * 30 + "\n")

        logger.info(f"Transcript saved to: {output_file}")

    def create_collection(self):
        """Create Qdrant collection for storing embeddings"""
        # Delete collection if it exists
        try:
            self.qdrant_client.delete_collection(self.collection_name)
        except:
            pass  # Collection doesn't exist, which is fine

        # Create new collection
        embedding_dimension = 384  # Default for all-MiniLM-L6-v2

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=int(embedding_dimension), distance=Distance.COSINE
            ),
        )

    def store_embeddings(self, segments: List[Dict]):
        """
        Create and store embeddings in Qdrant using the existing embedding service

        Args:
            segments (List[Dict]): Transcript segments
        """
        logger.info(f"Storing {len(segments)} segments in Qdrant")
        # Create collection
        self.create_collection()

        # Create embeddings and store in Qdrant
        points = []
        for i, segment in enumerate(segments):
            # Create embedding using existing embedding service
            embedding = self.embedding_service.encode_text([segment["text"]])[0]

            # Create point
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": segment["text"],
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "segment_id": i,
                },
            )
            points.append(point)

        # Upload points to Qdrant
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Stored {len(points)} segments in Qdrant")

    def process_video(self, video_path: str, save_transcript: bool = True):
        """
        Process video: extract audio, transcribe, and create embeddings

        Args:
            video_path (str): Path to video file
            save_transcript (bool): Whether to save transcript to file
        """
        logger.info(f"Processing video: {video_path}")
        self.video_path = video_path

        # Validate video path
        if not os.path.exists(video_path):
            raise Exception(f"Video file '{video_path}' not found.")

        # Extract audio
        logger.info("Extracting audio...")
        try:
            audio_path = self.extract_audio(video_path)
        except Exception as e:
            raise Exception(f"Audio extraction failed: {e}")

        # Transcribe with timestamps
        logger.info("Transcribing audio with timestamps...")
        try:
            self.transcript_segments = self.transcribe_with_timestamps(audio_path)
            logger.info(f"Transcribed {len(self.transcript_segments)} segments")
        except Exception as e:
            # Clean up temporary file first
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise Exception(f"Transcription failed: {e}")

        # Apply chunking to transcript segments
        logger.info(
            f"Chunking transcript with size={self.chunk_size}s and overlap={self.chunk_overlap}s..."
        )
        chunked_segments = self.chunk_transcript(self.transcript_segments)
        self.transcript_segments = chunked_segments
        logger.info(f"Created {len(self.transcript_segments)} chunks after processing")

        # Save transcript to file if requested
        if save_transcript and self.video_path:
            # Generate output filename based on video filename
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_file = f"{video_name}_transcript_chunks.txt"
            self.save_transcript_to_file(self.transcript_segments, output_file)

        # Store embeddings in Qdrant
        logger.info("Creating and storing embeddings in Qdrant...")
        if self.transcript_segments:
            self.store_embeddings(self.transcript_segments)
        else:
            logger.warning("No transcript segments found.")

        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        logger.info(f"Processed {len(self.transcript_segments)} segments")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant segments based on query using Qdrant

        Args:
            query (str): Search query
            top_k (int): Number of top results to return

        Returns:
            List[Dict]: Top matching segments with timestamps
        """
        if not self.transcript_segments:
            raise Exception("No video processed yet. Call process_video() first.")

        # Create embedding for query using existing embedding service
        query_embedding = self.embedding_service.encode_text([query])[0]

        # Search in Qdrant
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )

        # Prepare results
        results = []
        for result in search_result:
            payload = result.payload
            # Handle case where payload might be None
            if payload is not None:
                results.append(
                    {
                        "text": payload.get("text", ""),
                        "start_time": payload.get("start_time", 0),
                        "end_time": payload.get("end_time", 0),
                        "similarity": result.score,
                    }
                )
            else:
                results.append(
                    {
                        "text": "",
                        "start_time": 0,
                        "end_time": 0,
                        "similarity": result.score,
                    }
                )

        return results

    def format_time(self, seconds: float) -> str:
        """
        Format seconds to HH:MM:SS.mmm format

        Args:
            seconds (float): Time in seconds

        Returns:
            str: Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def display_results(self, results: List[Dict]):
        """
        Display search results with formatted timestamps

        Args:
            results (List[Dict]): Search results
        """
        if not results:
            logger.info("No results found.")
            return

        logger.info("\n--- Search Results ---")
        for i, result in enumerate(results, 1):
            start_formatted = self.format_time(result["start_time"])
            end_formatted = self.format_time(result["end_time"])
            logger.info(f"\n{i}. Similarity: {result['similarity']:.4f}")
            logger.info(f"   Time: {start_formatted} - {end_formatted}")
            logger.info(f"   Text: {result['text']}")