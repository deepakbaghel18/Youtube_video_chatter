"""
Configuration settings for the YouTube Video Chatter RAG application.
"""

import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # Text Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    RETRIEVAL_THRESHOLD: float = float(os.getenv("RETRIEVAL_THRESHOLD", "0.5"))

    # Vector Store
    VECTORSTORE_PATH: str = os.getenv(
        "VECTORSTORE_PATH", "./data/vectorstore_store"
    )
    INDEX_NAME: str = "faiss_index"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")

    # UI Configuration
    STREAMLIT_PAGE_TITLE: str = "YouTube Video Chatter - RAG QA Assistant"
    STREAMLIT_PAGE_ICON: str = "🎥"

    # Advanced Features
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "True").lower() == "true"
    ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "False").lower() == "true"
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "5"))


# Validate configuration
def validate_config() -> None:
    """Validate configuration settings."""
    if Config.CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be positive")
    if Config.TOP_K <= 0:
        raise ValueError("TOP_K must be positive")
    if not (0 <= Config.RETRIEVAL_THRESHOLD <= 1):
        raise ValueError("RETRIEVAL_THRESHOLD must be between 0 and 1")


if __name__ == "__main__":
    validate_config()
    print("Configuration loaded successfully!")
    print(f"Ollama URL: {Config.OLLAMA_BASE_URL}")
    print(f"LLM Model: {Config.LLM_MODEL}")
    print(f"Embedding Model: {Config.EMBEDDING_MODEL}")
