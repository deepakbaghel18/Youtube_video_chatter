"""
Utility helper functions for YouTube Video Chatter application.
"""

import logging
import os
import re
from typing import Optional
from datetime import datetime

from app.config import Config


def setup_logging() -> None:
    """Configure logging for the application."""
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler(),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats.

    Args:
        url: YouTube URL in various formats

    Returns:
        Video ID or None if invalid
    """
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",  # Direct video ID
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing line breaks.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r"[^\w\s\.\,\!\?\-\(\)\'\"]", "", text)
    return text.strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to maximum length while preserving words.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    # Find last complete word
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + "..."


def format_documents(docs: list) -> str:
    """
    Format documents for display and context.

    Args:
        docs: List of document objects with page_content attribute

    Returns:
        Formatted string with all documents
    """
    if not docs:
        return "No documents found."

    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        formatted.append(f"**Document {i}:**\n{content}\n")

    return "\n---\n".join(formatted)


def get_timestamp() -> str:
    """Get current timestamp as ISO string."""
    return datetime.utcnow().isoformat()


def validate_url(url: str) -> bool:
    """
    Validate if URL is a valid YouTube URL.

    Args:
        url: URL to validate

    Returns:
        True if valid YouTube URL
    """
    return extract_youtube_video_id(url) is not None


class MemoryBuffer:
    """Simple buffer for maintaining conversation history."""

    def __init__(self, max_size: int = 5):
        """
        Initialize memory buffer.

        Args:
            max_size: Maximum number of messages to keep
        """
        self.max_size = max_size
        self.messages: list = []

    def add_message(self, role: str, content: str) -> None:
        """
        Add message to buffer.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.messages.append({"role": role, "content": content, "timestamp": get_timestamp()})

        # Keep only last max_size messages
        if len(self.messages) > self.max_size * 2:  # *2 for pairs
            self.messages = self.messages[-(self.max_size * 2) :]

    def get_history(self) -> str:
        """Get formatted conversation history."""
        if not self.messages:
            return "No conversation history."

        history = []
        for msg in self.messages[-6:]:  # Last 3 exchanges
            role = msg["role"].upper()
            content = truncate_text(msg["content"], 200)
            history.append(f"{role}: {content}")

        return "\n".join(history)

    def clear(self) -> None:
        """Clear message buffer."""
        self.messages = []


if __name__ == "__main__":
    setup_logging()
    logger = get_logger(__name__)

    # Test helper functions
    video_id = extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    logger.info(f"Extracted video ID: {video_id}")

    text = extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ")
    logger.info(f"Extracted from short URL: {text}")
