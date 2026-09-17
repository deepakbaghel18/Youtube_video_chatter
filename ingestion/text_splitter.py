"""
Text splitting and chunking module.
"""

from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import Config
from utils.helpers import get_logger

logger = get_logger(__name__)


class TextSplitter:
    """Handle text splitting and chunking."""

    def __init__(
        self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def split_text(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for splitting")
            return []

        chunks = self.splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks (size: {self.chunk_size})")

        return chunks

    def split_documents(self, documents: list) -> list:
        """
        Split documents into chunks.

        Args:
            documents: List of document objects with page_content

        Returns:
            List of split documents
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []

        split_docs = self.splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        return split_docs


if __name__ == "__main__":
    # Test text splitter
    splitter = TextSplitter()

    test_text = """
    This is a test document with multiple sentences.
    It contains information about text splitting.
    
    The text splitter will break this into chunks.
    Each chunk will have some overlap with the previous one.
    This helps maintain context when retrieving information.
    """

    chunks = splitter.split_text(test_text)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
