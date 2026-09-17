"""
HuggingFace embedding model integration.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.helpers import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper for HuggingFace embeddings."""

    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Initialized HuggingFace embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(documents)