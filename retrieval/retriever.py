"""
Document retriever module.
"""

from typing import Optional
from langchain_core.documents import Document

from app.config import Config
from vectorstore.faiss_store import FAISSVectorStore
from utils.helpers import get_logger

logger = get_logger(__name__)


class DocumentRetriever:
    """Retrieve documents from vector store."""

    def __init__(
        self,
        vectorstore: Optional[FAISSVectorStore] = None,
        top_k: int = Config.TOP_K,
    ):
        """
        Initialize retriever.

        Args:
            vectorstore: FAISSVectorStore instance
            top_k: Number of results to return
        """
        if vectorstore is None:
            vectorstore = FAISSVectorStore()

        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Override default top_k for this query

        Returns:
            List of relevant documents
        """
        k = top_k or self.top_k

        try:
            results = self.vectorstore.search(query, k=k)

            if not results:
                logger.warning(f"No documents retrieved for query: {query}")
                return []

            documents = [doc for doc, _ in results]
            logger.info(f"Retrieved {len(documents)} documents for query")

            return documents

        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.

        Args:
            query: Query text
            top_k: Override default top_k for this query

        Returns:
            List of (document, score) tuples
        """
        k = top_k or self.top_k

        try:
            results = self.vectorstore.search(query, k=k)

            if not results:
                logger.warning(f"No documents retrieved for query: {query}")
                return []

            logger.info(f"Retrieved {len(results)} documents with scores for query")
            return results

        except Exception as e:
            logger.error(f"Retrieval with scores failed: {str(e)}")
            return []

    def set_top_k(self, top_k: int) -> None:
        """Set default top_k for retrievals."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        logger.info(f"Set top_k to {top_k}")


if __name__ == "__main__":
    try:
        from embeddings.embedding_model import EmbeddingModel

        embedding_model = EmbeddingModel()
        vectorstore = FAISSVectorStore(embedding_model=embedding_model)

        # Create test index
        test_docs = [
            "Artificial intelligence is revolutionizing technology.",
            "Machine learning models require large datasets.",
            "Deep learning uses neural networks with multiple layers.",
        ]
        vectorstore.create_index(test_docs)

        # Test retriever
        retriever = DocumentRetriever(vectorstore=vectorstore, top_k=2)
        results = retriever.retrieve("neural networks")

        print(f"Retrieved {len(results)} documents")
        for doc in results:
            print(f"- {doc.page_content}")

    except Exception as e:
        print(f"Error: {str(e)}")
