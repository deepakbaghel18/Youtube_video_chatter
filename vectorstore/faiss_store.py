"""
FAISS vector store management.
"""

import os
from typing import Optional
from langchain_community.vectorstores import FAISS

from app.config import Config
from embeddings.embedding_model import EmbeddingModel
from utils.helpers import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """Manage FAISS vector store for document embeddings."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        store_path: str = Config.VECTORSTORE_PATH,
        index_name: str = Config.INDEX_NAME,
    ):
        """
        Initialize FAISS vector store.

        Args:
            embedding_model: EmbeddingModel instance
            store_path: Path to store FAISS index
            index_name: Name of the index
        """
        if embedding_model is None:
            embedding_model = EmbeddingModel()

        self.embedding_model = embedding_model
        self.store_path = store_path
        self.index_name = index_name
        self.full_path = os.path.join(store_path, index_name)

        # Create store directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)

        self.vectorstore: Optional[FAISS] = None

        # Try to load existing index
        self._load_existing_index()

    def _load_existing_index(self) -> None:
        """Load existing FAISS index if available."""
        if os.path.exists(self.full_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.full_path,
                    self.embedding_model.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"Loaded existing FAISS index from {self.full_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {str(e)}")
                self.vectorstore = None
        else:
            logger.info(f"No existing index found at {self.full_path}")

    def create_index(self, documents: list[str], metadata: Optional[list] = None) -> None:
        """
        Create new FAISS index from documents.

        Args:
            documents: List of document texts
            metadata: Optional metadata for documents
        """
        if not documents:
            logger.error("Cannot create index with empty documents")
            return

        try:
            # Embed documents
            embeddings = self.embedding_model.embed_documents(documents)

            # Create FAISS index
            self.vectorstore = FAISS.from_texts(
                texts=documents,
                embedding=self.embedding_model.embeddings,
                metadatas=metadata,
            )

            logger.info(f"Created FAISS index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise

    def add_documents(self, documents: list[str], metadata: Optional[list] = None) -> None:
        """
        Add documents to existing index.

        Args:
            documents: List of document texts
            metadata: Optional metadata for documents
        """
        if not self.vectorstore:
            logger.warning("No existing index. Creating new one.")
            self.create_index(documents, metadata)
            return

        try:
            self.vectorstore.add_texts(texts=documents, metadatas=metadata)
            logger.info(f"Added {len(documents)} documents to index")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def search(self, query: str, k: int = Config.TOP_K) -> list[tuple]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def save_index(self) -> bool:
        """
        Save FAISS index to disk.

        Returns:
            True if successful
        """
        if not self.vectorstore:
            logger.error("No vector store to save")
            return False

        try:
            self.vectorstore.save_local(self.full_path)
            logger.info(f"Saved FAISS index to {self.full_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            return False

    def delete_index(self) -> bool:
        """
        Delete FAISS index from disk.

        Returns:
            True if successful
        """
        try:
            import shutil

            if os.path.exists(self.full_path):
                shutil.rmtree(self.full_path)
                self.vectorstore = None
                logger.info(f"Deleted FAISS index from {self.full_path}")
                return True
            else:
                logger.warning(f"Index not found at {self.full_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete index: {str(e)}")
            return False

    def get_vectorstore(self) -> Optional[FAISS]:
        """Get the underlying FAISS vectorstore."""
        return self.vectorstore

    def is_empty(self) -> bool:
        """Check if vector store is empty."""
        return self.vectorstore is None


if __name__ == "__main__":
    # Test FAISS store
    try:
        embedding_model = EmbeddingModel()
        store = FAISSVectorStore(embedding_model=embedding_model)

        # Create index
        test_docs = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a popular programming language",
            "Machine learning is transforming industries",
        ]

        store.create_index(test_docs)

        # Search
        results = store.search("programming", k=2)
        print(f"Search results: {len(results)}")
        for doc, score in results:
            print(f"Score: {score:.4f} - {doc.page_content[:50]}")

        # Save
        store.save_index()

    except Exception as e:
        print(f"Error: {str(e)}")
