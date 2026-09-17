"""
Main entry point for YouTube Video Chatter application.
"""

import sys
import os
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config, validate_config
from ingestion.youtube_loader import YouTubeLoader
from ingestion.text_splitter import TextSplitter
from embeddings.embedding_model import EmbeddingModel
from vectorstore.faiss_store import FAISSVectorStore
from retrieval.retriever import DocumentRetriever
from chains.rag_chain import RAGChain
from utils.helpers import setup_logging, get_logger, MemoryBuffer

logger = get_logger(__name__)


class YouTubeChatterApp:
    """Main application class for YouTube Video Chatter."""

    def __init__(self):
        """Initialize the application."""
        validate_config()
        setup_logging()
        logger.info("Initializing YouTube Video Chatter Application")

        self.embedding_model = EmbeddingModel()
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self.memory = MemoryBuffer(max_size=Config.MAX_HISTORY)

    def load_video(self, youtube_url: str) -> bool:
        """
        Load YouTube video and create RAG chain.

        Args:
            youtube_url: YouTube URL or video ID

        Returns:
            True if successful
        """
        try:
            logger.info(f"Loading video from: {youtube_url}")

            # Fetch transcript
            loader = YouTubeLoader()
            transcript = loader.fetch_and_clean(youtube_url)

            if not transcript:
                logger.error("Failed to fetch transcript")
                return False

            # Split text
            splitter = TextSplitter()
            chunks = splitter.split_text(transcript)

            if not chunks:
                logger.error("Failed to split transcript")
                return False

            # Create vector store
            logger.info("Creating vector store and embeddings")
            self.vectorstore = FAISSVectorStore(embedding_model=self.embedding_model)
            self.vectorstore.create_index(chunks)
            self.vectorstore.save_index()

            # Initialize retriever and chain
            self.retriever = DocumentRetriever(vectorstore=self.vectorstore, top_k=Config.TOP_K)
            self.chain = RAGChain(retriever=self.retriever)

            self.memory.clear()

            logger.info("Video loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            return False

    def ask_question(self, question: str) -> dict:
        """
        Ask a question about the loaded video.

        Args:
            question: User question

        Returns:
            Dictionary with answer and context
        """
        if not self.chain:
            return {
                "success": False,
                "error": "No video loaded. Please load a video first.",
            }

        try:
            # Add to memory
            self.memory.add_message("user", question)

            # Get answer
            result = self.chain.invoke_with_documents(question)
            answer = result["answer"]
            documents = result["documents"]

            # Add to memory
            self.memory.add_message("assistant", answer)

            return {
                "success": True,
                "question": question,
                "answer": answer,
                "documents": documents,
                "num_documents": len(documents),
            }

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "success": False,
                "error": str(e),
            }

    def interactive_chat(self) -> None:
        """Run interactive chat mode."""
        print("\n" + "=" * 60)
        print("🎥 YouTube Video Chatter - Interactive Mode")
        print("=" * 60)

        # Load video
        print("\nEnter YouTube URL or video ID (or 'quit' to exit):")
        youtube_url = input("> ").strip()

        if youtube_url.lower() == "quit":
            print("Goodbye!")
            return

        if not youtube_url:
            print("❌ Please provide a valid URL or video ID")
            return

        print("\n⏳ Loading video...")
        if not self.load_video(youtube_url):
            print("❌ Failed to load video")
            return

        print("✅ Video loaded successfully!")

        # Interactive Q&A
        print("\nEnter your questions (type 'quit' to exit, 'clear' to clear history):\n")

        while True:
            question = input("You: ").strip()

            if question.lower() == "quit":
                print("\nGoodbye!")
                break

            if question.lower() == "clear":
                self.memory.clear()
                print("Chat history cleared.\n")
                continue

            if not question:
                continue

            print("\n🤔 Thinking...")
            result = self.ask_question(question)

            if result["success"]:
                print(f"\nAssistant: {result['answer']}\n")

                if result["num_documents"] > 0:
                    print(f"📚 Used {result['num_documents']} source documents\n")
            else:
                print(f"❌ Error: {result['error']}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YouTube Video Chatter - RAG-powered Q&A Assistant"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="YouTube URL to load",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (requires --url)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run Streamlit web interface",
    )

    args = parser.parse_args()

    if args.streamlit:
        # Run Streamlit
        import subprocess

        subprocess.run(["streamlit", "run", "ui/streamlit_app.py"])
        return

    # Initialize app
    app = YouTubeChatterApp()

    if args.interactive or not args.url:
        # Interactive mode
        app.interactive_chat()
    else:
        # CLI mode
        print("Loading video...")
        if app.load_video(args.url):
            print("✅ Video loaded!")

            if args.question:
                print(f"\nAsking: {args.question}")
                result = app.ask_question(args.question)

                if result["success"]:
                    print(f"\n{result['answer']}")
                    if result["num_documents"] > 0:
                        print(f"\nSource documents used: {result['num_documents']}")
                else:
                    print(f"Error: {result['error']}")
            else:
                app.interactive_chat()
        else:
            print("❌ Failed to load video")


if __name__ == "__main__":
    main()
