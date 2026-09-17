"""
Streamlit web interface for YouTube Video Chatter RAG application.
"""

import streamlit as st
from streamlit_option_menu import option_menu
import sys  
import os

# Add parent directory to path (project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config, validate_config
from ingestion.youtube_loader import YouTubeLoader
from ingestion.text_splitter import TextSplitter
from embeddings.embedding_model import EmbeddingModel
from vectorstore.faiss_store import FAISSVectorStore
from retrieval.retriever import DocumentRetriever
from chains.rag_chain import RAGChain
from utils.helpers import setup_logging, get_logger, extract_youtube_video_id, MemoryBuffer, format_documents

# Setup
setup_logging()
logger = get_logger(__name__)
validate_config()

# Page configuration
st.set_page_config(
    page_title=Config.STREAMLIT_PAGE_TITLE,
    page_icon=Config.STREAMLIT_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #FF0000;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #808080;
        border-left: 4px solid #FF0000;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #808080;
        border-left: 4px solid #28a745;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #808080;
        border-left: 4px solid #dc3545;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.memory = MemoryBuffer(max_size=Config.MAX_HISTORY)
    st.session_state.current_video_id = None
    st.session_state.video_title = None


@st.cache_resource
def init_embedding_model():
    """Initialize embedding model (cached)."""
    return EmbeddingModel()


@st.cache_resource
def init_vectorstore(embedding_model):
    """Initialize vector store (cached)."""
    return FAISSVectorStore(embedding_model=embedding_model)

def load_video_transcript(youtube_url: str) -> tuple[bool, str]:
    """
    Load and process YouTube video transcript using Whisper (audio-based).
    """

    with st.spinner("🔄 Downloading & transcribing audio (Whisper)..."):
        try:
            video_id = extract_youtube_video_id(youtube_url)
            if not video_id:
                return False, "❌ Invalid YouTube URL. Please provide a valid URL."

            # ✅ FIXED: No retry_attempts + pass full URL
            loader = YouTubeLoader()
            transcript = loader.fetch_and_clean(youtube_url)

            if not transcript:
                return False, "❌ Failed to extract transcript using Whisper."

            # Split text
            with st.spinner("✂️ Splitting text into chunks..."):
                splitter = TextSplitter()
                chunks = splitter.split_text(transcript)

                if not chunks:
                    return False, "❌ Failed to split transcript."

            # Create vectorstore
            with st.spinner("🔐 Creating embeddings..."):
                embedding_model = init_embedding_model()
                vectorstore = FAISSVectorStore(embedding_model=embedding_model)
                vectorstore.create_index(chunks)
                vectorstore.save_index()

                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = DocumentRetriever(
                    vectorstore=vectorstore, top_k=Config.TOP_K
                )
                st.session_state.chain = RAGChain(
                    retriever=st.session_state.retriever
                )
                st.session_state.current_video_id = video_id
                st.session_state.memory.clear()

            return True, f"✅ Video processed successfully! ({len(chunks)} chunks)"

        except Exception as e:
            logger.error(f"Error loading transcript: {str(e)}")
            return False, f"❌ Error: {str(e)}"


def ask_question(question: str) -> tuple[str, list]:
    """
    Ask a question about the loaded video.

    Returns:
        (answer: str, documents: list)
    """
    if not st.session_state.chain:
        return "Please load a YouTube video first.", []

    try:
        # Add to memory
        st.session_state.memory.add_message("user", question)

        # Get answer
        result = st.session_state.chain.invoke_with_documents(question)
        answer = result["answer"]
        documents = result["documents"]

        # Add to memory
        st.session_state.memory.add_message("assistant", answer)

        return answer, documents

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"Error generating answer: {str(e)}", []


# Main UI
st.markdown('<h1 class="main-header">🎥 YouTube Video Chatter</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666;">RAG-powered Q&A Assistant for YouTube Videos</p>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Clear Index", use_container_width=True):
            if st.session_state.vectorstore:
                st.session_state.vectorstore.delete_index()
            st.session_state.chain = None
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.memory.clear()
            st.success("✅ Index cleared!")

    with col2:
        if st.button("💾 Save Index", use_container_width=True):
            if st.session_state.vectorstore:
                st.session_state.vectorstore.save_index()
                st.success("✅ Index saved!")
            else:
                st.warning("No index to save.")

    st.divider()

    st.subheader("Model Settings")
    top_k = st.slider(
        "Top-K Retrieved Documents",
        min_value=1,
        max_value=10,
        value=Config.TOP_K,
        help="Number of context documents to retrieve",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses (0=deterministic, 1=creative)",
    )

    if st.session_state.chain:
        st.session_state.chain.set_top_k(top_k)
        st.session_state.chain.set_temperature(temperature)

    st.divider()

    st.subheader("📊 Status")
    if st.session_state.chain:
        st.success(f"✅ Video Loaded: {st.session_state.current_video_id}")
        st.info(f"📚 Documents Loaded: {len(st.session_state.memory.messages)}")
    else:
        st.warning("⚠️ No video loaded")

# Main content
tab1, tab2, tab3 = st.tabs(["📹 Load Video", "❓ Ask Questions", "📖 API Info"])

with tab1:
    st.header("Load YouTube Video")
    col1, col2 = st.columns([4, 1])

    with col1:
        # Check if a test video was clicked
        default_url = st.session_state.get('temp_url', '')
        youtube_url = st.text_input(
            "YouTube URL or Video ID",
            value=default_url,
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            help="Enter a full YouTube URL or just the video ID",
        )
        
        # Clear the temp URL after using it
        if default_url and st.session_state.get('temp_url'):
            st.session_state.temp_url = ''

    with col2:
        load_button = st.button("Load", use_container_width=True, key="load_vid")

    if load_button and youtube_url:
        success, message = load_video_transcript(youtube_url)
        if success:
            st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box"><b>💡 How to Find Videos With Captions:</b><br/>'
        '• Check if the CC (Closed Caption) button appears in YouTube player<br/>'
        '• Try educational videos, TED Talks, or news channels<br/>'
        '• Test videos: dY6xG1s_ds0, ygPjdCCvvG8, 9bZkp7q19f0<br/>'
        '• You can use full URLs or just the video ID</div>',
        unsafe_allow_html=True,
    )
    
    with st.expander("🧪 Try These Test Videos", expanded=False):
        test_videos = {
            "Python Tutorial": "https://www.youtube.com/watch?v=dY6xG1s_ds0",
            "Data Science Talk": "https://www.youtube.com/watch?v=ygPjdCCvvG8",
            "AI Explained": "https://www.youtube.com/watch?v=9bZkp7q19f0",
            "Tech News": "https://www.youtube.com/watch?v=Jj0wOwgpXEI",
        }
        
        cols = st.columns(2)
        for i, (title, url) in enumerate(test_videos.items()):
            with cols[i % 2]:
                if st.button(f"📺 {title}", use_container_width=True, key=f"test_{i}"):
                    st.session_state.temp_url = url
                    st.rerun()

with tab2:
    st.header("Ask Questions")

    if not st.session_state.chain:
        st.warning("⚠️ Please load a YouTube video first in the 'Load Video' tab.")
    else:
        st.success(f"✅ Ready to answer questions about video: {st.session_state.current_video_id}")

        question = st.text_area(
            "Your Question",
            placeholder="Ask anything about the video...",
            height=100,
            key="question",
        )

        col1, col2 = st.columns([4, 1])
        with col1:
            ask_button = st.button("Ask", use_container_width=True, type="primary")
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.memory.clear()
                st.rerun()

        if ask_button and question:
            with st.spinner("🤔 Thinking..."):
                answer, documents = ask_question(question)

            st.markdown("### Answer")
            st.write(answer)

            if documents:
                with st.expander(f"📚 Source Documents ({len(documents)})"):
                    for i, doc in enumerate(documents, 1):
                        st.markdown(f"**Document {i}:**")
                        st.text(doc.page_content)
                        st.divider()
            else:
                st.info("No source documents retrieved.")

        # Chat history
        if st.session_state.memory.messages:
            st.divider()
            st.markdown("### 💬 Chat History")
            with st.expander("View history", expanded=False):
                for msg in st.session_state.memory.messages[-6:]:
                    role = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
                    st.markdown(f"**{role}:**")
                    st.text(msg["content"])
                    st.divider()

with tab3:
    st.header("API & System Info")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")
        st.json({
            "llm_model": Config.LLM_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL,
            "ollama_url": Config.OLLAMA_BASE_URL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "top_k": Config.TOP_K,
        })

    with col2:
        st.subheader("Quick Start")
        st.markdown("""
        **1. Prerequisites:**
        - Ollama running locally (`ollama serve`)
        - Python 3.8+
        
        **2. Install Models:**
        ```
        ollama pull llama3
        ollama pull nomic-embed-text
        ```
        
        **3. Run Application:**
        ```
        streamlit run ui/streamlit_app.py
        ```
        """)

    st.divider()

    st.subheader("Advanced")
    if st.checkbox("Show Debug Info"):
        st.json({
            "session_state_keys": list(st.session_state.keys()),
            "chain_initialized": st.session_state.chain is not None,
            "vectorstore_initialized": st.session_state.vectorstore is not None,
            "memory_size": len(st.session_state.memory.messages),
        })
