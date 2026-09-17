# 🎥 YouTube Video Chatter - Production RAG Application

A production-ready Retrieval Augmented Generation (RAG) application that answers questions about YouTube videos using local AI models (Ollama). No paid APIs, fully offline capable.

## ✨ Features

- **Local LLM Integration**: Uses Ollama (llama3 or mistral) for completely local inference
- **Local Embeddings**: `nomic-embed-text` for semantic search
- **RAG Pipeline**: Intelligent retrieval + generation workflow
- **YouTube Integration**: Automatic transcript fetching and processing
- **Vector Storage**: FAISS for efficient similarity search with persistence
- **Conversational Memory**: Chat history tracking across sessions
- **Web UI**: Beautiful Streamlit interface
- **CLI Support**: Command-line interface for automation
- **Production Ready**: Type hints, logging, error handling, modular design

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         Streamlit UI / CLI Interface             │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│              RAG Chain (LCEL)                    │
│   Retriever → Context → Prompt → LLM → Parser   │
└────────┬──────────────────────────┬─────────────┘
         │                          │
    ┌────v────┐              ┌──────v──────┐
    │ Retriever│              │ ChatOllama   │
    └────┬────┘              │ (llama3)     │
         │                   └──────────────┘
    ┌────v────────────────┐
    │  FAISS VectorStore   │
    │  (Persistent Index)  │
    └────┬────────────────┘
         │
    ┌────v────────────────┐
    │  OllamaEmbeddings    │
    │  (nomic-embed-text)  │
    └──────────────────────┘
         ▲
         │ (Embedding requests)
    ┌────┴───────────────┐
    │    Ingestion        │
    │ • YouTube Loader    │
    │ • Text Splitter     │
    │ • Chunk Processing  │
    └─────────────────────┘
```

## 📁 Project Structure

```
youtube-video-chatter/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   └── main.py                   # Main entry point
├── ingestion/
│   ├── __init__.py
│   ├── youtube_loader.py         # YouTube transcript fetching
│   └── text_splitter.py          # Text chunking
├── embeddings/
│   ├── __init__.py
│   └── embedding_model.py        # Ollama embeddings wrapper
├── vectorstore/
│   ├── __init__.py
│   └── faiss_store.py           # FAISS vector store manager
├── retrieval/
│   ├── __init__.py
│   └── retriever.py             # Document retriever
├── chains/
│   ├── __init__.py
│   └── rag_chain.py             # RAG chain implementation
├── utils/
│   ├── __init__.py
│   └── helpers.py               # Utility functions
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py         # Web UI
├── data/
│   └── vectorstore_store/       # Persistent FAISS indices
├── logs/                         # Application logs
├── .env.example                  # Environment variables template
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Ollama** (local LLM runtime)
- **4GB+ RAM** (6GB+ recommended)

### 1. Install Ollama

Download from [ollama.ai](https://ollama.ai)

### 2. Pull Required Models

```bash
# LLM model
ollama pull llama3

# Embedding model
ollama pull nomic-embed-text

# (Optional) Alternative LLM
ollama pull mistral
```

### 3. Start Ollama Service

```bash
# Start Ollama in the background
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### 4. Clone and Setup Project

```bash
cd youtube-video-chatter

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration
cp .env.example .env
```

### 5. Run the Application

**Option A: Web UI (Recommended)**
```bash
streamlit run ui/streamlit_app.py
```
Opens at `http://localhost:8501`

**Option B: Interactive CLI**
```bash
python app/main.py --interactive
```

**Option C: Single Query**
```bash
python app/main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
                   --question "What is this video about?"
```

## 💻 Usage Examples

### Web UI

1. Navigate to "Load Video" tab
2. Paste YouTube URL or video ID
3. Click "Load"
4. Go to "Ask Questions" tab
5. Type your question
6. View answer and source documents

### CLI - Interactive Mode

```bash
python app/main.py --interactive

# Output:
# 🎥 YouTube Video Chatter - Interactive Mode
# ============================================================
# 
# Enter YouTube URL or video ID (or 'quit' to exit):
# > https://www.youtube.com/watch?v=dQw4w9WgXcQ
# 
# ⏳ Loading video...
# ✅ Video loaded successfully!
# 
# Enter your questions (type 'quit' to exit, 'clear' to clear history):
# 
# You: What is the main topic?
# 🤔 Thinking...
# Assistant: [Answer from RAG Chain]
# 📚 Used 3 source documents
```

### Python API

```python
from app.main import YouTubeChatterApp

# Initialize
app = YouTubeChatterApp()

# Load video
app.load_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Ask questions
result = app.ask_question("What is this about?")
print(result['answer'])
print(f"Source documents: {result['num_documents']}")
```

## ⚙️ Configuration

Edit `.env` file to customize:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3              # llama3 or mistral
EMBEDDING_MODEL=nomic-embed-text

# Text Processing
CHUNK_SIZE=500                # Characters per chunk
CHUNK_OVERLAP=100             # Overlap between chunks

# Retrieval
TOP_K=3                       # Retrieved documents
RETRIEVAL_THRESHOLD=0.5       # Minimum similarity score

# Vector Store
VECTORSTORE_PATH=./data/vectorstore_store

# Logging
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/app.log

# Advanced
ENABLE_STREAMING=True
ENABLE_RERANKING=False
MAX_HISTORY=5                 # Conversation history size
```

## 📊 System Performance

**Typical Response Times** (on modern hardware):
- Video loading: 10-30 seconds (depends on transcript length)
- Question answering: 5-15 seconds (depends on model and complexity)
- Streaming responses: Progressive output (if enabled)

**Memory Usage**:
- Base: ~2GB (Ollama server)
- Per video: ~500MB-2GB (vector store size)
- Total: ~4-6GB recommended

## 🔧 Advanced Features

### Multi-Query Retrieval (Optional)

For better retrieval, you can implement multiple queries:

```python
# In chains/rag_chain.py
queries = [original_query, reworded_query]
for q in queries:
    docs.extend(retriever.retrieve(q))
```

### Streaming Responses

```python
# Enable in .env
ENABLE_STREAMING=True

# In chains/rag_chain.py
for chunk in chain.stream({"question": question}):
    print(chunk, end='', flush=True)
```

### Reranking (Lightweight)

```python
# Enable in .env
ENABLE_RERANKING=True

# Uses cross-encoder for better ranking
```

## 🐛 Troubleshooting

### "Connection refused" to Ollama

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### "Model not found" error

```bash
# Pull missing model
ollama pull llama3
ollama pull nomic-embed-text

# List available models
ollama list
```

### Slow responses

- **Reduce chunk_size**: Faster retrieval (less accurate)
- **Reduce top_k**: Fewer documents to process
- **Use mistral**: Faster than llama3 but less capable
- **Increase model context**: In Ollama settings

### Out of memory errors

```bash
# For FAISS index:
# - Delete index and recreate from smaller video
# - Use sparse embeddings (future enhancement)

# For Ollama:
# - Reduce model size (use mistral instead of llama3)
# - Allocate more RAM to system
```

## 📝 Code Examples

### Creating Custom Retriever

```python
from retrieval.retriever import DocumentRetriever
from vectorstore.faiss_store import FAISSVectorStore

store = FAISSVectorStore()
retriever = DocumentRetriever(vectorstore=store, top_k=5)

# Custom retrieval
docs = retriever.retrieve("your query")
```

### Extending RAG Chain

```python
from chains.rag_chain import RAGChain

chain = RAGChain()

# With custom temperature
chain.set_temperature(0.3)

# With custom top-k
chain.set_top_k(5)

# Get answer with documents
result = chain.invoke_with_documents("question")
```

### Batch Processing

```python
from app.main import YouTubeChatterApp

app = YouTubeChatterApp()

videos = [
    "https://www.youtube.com/watch?v=...",
    "https://www.youtube.com/watch?v=...",
]

for video_url in videos:
    app.load_video(video_url)
    
    questions = ["Q1?", "Q2?", "Q3?"]
    for q in questions:
        result = app.ask_question(q)
        print(result['answer'])
```

## 🧪 Testing

```bash
# Test individual modules
python ingestion/youtube_loader.py
python embeddings/embedding_model.py
python vectorstore/faiss_store.py
python chains/rag_chain.py

# Test configuration
python app/config.py
```

## 📊 Production Deployment

### Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "ui/streamlit_app.py"]
```

### Environment Variables

```bash
export OLLAMA_BASE_URL="http://ollama-service:11434"
export LLM_MODEL="llama3"
export LOG_LEVEL="INFO"
```

### Performance Optimization

1. **Use FAISS GPU variant** (optional): `pip install faiss-gpu`
2. **Enable model quantization**: In Ollama settings
3. **Implement caching**: For frequently asked questions
4. **Use connection pooling**: For multiple requests

## 📚 Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| langchain | LLM framework | 0.1.11 |
| langchain-community | Community integrations | 0.0.24 |
| faiss-cpu | Vector similarity search | 1.7.4 |
| streamlit | Web UI | 1.31.1 |
| youtube-transcript-api | Fetch transcripts | 0.6.2 |
| ollama | Ollama Python client | 0.1.1 |
| python-dotenv | Environment config | 1.0.0 |

## 🤝 Contributing

Improvements welcome! Areas for enhancement:

- [ ] GPU support for embeddings
- [ ] Multi-language support
- [ ] Document storage (PDFs, web pages)
- [ ] Advanced reranking
- [ ] API endpoints
- [ ] Database backend

## 📄 License

MIT License - feel free to use in your projects

## ❓ FAQ

**Q: Can I use different Ollama models?**
A: Yes! Change `LLM_MODEL` in `.env`. Available: llama3, mistral, neural-chat, etc.

**Q: Does it work offline?**
A: Yes, if Ollama and models are already downloaded. YouTube API requires internet for transcript fetching.

**Q: How accurate are responses?**
A: Depends on video quality, question clarity, and model capability. llama3 is more accurate than mistral but slower.

**Q: Can I use OpenAI instead?**
A: Not with current setup. Would require modifying config and using `langchain_openai`. Original prototype supports this.

**Q: How to improve response quality?**
A: Lower temperature (0.3-0.5), increase top_k (5-10), use llama3 model, ensure transcripts are clear.

## 🚀 Next Steps

1. ✅ Run `ollama serve`
2. ✅ Install requirements: `pip install -r requirements.txt`
3. ✅ Copy config: `cp .env.example .env`
4. ✅ Start UI: `streamlit run ui/streamlit_app.py`
5. ✅ Load a YouTube video
6. ✅ Ask your first question!

## 📞 Support

For issues:
1. Check logs in `logs/app.log`
2. Enable DEBUG logging: `LOG_LEVEL=DEBUG` in `.env`
3. Verify Ollama is running: `ollama list`
4. Check model availability: `ollama pull llama3`

---

**Happy learning! 🚀**
