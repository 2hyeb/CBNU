import os
from pathlib import Path

ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT / "knowledge"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "rag_docs"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
