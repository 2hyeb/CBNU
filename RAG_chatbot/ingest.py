import argparse
from pathlib import Path

import chromadb
import requests
from pypdf import PdfReader

from config import (
    KNOWLEDGE_DIR, CHROMA_DIR, COLLECTION,
    OLLAMA_URL, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
)

SUPPORTED = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out


def embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def main():
    parser = argparse.ArgumentParser(description="Ingest knowledge/ into Chroma")
    parser.add_argument("--rebuild", action="store_true", help="Drop existing collection first")
    args = parser.parse_args()

    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if args.rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION)

    files = [p for p in KNOWLEDGE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No supported docs found in {KNOWLEDGE_DIR}. Drop .txt/.md/.pdf and rerun.")
        return

    for f in files:
        text = read_file(f)
        chunks = chunk(text)
        if not chunks:
            continue
        rel = f.relative_to(KNOWLEDGE_DIR).as_posix()
        ids = [f"{rel}::{i}" for i in range(len(chunks))]
        embs = [embed(c) for c in chunks]
        metas = [{"source": rel} for _ in chunks]
        col.upsert(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)
        print(f"indexed {len(chunks):4d} chunks  <- {rel}")

    print(f"\nDone. Collection size: {col.count()}")


if __name__ == "__main__":
    main()
