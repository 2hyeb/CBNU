import json
from typing import Iterator

import chromadb
import requests

from config import (
    CHROMA_DIR, COLLECTION, OLLAMA_URL, CHAT_MODEL, EMBED_MODEL, TOP_K,
)

SYSTEM = (
    "You answer the user's question using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know based on the "
    "available documents instead of guessing. Cite sources inline as [source]."
)


def _embed(text: str) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION)
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[_embed(query)], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(docs)])[0]
    return [
        {"text": d, "source": m.get("source", "?"), "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]


def _build_messages(question: str, contexts: list[dict], history: list[dict]) -> list[dict]:
    if contexts:
        ctx_block = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in contexts)
        user_turn = f"Context:\n{ctx_block}\n\nQuestion: {question}"
    else:
        user_turn = question
    return [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": user_turn}]


def answer(question: str, history: list[dict], k: int = TOP_K) -> tuple[list[dict], Iterator[str]]:
    contexts = retrieve(question, k=k)
    messages = _build_messages(question, contexts, history)

    def token_stream() -> Iterator[str]:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": CHAT_MODEL, "messages": messages, "stream": True},
            stream=True,
            timeout=300,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break

    return contexts, token_stream()
