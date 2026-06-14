#!/usr/bin/env python
"""
build_index.py — FAISS 인덱스 빌드
langchain 불필요, faiss + transformers + torch만 사용
저장: data/faiss_index/index.faiss + data/faiss_index/chunks.pkl
"""
import os, pickle, re
from pathlib import Path

os.chdir("/home/user/CBNU")

RAW_DIR    = Path("data/raw")
INDEX_DIR  = Path("data/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "BAAI/bge-m3"
CHUNK_SIZE  = 400
OVERLAP     = 80
BATCH_SIZE  = 16
DEVICE      = "cuda"


# ── 문서 로딩 ──────────────────────────────────────────────────────
def load_pdf(path):
    try:
        import fitz
        pages = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                t = page.get_text().strip()
                if len(t) > 30:
                    pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        print(f"  [PDF ERR] {path.name}: {e}")
        return ""

def load_docx(path):
    try:
        import docx
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"  [DOCX ERR] {path.name}: {e}")
        return ""

def load_doc(path):
    try:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "txt:Text",
                 "--outdir", tmpdir, str(path)],
                capture_output=True, timeout=30
            )
            txt = Path(tmpdir) / (path.stem + ".txt")
            if txt.exists():
                return txt.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  [DOC ERR] {path.name}: {e}")
    return ""

LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".doc": load_doc}


# ── 청킹 ──────────────────────────────────────────────────────────
def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = (current + "\n" + para).strip()
        else:
            if len(current) >= 80:
                chunks.append(current)
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + "\n" + para).strip()
    if len(current) >= 80:
        chunks.append(current)
    return chunks


# ── 임베딩 ────────────────────────────────────────────────────────
def build_embeddings(texts, model_name=EMBED_MODEL, device=DEVICE, batch_size=BATCH_SIZE):
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"임베딩 모델 로드: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to(device)
    model.eval()
    print(f"  로드 완료, 총 {len(texts)}청크 인코딩 시작")

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state[:, 0, :].float()  # (B, D) float32 tensor
        norms = emb.norm(dim=1, keepdim=True)
        emb = emb / (norms + 1e-9)  # L2-normalized float32 tensor
        all_embs.append(emb.cpu())
        if i % (batch_size * 10) == 0:
            print(f"  {i+len(batch)}/{len(texts)} 완료")

    # Stack in PyTorch then convert once to numpy — avoids broken numpy internals
    return torch.cat(all_embs, dim=0).numpy()


# ── 메인 ──────────────────────────────────────────────────────────
def main():
    import faiss, numpy as np

    # 1. 문서 수집 + 청킹
    files = [f for f in RAW_DIR.rglob("*")
             if f.is_file() and f.suffix.lower() in LOADERS]
    print(f"처리 대상: {len(files)}개")

    chunks = []    # list of {"text": str, "source_file": str, "disaster_type": str}
    for fpath in sorted(files):
        loader = LOADERS[fpath.suffix.lower()]
        text = loader(fpath)
        if len(text.strip()) < 50:
            print(f"  [SKIP] {fpath.name}")
            continue
        cat = fpath.parent.name
        for ch in chunk_text(text):
            chunks.append({"text": ch, "source_file": fpath.name, "disaster_type": cat})
        print(f"  [OK] {fpath.name} -> {len(chunk_text(text))}청크")

    print(f"\n총 청크: {len(chunks)}개")

    # 2. 임베딩
    texts = [c["text"] for c in chunks]
    embeddings = build_embeddings(texts)
    print(f"임베딩 shape: {embeddings.shape}")

    # 3. FAISS 인덱스 (embeddings already L2-normalized by build_embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS 벡터 수: {index.ntotal}")

    # 4. 저장 (자체 포맷 — inference 스크립트와 호환)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n저장 완료!")
    print(f"  {INDEX_DIR}/index.faiss: {(INDEX_DIR/'index.faiss').stat().st_size//1024}KB")
    print(f"  {INDEX_DIR}/chunks.pkl:  {(INDEX_DIR/'chunks.pkl').stat().st_size//1024}KB")

    # 5. 검색 테스트
    from transformers import AutoTokenizer, AutoModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model2 = AutoModel.from_pretrained(EMBED_MODEL, dtype=torch.float16).to(DEVICE)
    model2.eval()

    q_text = "화재 발생 시 대피 요령은?"
    enc = tokenizer([q_text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model2(**enc)
        q_emb = out.last_hidden_state[:, 0, :].float()
    norms = q_emb.norm(dim=1, keepdim=True)
    q_emb = (q_emb / (norms + 1e-9)).cpu().numpy()

    D, I = index.search(q_emb, k=3)
    print("\n검색 테스트:")
    for idx in I[0]:
        c = chunks[idx]
        print(f"  [{c['disaster_type']}] {c['text'][:80]}...")


if __name__ == "__main__":
    main()
