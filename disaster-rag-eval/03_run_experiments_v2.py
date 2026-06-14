#!/usr/bin/env python
"""
03_run_experiments_v2.py
4-way 비교 추론: no_rag / rag / ft_only / ft_rag
모델: exaone / llama / qwen3  (4-bit bitsandbytes via vLLM)
FAISS: 자체 포맷 (faiss + chunks.pkl)
"""
import json, re, time, subprocess, os, socket, pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

os.chdir("/home/user/CBNU")

QA_PATH     = Path("data/qa_dataset.json")
INDEX_DIR   = Path("data/faiss_index")
P1_PATH     = Path("data/experiment_results.json")
OUT_PATH    = Path("data/experiment_results_v2.json")

VLLM_PORT   = 8000
VLLM_BIN    = "/home/user/miniconda3/envs/vllm/bin/python"
EMBED_MODEL = "BAAI/bge-m3"
TOP_K       = 3
TEMPERATURE = 0
MAX_TOKENS  = 512

MODEL_MAP = {
    "exaone": "/home/user/models/EXAONE-3.5-7.8B-Instruct",
    "llama":  "/home/user/models/Llama-3.1-8B-Instruct",
    "qwen3":  "/home/user/models/Qwen3-8B",
}
FT_MAP = {
    "exaone": "checkpoints/exaone-ft-merged",
    "llama":  "checkpoints/llama-ft-merged",
    "qwen3":  "checkpoints/qwen3-ft-merged",
}

NO_RAG_PROMPT = "당신은 한국어 재난·안전 전문가입니다.\n질문에 2~4문장으로 답하세요.\n\n질문: {question}\n답변:"
RAG_PROMPT    = "당신은 한국어 재난·안전 전문가입니다.\n아래 [참고 자료]를 바탕으로 2~4문장으로 답하세요.\n\n[참고 자료]\n{context}\n\n질문: {question}\n답변:"


# ── 유틸 ──────────────────────────────────────────────────────────
def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def wait_for_server(port, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.create_connection(("localhost", port), timeout=2)
            s.close()
            time.sleep(3)
            return True
        except Exception:
            time.sleep(5)
    return False


def start_vllm(model_path, port=VLLM_PORT):
    cmd = [
        VLLM_BIN, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--dtype", "float16",
        "--gpu-memory-utilization", "0.85",
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--enforce-eager",
    ]
    log = open(f"/home/user/CBNU/logs/vllm_{Path(model_path).name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    print(f"  vLLM PID={proc.pid}, 포트 {port} 대기...")
    if not wait_for_server(port):
        proc.kill()
        raise RuntimeError("vLLM 서버 시작 실패")
    print("  서버 준비 완료")
    return proc


def stop_vllm(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    time.sleep(5)


def save(results):
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── 임베딩 + RAG ──────────────────────────────────────────────────
class Embedder:
    def __init__(self, model_name, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, dtype=torch.float16).to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
            emb = out.last_hidden_state[:, 0, :].float()
        norms = emb.norm(dim=1, keepdim=True)
        return (emb / (norms + 1e-9)).cpu().numpy()


def load_faiss(index_dir):
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def get_context(question, embedder, index, chunks, k=TOP_K):
    q_emb = embedder.encode([question])
    faiss.normalize_L2(q_emb)
    _, I = index.search(q_emb, k)
    ctx = "\n\n".join(chunks[i]["text"] for i in I[0])
    return ctx[:200]


def infer_one(client, model_name, prompt):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


# ── 메인 ──────────────────────────────────────────────────────────
def main():
    with open(QA_PATH, encoding="utf-8") as f:
        qa_dataset = json.load(f)
    print(f"QA: {len(qa_dataset)}개")

    if P1_PATH.exists():
        with open(P1_PATH, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"P1 결과 {len(all_results)}개 로드")
    else:
        all_results = []

    done_keys = {(r["id"], r["model"], r["condition"]) for r in all_results}

    # FAISS + 임베더 로드
    print("FAISS 인덱스 로드...")
    index, chunks = load_faiss(INDEX_DIR)
    print(f"  벡터 수: {index.ntotal}")
    print("임베딩 모델 로드 (CPU)...")
    embedder = Embedder(EMBED_MODEL, device="cpu")

    for model_key in ["exaone", "llama", "qwen3"]:
        base_path = MODEL_MAP[model_key]
        ft_path   = FT_MAP[model_key]

        if not Path(base_path).exists():
            print(f"[SKIP] {model_key} 모델 없음")
            continue

        run_ft = Path(ft_path).exists()
        base_todo = [c for c in ["no_rag", "rag"]
                     if any((q["id"], model_key, c) not in done_keys for q in qa_dataset)]
        ft_todo = ["ft_only", "ft_rag"] if run_ft else []

        print(f"\n=== {model_key} | base:{base_todo} ft:{ft_todo} ===")

        # base 추론
        if base_todo:
            proc = start_vllm(base_path)
            cli  = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="EMPTY")
            serving = cli.models.list().data[0].id

            for cond in base_todo:
                print(f"  [{model_key}/{cond}]")
                for qa in qa_dataset:
                    key = (qa["id"], model_key, cond)
                    if key in done_keys:
                        continue
                    ctx    = get_context(qa["question"], embedder, index, chunks) if cond == "rag" else ""
                    prompt = (RAG_PROMPT.format(context=ctx, question=qa["question"])
                              if cond == "rag" else
                              NO_RAG_PROMPT.format(question=qa["question"]))
                    try:
                        ans = infer_one(cli, serving, prompt)
                        if model_key == "qwen3":
                            ans = strip_think(ans)
                    except Exception as e:
                        ans = f"ERROR: {e}"
                    all_results.append({
                        "id": qa["id"], "model": model_key, "condition": cond,
                        "question_type": qa.get("question_type", ""),
                        "question": qa["question"], "answer": ans,
                        "context": ctx,
                        "ground_truth": qa.get("answer", ""),
                    })
                    done_keys.add(key)
                save(all_results)
                print(f"    {cond} 완료, 누적 {len(all_results)}개")
            stop_vllm(proc)

        # FT 추론
        if ft_todo:
            proc = start_vllm(ft_path)
            cli  = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="EMPTY")
            serving = cli.models.list().data[0].id

            for cond in ft_todo:
                print(f"  [{model_key}/{cond}]")
                for qa in qa_dataset:
                    key = (qa["id"], model_key, cond)
                    if key in done_keys:
                        continue
                    ctx    = get_context(qa["question"], embedder, index, chunks) if cond == "ft_rag" else ""
                    prompt = (RAG_PROMPT.format(context=ctx, question=qa["question"])
                              if cond == "ft_rag" else
                              NO_RAG_PROMPT.format(question=qa["question"]))
                    try:
                        ans = infer_one(cli, serving, prompt)
                        if model_key == "qwen3":
                            ans = strip_think(ans)
                    except Exception as e:
                        ans = f"ERROR: {e}"
                    all_results.append({
                        "id": qa["id"], "model": model_key, "condition": cond,
                        "question_type": qa.get("question_type", ""),
                        "question": qa["question"], "answer": ans,
                        "context": ctx,
                        "ground_truth": qa.get("answer", ""),
                    })
                    done_keys.add(key)
                save(all_results)
                print(f"    {cond} 완료, 누적 {len(all_results)}개")
            stop_vllm(proc)

    save(all_results)
    print(f"\n전체 완료! 총 {len(all_results)}개 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
