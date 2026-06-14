# -*- coding: utf-8 -*-
"""
05_ragas_qtype_eval.py
질문 유형(q_type)별 RAGAS 평가 스크립트
- qa_dataset.json의 type 정보를 experiment_results_v2.json에 매핑
- (model, condition)별 per-question RAGAS 점수 저장
- q_type별 집계 통계 출력
대상 조건: rag 3개 + ft_only 3개 (exaone/qwen3/llama)
"""

import json
import os
import subprocess
import socket
import time
import pandas as pd
import collections
from pathlib import Path
from openai import OpenAI as OAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE        = Path("/home/user/CBNU")
QA_PATH     = BASE / "data/qa_dataset.json"
EXP_PATH    = BASE / "data/experiment_results_v2.json"
RAW_OUT     = BASE / "data/eval_qtype_raw.json"
SUM_OUT     = BASE / "data/eval_summary_qtype.csv"
LOG_DIR     = BASE / "logs"

JUDGE_MODEL = "/home/user/models/Qwen3-32B-AWQ"
JUDGE_PORT  = 8001
VLLM_BIN    = "/home/user/miniconda3/envs/vllm/bin/python"

TARGET_CONDITIONS = [
    ("exaone", "rag"),
    ("qwen3",  "rag"),
    ("llama31", "rag"),
    ("exaone", "ft_only"),
    ("qwen3",  "ft_only"),
    ("llama31", "ft_only"),
]

MAX_CTX_CHARS = 4000
MAX_ANS_CHARS = 2500


def wait_for_server(port, timeout=300):
    for _ in range(timeout):
        try:
            s = socket.create_connection(("localhost", port), timeout=1)
            s.close()
            return True
        except Exception:
            time.sleep(1)
    return False


def start_judge_vllm():
    try:
        s = socket.create_connection(("localhost", JUDGE_PORT), timeout=2)
        s.close()
        print(f"  Judge already on port {JUDGE_PORT} — reusing")
        return None
    except Exception:
        pass
    cmd = [
        VLLM_BIN, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(JUDGE_MODEL),
        "--port", str(JUDGE_PORT),
        "--dtype", "half",
        "--quantization", "awq",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
        "--max-model-len", "8192",
        "--enforce-eager",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1,2"
    log = open(LOG_DIR / "vllm_judge_qtype.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
    print(f"  Judge PID={proc.pid}, waiting...")
    if not wait_for_server(JUDGE_PORT, timeout=300):
        proc.kill()
        raise RuntimeError("Judge vLLM failed to start")
    print("  Judge ready")
    return proc


def stop_judge_vllm(proc):
    if proc is None:
        print("  Judge was pre-started — leaving it running")
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    time.sleep(3)


def build_qtype_map():
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    return {item["question"].strip(): item["type"] for item in qa}


def load_experiments(qtype_map):
    data = json.loads(EXP_PATH.read_text(encoding="utf-8"))
    for item in data:
        q = item.get("question", "").strip()
        item["q_type"] = qtype_map.get(q, "")
    return data


def eval_condition(model_name, condition, items, ragas_llm, ragas_emb, run_config):
    has_context = condition in ("rag", "ft_rag")

    rows = []
    for it in items:
        ctx = it.get("context", "") or ""
        ans = it.get("pred_answer", "") or ""
        if str(ans).startswith("ERROR"):
            continue
        rows.append({
            "question":     it["question"],
            "answer":       ans[:MAX_ANS_CHARS],
            "contexts":     [ctx[:MAX_CTX_CHARS]] if has_context else [""],
            "ground_truth": it.get("gold_answer", ""),
            "q_type":       it.get("q_type", ""),
        })

    if not rows:
        print(f"  [{model_name}/{condition}] 유효 항목 없음")
        return []

    ds = Dataset.from_list(rows)
    metrics = [faithfulness, answer_relevancy, context_recall] if has_context else [answer_relevancy]

    result = evaluate(
        dataset=ds,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config,
        raise_exceptions=False,
    )

    scores_df = result.to_pandas()

    per_q = []
    for i, row in enumerate(rows):
        entry = {
            "model":     model_name,
            "condition": condition,
            "q_type":    row["q_type"],
            "question":  row["question"],
        }
        for col in ["faithfulness", "answer_relevancy", "context_recall"]:
            if col in scores_df.columns:
                val = scores_df.iloc[i][col]
                entry[col] = float(val) if pd.notna(val) else None
            else:
                entry[col] = None
        per_q.append(entry)

    return per_q


def print_qtype_summary(results):
    metrics = ["faithfulness", "answer_relevancy", "context_recall"]
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        key = (r["model"], r["condition"], r.get("q_type", ""))
        for m in metrics:
            v = r.get(m)
            if v is not None:
                acc[key][m].append(v)

    rows = []
    for (model, cond, qtype), vals in sorted(acc.items()):
        row = {"model": model, "condition": cond, "q_type": qtype}
        for m in metrics:
            vs = vals.get(m, [])
            row[m] = round(sum(vs)/len(vs), 4) if vs else None
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def save_qtype_csv(results):
    metrics = ["faithfulness", "answer_relevancy", "context_recall"]
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        key = (r["model"], r["condition"], r.get("q_type", ""))
        for m in metrics:
            v = r.get(m)
            if v is not None:
                acc[key][m].append(v)

    rows = []
    for (model, cond, qtype), vals in sorted(acc.items()):
        row = {"model": model, "condition": cond, "q_type": qtype}
        for m in metrics:
            vs = vals.get(m, [])
            row[m] = round(sum(vs)/len(vs), 4) if vs else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(SUM_OUT, index=False, encoding="utf-8-sig")


def main():
    LOG_DIR.mkdir(exist_ok=True)

    print("=== q_type 매핑 로드 ===")
    qtype_map = build_qtype_map()
    print(f"  {len(qtype_map)}개 질문 매핑")

    print("=== 실험 결과 로드 ===")
    data = load_experiments(qtype_map)
    print(f"  총 {len(data)}개 항목")

    if RAW_OUT.exists():
        existing = json.loads(RAW_OUT.read_text(encoding="utf-8"))
        done_keys = {(r["model"], r["condition"]) for r in existing}
        print(f"  기존 결과: {len(existing)}개 ({len(done_keys)}개 조건 완료)")
    else:
        existing = []
        done_keys = set()

    pending = [(m, c) for m, c in TARGET_CONDITIONS if (m, c) not in done_keys]
    if not pending:
        print("모든 조건 완료됨")
        print_qtype_summary(existing)
        save_qtype_csv(existing)
        return

    print(f"  평가 대기: {pending}")

    print("\n=== Judge 시작 ===")
    judge_proc = start_judge_vllm()

    try:
        cli = OAI(base_url=f"http://localhost:{JUDGE_PORT}/v1", api_key="EMPTY")
        serving_model = cli.models.list().data[0].id
        print(f"  Serving model: {serving_model}")

        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=serving_model,
                base_url=f"http://localhost:{JUDGE_PORT}/v1",
                api_key="EMPTY",
                temperature=0,
                max_tokens=2048,
            )
        )

        print("  BGE-M3 임베딩 로드 (CPU)...")
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )

        run_config = RunConfig(timeout=300, max_workers=8, max_retries=3)

        for m in [faithfulness, answer_relevancy, context_recall]:
            m.llm = ragas_llm
        answer_relevancy.embeddings = ragas_emb

        all_results = list(existing)

        for model_name, condition in pending:
            print(f"\n>>> [{model_name}/{condition}] 평가 시작")
            items = [r for r in data if r["model"] == model_name and r["condition"] == condition]
            print(f"    항목 수: {len(items)}")

            per_q = eval_condition(model_name, condition, items, ragas_llm, ragas_emb, run_config)
            all_results.extend(per_q)

            RAW_OUT.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{model_name}/{condition}] 완료: {len(per_q)}개 저장")
            print_qtype_summary(all_results)

    finally:
        stop_judge_vllm(judge_proc)

    print("\n=== 최종 q_type 집계 ===")
    print_qtype_summary(all_results)
    save_qtype_csv(all_results)
    print(f"\n저장 완료: {SUM_OUT}")


if __name__ == "__main__":
    main()
