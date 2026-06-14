#!/usr/bin/env python
"""04_ragas_eval.py — RAGAS evaluation for 4-way experiment results
Judge: EXAONE-3.5-32B-AWQ via vLLM (port 8001)
Embed: BAAI/bge-m3 via HuggingFace (local, CPU)
"""

import json, subprocess, socket, time, warnings
from pathlib import Path

import pandas as pd
warnings.filterwarnings("ignore")

RESULTS_PATH = Path("data/experiment_results_v2.json")
SUMMARY_PATH = Path("data/eval_summary_by_type.csv")
JUDGE_MODEL  = "/home/user/models/Qwen3-32B-AWQ"
JUDGE_PORT   = 8001
VLLM_BIN     = "/home/user/miniconda3/envs/vllm/bin/python"

MODELS     = ["exaone", "llama", "qwen3"]
CONDITIONS = ["no_rag", "rag", "ft_only", "ft_rag"]

MAX_CTX_CHARS = 4000   # truncate context (budget: 8192 token limit)
MAX_ANS_CHARS = 2500   # truncate answer (budget: 8192 token limit)


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


def start_judge_vllm():
    # Reuse externally pre-started judge if port already listening
    try:
        s = socket.create_connection(("localhost", JUDGE_PORT), timeout=2)
        s.close()
        print(f"  Judge already running on port {JUDGE_PORT} ? reusing")
        return None
    except Exception:
        pass
    cmd = [
        VLLM_BIN, "-m", "vllm.entrypoints.openai.api_server",
        "--model", JUDGE_MODEL,
        "--port", str(JUDGE_PORT),
        "--dtype", "half",
        "--quantization", "awq",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
        "--max-model-len", "8192",
        "--enforce-eager",
    ]
    log = open("/home/user/CBNU/logs/vllm_judge.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    print(f"  Judge vLLM PID={proc.pid}, waiting on port {JUDGE_PORT}...")
    if not wait_for_server(JUDGE_PORT, timeout=300):
        proc.kill()
        raise RuntimeError("Judge vLLM server failed to start")
    print("  Judge server ready")
    return proc


def stop_judge_vllm(proc):
    if proc is None:
        print("  Judge was pre-started externally ? leaving it running")
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    time.sleep(3)


def main():
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.run_config import RunConfig

    # Load results
    with open(RESULTS_PATH, encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} inference records")

    # Start judge
    print("Starting EXAONE-32B judge server...")
    judge_proc = start_judge_vllm()

    try:
        from openai import OpenAI as OAI
        cli = OAI(base_url=f"http://localhost:{JUDGE_PORT}/v1", api_key="EMPTY")
        serving_model = cli.models.list().data[0].id
        print(f"Serving model: {serving_model}")

        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=serving_model,
                base_url=f"http://localhost:{JUDGE_PORT}/v1",
                api_key="EMPTY",
                temperature=0,
                max_tokens=2048,
            )
        )

        print("Loading BGE-M3 embeddings...")
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

        summary = []
        done_keys = set()
        if SUMMARY_PATH.exists():
            try:
                df_ex = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
                for _, row in df_ex.iterrows():
                    has_ctx  = row["condition"] in ("rag", "ft_rag")
                    ans_ok   = pd.notna(row.get("answer_relevancy"))
                    faith_ok = pd.notna(row.get("faithfulness"))  if has_ctx else True
                    ctx_ok   = pd.notna(row.get("context_recall")) if has_ctx else True
                    if ans_ok and faith_ok and ctx_ok:
                        done_keys.add((row["model"], row["condition"]))
                        summary.append(row.to_dict())
                        print(f"[RESUME] {row['model']}/{row['condition']} — already valid, skipping")
            except Exception as e:
                print(f"Could not read existing CSV: {e}")

        for model in MODELS:
            for cond in CONDITIONS:
                subset = [r for r in records if r["model"] == model and r["condition"] == cond]
                if not subset:
                    print(f"[SKIP] {model}/{cond} — no data")
                    continue

                if (model, cond) in done_keys:
                    print(f"[SKIP] {model}/{cond} — already valid in CSV")
                    continue

                print(f"\n[{model}/{cond}] n={len(subset)}")

                has_context = cond in ("rag", "ft_rag")
                active_metrics = [faithfulness, answer_relevancy, context_recall] if has_context else [answer_relevancy]

                ds_dict = {
                    "question":     [r["question"] for r in subset],
                    "answer":       [(r.get("answer") or r.get("pred_answer", ""))[:MAX_ANS_CHARS] for r in subset],
                    "contexts":     [[r.get("context", "")[:MAX_CTX_CHARS]] for r in subset],
                    "ground_truth": [r.get("ground_truth") or r.get("gold_answer", "") for r in subset],
                }

                try:
                    ds = Dataset.from_dict(ds_dict)
                    result = evaluate(
                        ds,
                        metrics=active_metrics,
                        llm=ragas_llm,
                        embeddings=ragas_emb,
                        run_config=run_config,
                        raise_exceptions=False,
                    )
                    df = result.to_pandas()

                    row = {"model": model, "condition": cond, "n": len(subset)}
                    for col in ["faithfulness", "answer_relevancy", "context_recall"]:
                        if col in df.columns:
                            row[col] = round(df[col].dropna().mean(), 4)
                        else:
                            row[col] = None

                    summary.append(row)
                    print(f"  faithfulness={row.get('faithfulness')}  "
                          f"answer_rel={row.get('answer_relevancy')}  "
                          f"ctx_recall={row.get('context_recall')}")

                    pd.DataFrame(summary).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
                    print(f"  Saved interim to {SUMMARY_PATH}")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    summary.append({"model": model, "condition": cond, "n": len(subset), "error": str(e)})

        df_final = pd.DataFrame(summary)
        print("\n=== Final Results ===")
        print(df_final.to_string(index=False))
        df_final.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
        print(f"\nSaved: {SUMMARY_PATH}")

    finally:
        print("Stopping judge server...")
        stop_judge_vllm(judge_proc)
        print("Done")


if __name__ == "__main__":
    main()
