"""
convert_ft_dataset.py
기존 ft_dataset.json (context 없음) → ft_dataset_rag.json (FAISS 검색 context 포함)
inference 프롬프트 형식과 완전 동일하게 맞춤 (vLLM 불필요)

실행: python convert_ft_dataset.py
"""
import json, pickle, faiss, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

DATA_PATH   = Path("data/ft_dataset.json")
OUT_PATH    = Path("data/ft_dataset_rag.json")
INDEX_DIR   = Path("data/faiss_index")
EMBED_MODEL = "BAAI/bge-m3"
TOP_K       = 3
MAX_CTX_CHARS = 800  # 학습 메모리 제한 — ~250 tokens, 전체 시퀀스 ~450 tokens 목표

# inference와 완전 동일한 RAG 프롬프트 (03_run_experiments_v2.py의 RAG_PROMPT)
RAG_PROMPT = (
    "당신은 한국어 재난·안전 전문가입니다.\n"
    "아래 [참고 자료]를 바탕으로 2~4문장으로 답하세요.\n\n"
    "[참고 자료]\n{context}\n\n"
    "질문: {question}\n답변:"
)


class Embedder:
    def __init__(self, model_name, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16).to(device)
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


def get_context(question, embedder, index, chunks):
    q_emb = embedder.encode([question])
    faiss.normalize_L2(q_emb)
    _, I = index.search(q_emb, TOP_K)
    ctx = "\n\n".join(chunks[i]["text"] for i in I[0])
    return ctx[:MAX_CTX_CHARS]  # 학습 시퀀스 길이 제한


def extract_question(item):
    """현재 ft_dataset의 messages에서 질문 텍스트 추출"""
    for msg in item["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def extract_answer(item):
    for msg in item["messages"]:
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def main():
    print("=== FAISS 로드 ===")
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"  인덱스: {index.ntotal}개 벡터, 청크: {len(chunks)}개")

    print("\n=== 임베더 로드 (BGE-M3, CPU) ===")
    embedder = Embedder(EMBED_MODEL, device="cpu")

    print("\n=== ft_dataset.json 로드 ===")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  총 {len(data)}개 항목")

    print("\n=== 변환 시작 ===")
    new_data = []
    for i, item in enumerate(data):
        question = extract_question(item)
        answer   = extract_answer(item)

        # FAISS로 context 검색 (inference와 동일 방식)
        context = get_context(question, embedder, index, chunks)

        # inference 프롬프트와 완전 동일한 형식으로 구성 (system 없음, user에 모두 포함)
        new_item = {
            "messages": [
                {
                    "role": "user",
                    "content": RAG_PROMPT.format(context=context, question=question)
                },
                {
                    "role": "assistant",
                    "content": answer
                },
            ],
            "question_type":   item.get("question_type", ""),
            "source_category": item.get("source_category", ""),
        }
        new_data.append(new_item)

        if (i + 1) % 20 == 0 or (i + 1) == len(data):
            print(f"  [{i+1}/{len(data)}] 완료")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUT_PATH} ({len(new_data)}개)")
    # 샘플 확인
    sample = new_data[0]
    print("\n=== 샘플 확인 (첫 번째 항목) ===")
    user_preview = sample["messages"][0]["content"][:200]
    print(f"USER (첫 200자):\n{user_preview}\n...")
    print(f"ASSISTANT: {sample['messages'][1]['content'][:80]}...")


if __name__ == "__main__":
    main()
