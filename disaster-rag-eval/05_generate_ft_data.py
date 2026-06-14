"""
05_generate_ft_data.py
───────────────────────────────────────────────────────────────────
한국어 재난·안전 도메인 QLoRA 파인튜닝용 학습 데이터 생성

사용법:
    python 05_generate_ft_data.py

Titan 서버에서 vLLM 먼저 실행:
    python -m vllm.entrypoints.openai.api_server \
        --model <모델 경로 또는 HF 허브명> \
        --port 8000 --dtype bfloat16

출력: data/ft_dataset.json
───────────────────────────────────────────────────────────────────
"""

import json
import os
import re
import time
import random
from pathlib import Path
from collections import Counter

# ─── 설정 ───────────────────────────────────────────────────────
VLLM_URL     = "http://localhost:8000/v1"
VLLM_KEY     = "EMPTY"          # vLLM은 아무 값이나 OK
GEN_MODEL    = "/home/user/models/EXAONE-3.5-7.8B-Instruct"

DATA_DIR     = Path("data/raw")
EVAL_JSON    = Path("data/qa_dataset.json")
OUTPUT_JSON  = Path("data/ft_dataset.json")

CHUNK_SIZE   = 600   # 글자 수 기준
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 120  # 너무 짧은 청크 제외
TARGET_PER_CATEGORY = 30  # 카테고리당 최대 QA 수

# 유형별 목표 비율 (평가셋과 유사하게)
TYPE_WEIGHTS = {
    "수치법령형": 0.20,
    "절차형":    0.35,
    "안전판단형": 0.28,
    "복합추론형": 0.17,
}
QUESTION_TYPES = list(TYPE_WEIGHTS.keys())
# ────────────────────────────────────────────────────────────────


# ─── 1. 문서 로딩 ────────────────────────────────────────────────
def load_pdf(path: Path) -> str:
    try:
        import fitz
        doc = fitz.open(str(path))
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"  [PDF 오류] {path.name}: {e}")
        return ""

def load_docx(path: Path) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"  [DOCX 오류] {path.name}: {e}")
        return ""

def load_doc(path: Path) -> str:
    """구형 .doc 파일 — docx2txt 또는 LibreOffice 활용"""
    try:
        import docx2txt
        return docx2txt.process(str(path))
    except ImportError:
        pass
    # LibreOffice fallback
    try:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "txt:Text",
                 "--outdir", tmpdir, str(path)],
                capture_output=True, timeout=30
            )
            txt_path = Path(tmpdir) / (path.stem + ".txt")
            if txt_path.exists():
                return txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  [DOC 오류] {path.name}: {e}")
    return ""

def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext == ".doc":
        return load_doc(path)
    # .hwpx, .jpg, .png 등은 현재 스킵 (OCR 필요)
    print(f"  [스킵] 지원하지 않는 형식: {path.name}")
    return ""


# ─── 2. 텍스트 청킹 ──────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    # 빈 줄 기준으로 1차 분리 → 이후 size 기준으로 합치기
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = (current + "\n" + para).strip()
        else:
            if len(current) >= MIN_CHUNK_LEN:
                chunks.append(current)
            # 오버랩: 이전 청크 마지막 부분 포함
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + "\n" + para).strip()
    if len(current) >= MIN_CHUNK_LEN:
        chunks.append(current)
    return chunks


# ─── 3. 평가셋 소스 추출 (데이터 누수 방지) ──────────────────────
def get_eval_sources() -> set:
    """qa_dataset.json에서 source/context 키를 찾아 사용된 텍스트 조각을 반환"""
    if not EVAL_JSON.exists():
        print(f"[경고] {EVAL_JSON} 없음 — 소스 제외 건너뜀")
        return set()
    with open(EVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)

    sources = set()
    if isinstance(data, list):
        for item in data:
            # 가능한 키명들
            for key in ["context", "source", "reference", "document", "passage"]:
                val = item.get(key, "")
                if isinstance(val, str) and len(val) > 30:
                    # 앞 100자를 fingerprint로 사용
                    sources.add(val[:100].strip())
            # 정답 텍스트도 등록 (혹시 청크에서 직접 나온 경우)
            for key in ["answer", "output", "response"]:
                val = item.get(key, "")
                if isinstance(val, str) and len(val) > 20:
                    sources.add(val[:80].strip())
    print(f"[소스 필터] 평가셋 fingerprint {len(sources)}개 등록")
    return sources

def is_eval_chunk(chunk: str, eval_sources: set, threshold: int = 60) -> bool:
    """청크가 평가셋 소스와 겹치면 True"""
    prefix = chunk[:100].strip()
    for src in eval_sources:
        # 앞부분 60자가 겹치면 동일 문서로 판단
        if prefix[:threshold] == src[:threshold]:
            return True
    return False


# ─── 4. LLM으로 QA 생성 ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "당신은 한국어 재난·안전 도메인 전문가입니다.\n"
    "주어진 문서를 바탕으로 학습용 QA 쌍을 생성해주세요.\n\n"
    "질문 유형:\n"
    "- 수치법령형: 법령, 수치, 기준값 확인 (예: '~는 몇 m 이상인가?')\n"
    "- 절차형: 행동 순서·절차 (예: '~할 때 올바른 순서는?')\n"
    "- 안전판단형: 상황별 안전 판단 (예: '~상황에서 어떻게 해야 하는가?')\n"
    "- 복합추론형: 복수 정보 종합 추론 (예: '~인 경우와 ~인 경우의 차이는?')\n\n"
    "반드시 아래 JSON 형식만 출력하세요 (다른 텍스트 금지):\n"
    '{"question_type": "유형명", "question": "질문", "answer": "정답"}'
)

def generate_qa(client, chunk: str, category: str, qa_type: str) -> dict | None:
    user_msg = (
        f"다음 재난안전 문서에서 『{qa_type}』 유형의 QA 1개를 생성하세요.\n\n"
        f"[문서]\n{chunk[:500]}\n\n"
        "JSON만 출력하세요."
    )
    try:
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.75,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        # JSON만 추출 (```json ... ``` 래핑 처리)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        qa = json.loads(m.group())
        if not (qa.get("question") and qa.get("answer")):
            return None
        qa["source_category"] = category
        qa["context"] = chunk[:500]
        return qa
    except Exception as e:
        print(f"    [생성 오류] {e}")
        return None


# ─── 5. SFTTrainer용 포맷 변환 ───────────────────────────────────
SYSTEM_FT = (
    "당신은 한국어 재난·안전 전문가입니다. "
    "질문에 대해 정확하고 구체적인 답변을 제공하세요."
)

def to_sft_format(qa: dict) -> dict:
    """SFTTrainer ChatML 포맷 (messages 키)"""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_FT},
            {"role": "user",      "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ],
        "question_type":   qa.get("question_type", ""),
        "source_category": qa.get("source_category", ""),
    }


# ─── 6. 메인 ─────────────────────────────────────────────────────
def main():
    from openai import OpenAI
    client = OpenAI(base_url=VLLM_URL, api_key=VLLM_KEY)

    eval_sources = get_eval_sources()

    # 유형 순서 결정 (가중치 기반 셔플)
    def sample_type():
        return random.choices(QUESTION_TYPES, weights=list(TYPE_WEIGHTS.values()))[0]

    all_ft = []
    category_counts: dict[str, int] = {}

    categories = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )
    print(f"\n총 카테고리: {len(categories)}개\n{'='*50}")

    for cat_dir in categories:
        cat = cat_dir.name
        category_counts[cat] = 0
        print(f"\n[{cat}]")

        doc_files = [
            f for f in sorted(cat_dir.iterdir())
            if f.suffix.lower() in {".pdf", ".docx", ".doc"}
        ]
        if not doc_files:
            print("  텍스트 파일 없음 (이미지/hwpx 제외됨)")
            continue

        for doc_file in doc_files:
            print(f"  로딩: {doc_file.name}")
            text = load_document(doc_file)
            if len(text.strip()) < 200:
                print(f"  텍스트 너무 짧음, 스킵")
                continue

            chunks = chunk_text(text)
            print(f"  청크 {len(chunks)}개")

            for chunk in chunks:
                if category_counts[cat] >= TARGET_PER_CATEGORY:
                    break
                if is_eval_chunk(chunk, eval_sources):
                    print(f"  [제외] 평가셋 소스와 겹침")
                    continue

                qa_type = sample_type()
                qa = generate_qa(client, chunk, cat, qa_type)
                if qa:
                    sft_item = to_sft_format(qa)
                    all_ft.append(sft_item)
                    category_counts[cat] += 1
                    q_short = qa["question"][:45].replace("\n", " ")
                    print(f"  ✓ [{qa_type}] {q_short}...")
                    time.sleep(0.05)

            if category_counts[cat] >= TARGET_PER_CATEGORY:
                print(f"  카테고리 목표 {TARGET_PER_CATEGORY}개 달성")

    # ─── 저장 ───
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_ft, f, ensure_ascii=False, indent=2)

    # ─── 통계 ───
    total = len(all_ft)
    print(f"\n{'='*50}")
    print(f"총 생성 QA: {total}개")
    print(f"저장 경로: {OUTPUT_JSON}")

    type_dist = Counter(item["question_type"] for item in all_ft)
    print("\n[유형별 분포]")
    for t in QUESTION_TYPES:
        n = type_dist.get(t, 0)
        pct = n / total * 100 if total else 0
        print(f"  {t:<10}: {n:>4}개 ({pct:.1f}%)")

    print("\n[카테고리별 생성 수]")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25}: {cnt}개")


if __name__ == "__main__":
    main()
