# RAG Chatbot

로컬 LLM(Ollama) + 벡터 DB(ChromaDB) + Streamlit 웹 UI로 구성된 간단한 RAG 챗봇.

## Stack

- **UI**: Streamlit
- **LLM**: Ollama (default `llama3.1:8b`)
- **Embedding**: Ollama (`nomic-embed-text`)
- **Vector store**: ChromaDB (local persistent)

## Setup

```bash
# 1) Ollama 설치 후 모델 pull (https://ollama.com)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2) Python deps
pip install -r requirements.txt
```

## Usage

```bash
# 1) 지식 저장소에 문서 투입: knowledge/ 에 .txt / .md / .pdf 넣기

# 2) 인덱싱
python ingest.py --rebuild

# 3) 챗봇 실행
streamlit run app.py
```

브라우저가 자동으로 열리면 질문 입력, 답변과 함께 검색된 출처 청크 표시.

## Config

환경변수로 모델/엔드포인트 변경 가능:

```bash
export CHAT_MODEL=qwen2.5:7b
export EMBED_MODEL=nomic-embed-text
export OLLAMA_URL=http://localhost:11434
```

청크 사이즈·오버랩·top_k는 `config.py` 수정.

## Files

| File | Role |
|------|------|
| `app.py` | Streamlit 채팅 UI (대화 이력, 출처 표시) |
| `rag.py` | 리트리벌 + 프롬프트 구성 + Ollama 스트리밍 호출 |
| `ingest.py` | `knowledge/` 문서 → 청킹 → 임베딩 → Chroma 저장 |
| `config.py` | 모델명·경로·하이퍼파라미터 |
| `knowledge/` | 사용자 문서 투입 디렉토리 (gitignore) |
| `chroma_db/` | 벡터 DB 영속 저장소 (자동 생성, gitignore) |
