import streamlit as st

from config import CHAT_MODEL, EMBED_MODEL, TOP_K
from rag import answer

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("RAG Chatbot")

with st.sidebar:
    st.caption("Config")
    st.write(f"Chat model: `{CHAT_MODEL}`")
    st.write(f"Embed model: `{EMBED_MODEL}`")
    k = st.slider("Top-K retrieval", 1, 10, TOP_K)
    show_sources = st.toggle("Show retrieved sources", value=True)
    if st.button("Clear chat"):
        st.session_state.pop("history", None)
        st.rerun()
    st.markdown("---")
    st.caption("How to add knowledge")
    st.markdown(
        "1. Drop `.txt` / `.md` / `.pdf` into `knowledge/`\n"
        "2. Run `python ingest.py --rebuild`\n"
        "3. Reload this page"
    )

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(sources: list[dict]):
    with st.expander(f"Retrieved sources ({len(sources)})"):
        for s in sources:
            dist = s.get("distance")
            header = f"**{s['source']}**"
            if dist is not None:
                header += f"  ·  distance={dist:.3f}"
            st.markdown(header)
            preview = s["text"][:500] + ("..." if len(s["text"]) > 500 else "")
            st.code(preview)


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            render_sources(msg["sources"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.history[:-1]
        ]
        try:
            contexts, token_gen = answer(prompt, llm_history, k=k)
            reply = st.write_stream(token_gen)
        except Exception as e:
            reply = f"⚠️ Error: {e}\n\nIs Ollama running? Try `ollama serve`."
            st.error(reply)
            contexts = []

        if show_sources and contexts:
            render_sources(contexts)

    st.session_state.history.append(
        {"role": "assistant", "content": reply, "sources": contexts}
    )
