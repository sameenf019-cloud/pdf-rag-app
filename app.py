"""
PDF RAG Chat App
----------------
Pipeline:
  PDF upload -> text extraction (pypdf) -> token-based chunking (tiktoken)
  -> embeddings (fastembed, open-source BAAI/bge-small-en-v1.5)
  -> FAISS vector index (open-source, in-memory)
  -> retrieval -> Groq-hosted open-source LLM answers using the retrieved context.
"""

import hashlib
import os

import faiss
import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Chat with your PDF (RAG)", page_icon="📄", layout="wide")

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # open-source, 384-dim, fast on CPU

# Open-weight models served by Groq. Availability can change over time:
# check https://console.groq.com/docs/models if one stops working.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a PDF document. "
    "Use ONLY the provided context excerpts to answer. "
    "If the answer is not in the context, say you could not find it in the document. "
    "Mention page numbers (e.g. 'p. 3') when you use information from the context. "
    "Be clear and concise."
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL_NAME)


@st.cache_resource
def get_encoder():
    """Tokenizer used for token-based chunking. Falls back to words if unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def get_groq_api_key() -> str:
    """Look for the key in Streamlit secrets, then env vars, then the sidebar input."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "") or st.session_state.get("sidebar_key", "")


def key_is_configured() -> bool:
    """True if the key comes from Streamlit secrets or an environment variable."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return True
    except Exception:
        pass
    return bool(os.environ.get("GROQ_API_KEY"))


def extract_pages(pdf_file) -> list[dict]:
    """Return [{'page': 1, 'text': '...'}, ...] for pages that contain text."""
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def split_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of `chunk_size` tokens."""
    step = max(chunk_size - overlap, 1)
    enc = get_encoder()
    pieces = []
    if enc is not None:
        tokens = enc.encode(text)
        for start in range(0, len(tokens), step):
            pieces.append(enc.decode(tokens[start : start + chunk_size]))
            if start + chunk_size >= len(tokens):
                break
    else:  # fallback: whitespace "tokens"
        words = text.split()
        for start in range(0, len(words), step):
            pieces.append(" ".join(words[start : start + chunk_size]))
            if start + chunk_size >= len(words):
                break
    return [p.strip() for p in pieces if p.strip()]


def build_index(pdf_bytes: bytes, chunk_size: int, overlap: int):
    """PDF bytes -> (faiss index, chunk metadata list)."""
    from io import BytesIO

    pages = extract_pages(BytesIO(pdf_bytes))
    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned images (needs OCR)."
        )

    chunks = []
    for p in pages:
        for piece in split_tokens(p["text"], chunk_size, overlap):
            chunks.append({"page": p["page"], "text": piece})

    embedder = load_embedder()
    vectors = np.array(
        list(embedder.embed([c["text"] for c in chunks])), dtype="float32"
    )
    faiss.normalize_L2(vectors)  # cosine similarity via inner product

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks, len(pages)


def retrieve(question: str, top_k: int) -> list[dict]:
    embedder = load_embedder()
    q_vec = np.array(list(embedder.query_embed([question])), dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, ids = st.session_state.index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = st.session_state.chunks[idx]
        results.append({**chunk, "score": float(score)})
    return results


def stream_answer(client: Groq, model: str, messages: list[dict]):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("index", None)
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("doc_name", None)
st.session_state.setdefault("doc_hash", None)
st.session_state.setdefault("messages", [])  # [{'role','content','sources'}]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if not key_is_configured():
        st.text_input(
            "Groq API key",
            type="password",
            key="sidebar_key",
            help="Get a free key at https://console.groq.com/keys",
        )

    model = st.selectbox("LLM (open-source, via Groq)", GROQ_MODELS)

    st.subheader("Chunking & retrieval")
    chunk_size = st.slider("Chunk size (tokens)", 100, 450, 300, step=50)
    overlap = st.slider("Chunk overlap (tokens)", 0, 150, 50, step=10)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)

    st.subheader("📄 Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("Process document", type="primary", disabled=uploaded is None):
        pdf_bytes = uploaded.getvalue()
        with st.spinner("Extracting, chunking, embedding and indexing..."):
            try:
                index, chunks, n_pages = build_index(pdf_bytes, chunk_size, overlap)
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.doc_name = uploaded.name
                st.session_state.doc_hash = hashlib.md5(pdf_bytes).hexdigest()
                st.session_state.messages = []
                st.success(f"Indexed {n_pages} pages into {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Could not process PDF: {e}")

    if st.session_state.doc_name:
        st.caption(
            f"Active: **{st.session_state.doc_name}** "
            f"({len(st.session_state.chunks)} chunks)"
        )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main chat UI
# ----------------------------------------------------------------------------
st.title("📄 Chat with your PDF")
st.caption(
    "RAG pipeline: PDF → chunks → embeddings → FAISS → Groq open-source LLM"
)

if st.session_state.index is None:
    st.info("👈 Upload a PDF in the sidebar and click **Process document** to begin.")

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']}** (similarity {s['score']:.2f})")
                    st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

question = st.chat_input(
    "Ask a question about the document...",
    disabled=st.session_state.index is None,
)

if question:
    api_key = get_groq_api_key()
    if not api_key:
        st.error("Please provide your Groq API key (sidebar or Streamlit secrets).")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            sources = retrieve(question, top_k)
            context = "\n\n".join(
                f"[Page {s['page']}]\n{s['text']}" for s in sources
            )

            # Keep a little conversation history for follow-up questions
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-6:]
            ]
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + history
                + [
                    {
                        "role": "user",
                        "content": f"Context from the document:\n{context}\n\n"
                        f"Question: {question}",
                    }
                ]
            )

            client = Groq(api_key=api_key)
            answer = st.write_stream(stream_answer(client, model, messages))

            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**Page {s['page']}** (similarity {s['score']:.2f})")
                    st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Error: {e}")
