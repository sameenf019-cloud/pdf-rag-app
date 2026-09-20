"""
DocuChat AI - a professional PDF RAG assistant
-----------------------------------------------
Pipeline:
  PDFs -> text extraction + cleaning (pypdf) -> token-based chunking (tiktoken)
  -> embeddings (fastembed, open-source) -> FAISS vector index
  + BM25 keyword index (hybrid search, reciprocal-rank fusion)
  -> retrieval -> Groq-hosted open-source LLM answers with page-level citations.

Views: Chat | Insights (AI summary, stats, charts) | Explore (semantic search)
"""

import html
import io
import json
import os
import re
import time
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuChat AI - Chat with your PDFs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EMBED_MODELS = {
    "BAAI/bge-small-en-v1.5": "BGE Small - fast (English / Roman Urdu)",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM - supports Urdu script",
}

# Open-weight models served by Groq. If one stops working, check
# https://console.groq.com/docs/models and edit this list.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

ANSWER_STYLES = {
    "Balanced": "Give a clear, well-organised answer of moderate length.",
    "Concise": "Answer in at most 3 short sentences.",
    "Detailed": "Give a thorough, structured answer with all relevant details from the context.",
    "Bullet points": "Answer using short bullet points.",
    "Explain simply": "Explain in very simple words, as if to someone new to the topic.",
}

LANGUAGES = {
    "Auto (match my question)": "Reply in the same language and script as the user's question "
    "(English, Urdu script, or Roman Urdu).",
    "English": "Always reply in English.",
    "Urdu (اردو)": "Always reply in Urdu script (اردو).",
    "Roman Urdu": "Always reply in Roman Urdu (Urdu written with Latin letters).",
}

VIEWS = ["💬 Chat", "📊 Insights", "🔎 Explore"]

# PDF text extraction often produces broken ligatures; map them back.
LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
    "Ɵ": "ti", "Ō": "ft", "Ʃ": "tt",
}

BASE_PROMPT = (
    "You are DocuChat AI, an assistant that answers questions about the user's PDF documents. "
    "Use ONLY the numbered context excerpts provided. "
    "If the answer is not in the context, say clearly that you could not find it in the documents - "
    "never invent facts. "
    "Cite where information came from using the file name and page, e.g. (notes.pdf, p. 3)."
)

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
footer {visibility: hidden;}
.hero {
  padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.2rem; color: #fff;
  background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 55%, #06B6D4 100%);
  box-shadow: 0 8px 24px rgba(59,130,246,.25);
}
.hero h1 {margin: 0; padding: 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: .4rem 0 0; opacity: .95; font-size: 1.05rem;}
.card {
  border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1rem 1.2rem;
  background: rgba(128,128,128,.07); height: 100%;
}
.card h4 {margin: 0 0 .35rem; padding: 0;}
.card p {margin: 0; opacity: .85; font-size: .95rem;}
.src {
  border-left: 3px solid #6C63FF; padding: .55rem .85rem; margin: .5rem 0;
  background: rgba(108,99,255,.08); border-radius: 6px; font-size: .9rem;
}
.badge {
  display: inline-block; padding: 1px 10px; border-radius: 999px; font-size: .75rem;
  background: rgba(108,99,255,.22); margin-right: 6px; font-weight: 600;
}
mark {background: rgba(255,214,0,.45); color: inherit; padding: 0 2px; border-radius: 3px;}
div[data-testid="stMetric"] {
  border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .6rem .9rem;
  background: rgba(128,128,128,.07);
}
</style>
"""


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def esc(text: str) -> str:
    return html.escape(text or "")


def tokenize(text: str) -> list[str]:
    """Simple unicode-aware word tokenizer used for BM25 and highlighting."""
    return re.findall(r"\w+", text.lower())


def clean_text(text: str) -> str:
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    text = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\ue000-\uf8ff]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def highlight(text: str, query: str) -> str:
    """HTML-escape text and wrap query terms in <mark>."""
    safe = esc(text)
    terms = sorted({t for t in tokenize(query) if len(t) > 2}, key=len, reverse=True)
    if terms:
        pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
        safe = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)
    return safe.replace("\n", "<br>")


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"summary": text.strip(), "key_points": [], "suggested_questions": []}


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "401" in low or "invalid_api_key" in low or "invalid api key" in low:
        return "Your Groq API key looks invalid. Please check it and try again."
    if "429" in low or "rate limit" in low or "rate_limit" in low:
        return "Groq rate limit reached. Wait a few seconds or switch to a smaller model."
    if "model" in low and ("not found" in low or "decommission" in low or "does not exist" in low):
        return "This model is not available on Groq right now. Please pick another model."
    return f"Something went wrong: {msg}"


# ----------------------------------------------------------------------------
# Config / keys
# ----------------------------------------------------------------------------
def key_is_configured() -> bool:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return True
    except Exception:
        pass
    return bool(os.environ.get("GROQ_API_KEY"))


def get_groq_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "") or st.session_state.get("sidebar_key", "")


# ----------------------------------------------------------------------------
# Models (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embedder(name: str) -> TextEmbedding:
    return TextEmbedding(model_name=name)


@st.cache_resource
def get_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return int(len(text.split()) * 1.3)


# ----------------------------------------------------------------------------
# Ingestion: PDF -> chunks -> embeddings -> FAISS + BM25
# ----------------------------------------------------------------------------
def extract_pages(data: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password protected.")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": i, "text": text})
    return pages


def split_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(chunk_size - overlap, 1)
    enc = get_encoder()
    pieces = []
    if enc is not None:
        tokens = enc.encode(text)
        for start in range(0, len(tokens), step):
            pieces.append(enc.decode(tokens[start : start + chunk_size]))
            if start + chunk_size >= len(tokens):
                break
    else:
        words = text.split()
        for start in range(0, len(words), step):
            pieces.append(" ".join(words[start : start + chunk_size]))
            if start + chunk_size >= len(words):
                break
    return [p.strip() for p in pieces if p.strip()]


def build_kb(files, chunk_size, overlap, embed_name, on_progress) -> dict:
    """files: list of (filename, bytes). Returns the knowledge-base dict."""
    overlap = min(overlap, chunk_size // 2)
    chunks, docs, page_words, skipped = [], [], {}, []
    total_words = total_tokens = 0

    for f_i, (name, data) in enumerate(files):
        on_progress(0.05 + 0.25 * f_i / len(files), f"Reading {name}...")
        try:
            pages = extract_pages(data)
        except Exception as e:
            skipped.append(f"{name} ({e})")
            continue
        if not pages:
            skipped.append(f"{name} (no extractable text - scanned PDF?)")
            continue

        n_chunks, n_words = 0, 0
        page_words[name] = {}
        for p in pages:
            words = len(p["text"].split())
            page_words[name][p["page"]] = words
            n_words += words
            total_tokens += count_tokens(p["text"])
            for piece in split_tokens(p["text"], chunk_size, overlap):
                chunks.append({"id": len(chunks), "doc": name, "page": p["page"], "text": piece})
                n_chunks += 1
        total_words += n_words
        docs.append({"name": name, "pages": len(pages), "chunks": n_chunks, "words": n_words})

    if not chunks:
        raise ValueError(
            "No extractable text found. The PDFs may be scanned images (they need OCR first)."
        )

    embedder = load_embedder(embed_name)
    texts = [c["text"] for c in chunks]
    vectors = []
    batch = 32
    for i in range(0, len(texts), batch):
        vectors.extend(embedder.embed(texts[i : i + batch]))
        done = min(i + batch, len(texts))
        on_progress(0.3 + 0.65 * done / len(texts), f"Embedding chunks {done}/{len(texts)}...")
    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)

    on_progress(0.97, "Building search indexes...")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    bm25 = BM25Okapi([tokenize(c["text"]) or ["_"] for c in chunks])

    on_progress(1.0, "Done")
    return {
        "id": datetime.now().strftime("%H%M%S%f"),
        "index": index,
        "bm25": bm25,
        "chunks": chunks,
        "docs": docs,
        "page_words": page_words,
        "skipped": skipped,
        "words": total_words,
        "tokens": total_tokens,
        "embed_name": embed_name,
        "built_at": datetime.now(),
    }


# ----------------------------------------------------------------------------
# Retrieval (hybrid: semantic FAISS + BM25 keyword, fused with RRF)
# ----------------------------------------------------------------------------
def retrieve(question: str, top_k: int, hybrid: bool, allowed=None) -> list[dict]:
    kb = st.session_state.kb
    chunks = kb["chunks"]
    embedder = load_embedder(kb["embed_name"])

    q = np.array(list(embedder.query_embed([question])), dtype="float32")
    faiss.normalize_L2(q)
    _, ids = kb["index"].search(q, len(chunks))

    def allowed_ok(i: int) -> bool:
        return not allowed or chunks[i]["doc"] in allowed

    dense = [int(i) for i in ids[0] if i != -1 and allowed_ok(int(i))]

    if hybrid:
        bm = kb["bm25"].get_scores(tokenize(question))
        sparse = [int(i) for i in np.argsort(bm)[::-1] if bm[i] > 0 and allowed_ok(int(i))]
        fused: dict[int, float] = {}
        for rank, i in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
        for rank, i in enumerate(sparse):
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
        order = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
    else:
        order = dense

    results = []
    for i in order[:top_k]:
        vec = kb["index"].reconstruct(i)
        results.append({**chunks[i], "score": float(np.dot(q[0], vec))})
    return results


# ----------------------------------------------------------------------------
# LLM helpers (Groq)
# ----------------------------------------------------------------------------
def llm_kwargs(model: str, effort: str) -> dict:
    """gpt-oss models accept a reasoning_effort parameter."""
    if model.startswith("openai/gpt-oss"):
        return {"extra_body": {"reasoning_effort": effort}}
    return {}


def build_messages(question, sources, history, style, language) -> list[dict]:
    system = f"{BASE_PROMPT}\n\nStyle: {ANSWER_STYLES[style]}\nLanguage: {LANGUAGES[language]}"
    context = "\n\n".join(
        f"[Source {n}: {s['doc']}, page {s['page']}]\n{s['text']}" for n, s in enumerate(sources, 1)
    )
    user = f"Context excerpts:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]


def stream_answer(client, model, messages, temperature, max_tokens, effort):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        **llm_kwargs(model, effort),
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def sample_context(chunks: list[dict], max_chars: int = 12000) -> str:
    n = min(12, len(chunks))
    idxs = sorted(set(np.linspace(0, len(chunks) - 1, num=n).astype(int).tolist()))
    per = max_chars // max(len(idxs), 1)
    return "\n\n".join(
        f"[{chunks[i]['doc']}, p.{chunks[i]['page']}]\n{chunks[i]['text'][:per]}" for i in idxs
    )


def generate_insights(client, model, kb, effort) -> dict:
    prompt = (
        "You are analysing a document from the excerpts below. Return ONLY valid JSON with keys:\n"
        '"summary" (a 3-5 sentence overview), '
        '"key_points" (list of 5-7 short strings), '
        '"suggested_questions" (list of 5 questions a reader might ask that the document can answer).\n'
        "Write all values in the same language and script as the document "
        "(if it is Roman Urdu, answer in Roman Urdu).\n\n"
        f"Excerpts:\n{sample_context(kb['chunks'])}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
        **llm_kwargs(model, effort),
    )
    data = parse_json(resp.choices[0].message.content or "")
    return {
        "summary": str(data.get("summary", "")),
        "key_points": [str(x) for x in data.get("key_points", []) if x][:8],
        "suggested_questions": [str(x) for x in data.get("suggested_questions", []) if x][:6],
    }


# ----------------------------------------------------------------------------
# Session state + callbacks
# ----------------------------------------------------------------------------
def init_state():
    defaults = {"kb": None, "messages": [], "insights": None, "pending_q": None, "view": VIEWS[0]}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def ask(q: str):
    st.session_state.pending_q = q
    st.session_state.view = VIEWS[0]


def clear_chat():
    st.session_state.messages = []


def reset_all():
    st.session_state.kb = None
    st.session_state.messages = []
    st.session_state.insights = None


def chat_to_markdown() -> str:
    kb = st.session_state.kb
    names = ", ".join(d["name"] for d in kb["docs"]) if kb else "-"
    lines = ["# DocuChat AI - chat export", f"Documents: {names}", f"Exported: {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"**{who}:** {m['content']}\n")
        for n, s in enumerate(m.get("sources", []), 1):
            lines.append(f"- Source {n}: {s['doc']}, p. {s['page']}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# UI components
# ----------------------------------------------------------------------------
def render_sources(sources: list[dict]):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for n, s in enumerate(sources, 1):
            snippet = s["text"][:500] + ("..." if len(s["text"]) > 500 else "")
            st.markdown(
                f'<div class="src"><span class="badge">[{n}] {esc(s["doc"])} · p.{s["page"]}</span>'
                f'<span class="badge">similarity {s["score"]:.2f}</span><br>{esc(snippet)}</div>',
                unsafe_allow_html=True,
            )


def render_message(m: dict):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("sources"):
                render_sources(m["sources"])
            meta = m.get("meta")
            if meta:
                st.caption(
                    f"⏱ retrieval {meta['retrieval_ms']} ms · generation {meta['gen_s']:.1f}s · {meta['model']}"
                )


def render_hero():
    st.markdown(
        '<div class="hero"><h1>📄 DocuChat AI</h1>'
        "<p>Upload your PDFs, ask questions in English, Urdu or Roman Urdu, and get answers "
        "with page-level citations - powered by open-source models.</p></div>",
        unsafe_allow_html=True,
    )


def render_landing():
    st.markdown("### 🚀 Get started in 3 steps")
    steps = [
        ("1️⃣ Upload", "Add one or more PDF files from the sidebar."),
        ("2️⃣ Process", "Click <b>Process documents</b>. Text is cleaned, chunked, embedded and indexed."),
        ("3️⃣ Ask", "Chat with your documents, explore insights, or run a semantic search."),
    ]
    for col, (title, text) in zip(st.columns(3), steps):
        col.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown("### ✨ What you get")
    feats = [
        ("🔍 Hybrid search", "Semantic (FAISS) + keyword (BM25) retrieval for better accuracy."),
        ("📚 Cited answers", "Every answer shows the file, page and source text it came from."),
        ("🌍 Multilingual", "English, Urdu and Roman Urdu - choose the answer language."),
        ("📊 Insights", "Auto summary, key points, suggested questions and document stats."),
    ]
    for col, (title, text) in zip(st.columns(4), feats):
        col.markdown(f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def view_chat(cfg: dict):
    pending = st.session_state.pop("pending_q", None)
    typed = st.chat_input("Ask anything about your documents...")
    question = typed or pending

    if not st.session_state.messages and not question:
        st.markdown("#### 👋 Ask me anything about your documents")
        suggestions = (st.session_state.insights or {}).get("suggested_questions", [])
        if suggestions:
            st.caption("Suggested questions - click to ask:")
            for i, q in enumerate(suggestions):
                st.button(q, key=f"chip_{i}", on_click=ask, args=(q,))
        else:
            st.caption("Type a question below, or generate suggested questions in the Insights tab.")

    for m in st.session_state.messages:
        render_message(m)

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]
    ]
    st.session_state.messages.append({"role": "user", "content": question})

    answered = False
    with st.chat_message("assistant"):
        api_key = get_groq_api_key()
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar (or add it to Streamlit secrets).")
            return
        try:
            t0 = time.time()
            with st.spinner("Searching your documents..."):
                sources = retrieve(question, cfg["top_k"], cfg["hybrid"], cfg["allowed"])
            t1 = time.time()
            if not sources:
                st.warning("No relevant passages found in the selected documents.")
                return

            messages = build_messages(question, sources, history, cfg["style"], cfg["language"])
            client = Groq(api_key=api_key)
            answer = st.write_stream(
                stream_answer(client, cfg["model"], messages, cfg["temperature"], cfg["max_tokens"], cfg["effort"])
            )
            t2 = time.time()

            render_sources(sources)
            meta = {"retrieval_ms": int((t1 - t0) * 1000), "gen_s": t2 - t1, "model": cfg["model"]}
            st.caption(f"⏱ retrieval {meta['retrieval_ms']} ms · generation {meta['gen_s']:.1f}s · {meta['model']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "meta": meta}
            )
            answered = True
        except Exception as e:
            st.error(friendly_error(e))

    if answered:
        st.rerun()  # refresh the sidebar (export button) with the new message


def view_insights(cfg: dict):
    kb = st.session_state.kb
    cols = st.columns(5)
    cols[0].metric("Documents", len(kb["docs"]))
    cols[1].metric("Pages", sum(d["pages"] for d in kb["docs"]))
    cols[2].metric("Chunks", len(kb["chunks"]))
    cols[3].metric("Words", f"{kb['words']:,}")
    cols[4].metric("Read time", f"{max(1, round(kb['words'] / 200))} min")

    st.markdown("### ✨ AI summary")
    ins = st.session_state.insights
    if ins and ins.get("summary"):
        st.markdown(f'<div class="card">{esc(ins["summary"])}</div>', unsafe_allow_html=True)
        if ins.get("key_points"):
            st.markdown("**Key points**")
            for kp in ins["key_points"]:
                st.markdown(f"- {kp}")
        if ins.get("suggested_questions"):
            st.markdown("**Suggested questions** - click to ask in Chat")
            for i, q in enumerate(ins["suggested_questions"]):
                st.button(q, key=f"ins_q_{i}", on_click=ask, args=(q,))
    else:
        st.info("No summary yet. Click the button below to generate one.")

    if st.button("✨ Generate / refresh summary"):
        api_key = get_groq_api_key()
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Analysing your documents..."):
                try:
                    st.session_state.insights = generate_insights(
                        Groq(api_key=api_key), cfg["model"], kb, cfg["effort"]
                    )
                    st.rerun()
                except Exception as e:
                    st.error(friendly_error(e))

    st.markdown("### 📑 Documents")
    st.dataframe(pd.DataFrame(kb["docs"]).rename(columns=str.title), hide_index=True)

    st.markdown("### 📈 Words per page")
    doc_name = st.selectbox("Document", list(kb["page_words"].keys()))
    pw = kb["page_words"][doc_name]
    st.bar_chart(pd.DataFrame({"Words": list(pw.values())}, index=[f"p.{p}" for p in pw.keys()]))

    with st.expander("ℹ️ Index details"):
        st.write(
            f"Embedding model: `{kb['embed_name']}` · Approx. tokens: {kb['tokens']:,} · "
            f"Built at {kb['built_at']:%H:%M:%S}"
        )


def view_explore(cfg: dict):
    st.markdown("### 🔎 Semantic search")
    st.caption("Find passages by meaning and keywords - no LLM needed.")
    query = st.text_input("Search", placeholder="e.g. main problem, dosage, pricing...", label_visibility="collapsed")
    n = st.slider("Number of results", 3, 15, 6)
    if not query:
        return
    with st.spinner("Searching..."):
        results = retrieve(query, n, cfg["hybrid"], cfg["allowed"])
    if not results:
        st.warning("No matching passages.")
        return
    for i, r in enumerate(results, 1):
        st.markdown(
            f'<div class="src"><span class="badge">#{i} {esc(r["doc"])} · p.{r["page"]}</span>'
            f'<span class="badge">similarity {r["score"]:.2f}</span><br>{highlight(r["text"], query)}</div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
init_state()
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 DocuChat AI")
    st.caption("Open-source RAG · FAISS · Groq")

    if not key_is_configured():
        st.text_input(
            "Groq API key",
            type="password",
            key="sidebar_key",
            help="Get a free key at https://console.groq.com/keys",
        )

    st.markdown("#### 1 · Upload")
    uploads = st.file_uploader(
        "PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    with st.expander("🔧 Retrieval settings"):
        embed_name = st.selectbox(
            "Embedding model", list(EMBED_MODELS.keys()), format_func=lambda k: EMBED_MODELS[k]
        )
        if "multilingual" in embed_name:
            st.caption("Tip: this model works best with smaller chunks (100-200 tokens).")
        chunk_size = st.slider("Chunk size (tokens)", 100, 450, 300, step=50)
        overlap = st.slider("Chunk overlap (tokens)", 0, 150, 50, step=10)
        top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)
        hybrid = st.toggle("Hybrid search (semantic + keyword)", value=True)
        st.caption("Changing embedding/chunk settings requires clicking Process again.")

    with st.expander("🤖 Model settings"):
        model = st.selectbox("LLM (open-source, via Groq)", GROQ_MODELS)
        style = st.selectbox("Answer style", list(ANSWER_STYLES.keys()))
        language = st.selectbox("Answer language", list(LANGUAGES.keys()))
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.1)
        max_tokens = st.slider("Max answer length (tokens)", 512, 4096, 1500, step=256)
        effort = st.selectbox(
            "Reasoning effort (gpt-oss only)", ["low", "medium", "high"], index=1
        )
        auto_insights = st.toggle("Auto-generate summary after processing", value=True)

    st.markdown("#### 2 · Process")
    process = st.button("⚡ Process documents", type="primary", disabled=not uploads)

    if process:
        files = [(f.name, f.getvalue()) for f in uploads]
        bar = st.progress(0.0, text="Starting...")
        try:
            kb_new = build_kb(
                files, chunk_size, overlap, embed_name,
                lambda frac, msg: bar.progress(min(frac, 1.0), text=msg),
            )
            st.session_state.kb = kb_new
            st.session_state.messages = []
            st.session_state.insights = None
            st.session_state.view = VIEWS[0]
            bar.empty()
            st.success(f"Indexed {len(kb_new['docs'])} document(s) into {len(kb_new['chunks'])} chunks.")
            for s in kb_new["skipped"]:
                st.warning(f"Skipped: {s}")

            api_key = get_groq_api_key()
            if auto_insights and api_key:
                with st.spinner("Generating summary and suggested questions..."):
                    try:
                        st.session_state.insights = generate_insights(
                            Groq(api_key=api_key), model, kb_new, effort
                        )
                    except Exception as e:
                        st.warning(f"Summary skipped: {friendly_error(e)}")
        except Exception as e:
            bar.empty()
            st.error(f"Could not process PDFs: {e}")

    kb = st.session_state.kb
    allowed = None
    if kb:
        st.markdown("#### Active knowledge base")
        for d in kb["docs"]:
            st.caption(f"📎 {d['name']} · {d['pages']} pages · {d['chunks']} chunks")
        if len(kb["docs"]) > 1:
            names = [d["name"] for d in kb["docs"]]
            allowed = st.multiselect("Search in", names, default=names, key=f"doc_filter_{kb['id']}")
        c1, c2 = st.columns(2)
        c1.button("🧹 Clear chat", on_click=clear_chat)
        c2.button("♻️ Reset", on_click=reset_all)
        st.download_button(
            "⬇️ Export chat (.md)",
            data=chat_to_markdown(),
            file_name="docuchat_export.md",
            mime="text/markdown",
            disabled=not st.session_state.messages,
        )

cfg = {
    "model": model, "style": style, "language": language, "temperature": temperature,
    "max_tokens": max_tokens, "effort": effort, "top_k": top_k, "hybrid": hybrid, "allowed": allowed,
}

render_hero()

if st.session_state.kb is None:
    render_landing()
else:
    st.radio("View", VIEWS, key="view", horizontal=True, label_visibility="collapsed")
    if st.session_state.view == VIEWS[0]:
        view_chat(cfg)
    elif st.session_state.view == VIEWS[1]:
        view_insights(cfg)
    else:
        view_explore(cfg)
