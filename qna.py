import streamlit as st
import requests
import time
from typing import List, Dict
from html import escape  # ✅ FIX — swapped st.escape

st.set_page_config(page_title="Repo Q/A Chatbot", layout="centered")

# --- Dark theme CSS for chat bubbles ---
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    html, body, .stApp { background-color: #000000; color: #e6eef6; }

    .title { font-size:32px; font-weight:800; background: linear-gradient(90deg,#06b6d4,#3b82f6); -webkit-background-clip: text; color: transparent; }
    .muted { color: #94a3b8; }
    .chat-shell { background: linear-gradient(180deg, rgba(17,24,39,0.6), rgba(10,12,14,0.4)); border: 1px solid #1f2937; padding: 16px; border-radius: 14px; }

    .bot-bubble { background: #0f1724; border: 1px solid #1f2937; color: #d1d5db; padding: 12px 14px; border-radius: 12px; display:inline-block; max-width:84%; }
    .user-bubble { background: transparent; border: 2px solid #2dd4bf; color: #e6eef6; padding: 12px 14px; border-radius: 12px; display:inline-block; max-width:84%; }

    .code-preview { background: #071018; border:1px solid #12202a; padding:10px; border-radius:8px; font-family: monospace; color:#cbd5e1; }

    .chat-list { max-height: 52vh; overflow-y: auto; padding-right: 8px; }

    .meta { color: #94a3b8; font-size:13px; }

    .stTextInput > div > input { background: #0b1220 !important; color: #e6eef6 !important; }
    .stButton>button{ background: linear-gradient(90deg,#06b6d4,#3b82f6) !important; color: #001018 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""
if "processed" not in st.session_state:
    st.session_state.processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loading" not in st.session_state:
    st.session_state.loading = False
if "processing" not in st.session_state:
    st.session_state.processing = False

API_BASE = "http://127.0.0.1:8000/graph-rag"

def call_process_repo(repo_url: str, timeout: int = 60) -> Dict:
    try:
        r = requests.post(f"{API_BASE}/process-repo", json={"repo_url": repo_url}, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def call_query(query: str, top_k: int = 5, timeout: int = 60) -> Dict:
    try:
        r = requests.post(f"{API_BASE}/query", json={"query": query, "top_k": top_k}, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# UI
with st.container():
    st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
    st.markdown('<div class="title">Repository Q/A — Chatbot</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<p class="muted" style="text-align:center">Process a GitHub repository and chat with it.</p>', unsafe_allow_html=True)

    cols = st.columns([3,1])
    with cols[0]:
        repo_input = st.text_input("GitHub repo URL", value=st.session_state.repo_url, placeholder="https://github.com/owner/repo")
    with cols[1]:
        if st.session_state.processed:
            st.button("✅ Repo Processed", key="processed_btn", disabled=True)
        else:
            if st.button("Process Repo", key="process_btn"):
                if not repo_input:
                    st.warning("Please enter a repo URL.")
                else:
                    st.session_state.repo_url = repo_input
                    st.session_state.processing = True
                    with st.spinner("Processing repository..."):
                        res = call_process_repo(repo_input)
                        time.sleep(0.35)
                    st.session_state.processing = False

                    if res.get("error"):
                        st.error("Failed to process repo: " + res.get("error"))
                        st.session_state.messages.append({"role": "bot", "text": "Failed to process repo. Backend may be offline.", "context": []})
                    else:
                        st.success(res.get("status", "Repo processed successfully."))
                        st.session_state.processed = True
                        st.session_state.messages.append({"role": "bot", "text": res.get("status", "Repository processed successfully."), "context": []})

    st.write("")
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

    left, right = st.columns([3,1], gap="small")
    with left:
        st.markdown('<div class="chat-list" id="chat-list">', unsafe_allow_html=True)

        if not st.session_state.messages:
            st.markdown('<div class="meta" style="text-align:center; padding:28px 0">Start by processing a repo, then ask a question.</div>', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg.get("role") == "user":
                st.markdown(
                    f"<div style='display:flex; justify-content:flex-end; margin-bottom:10px;'>"
                    f"<div class='user-bubble'>{escape(msg.get('text',''))}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                bot_html = (
                    f"<div style='display:flex; gap:10px; margin-bottom:10px;'>"
                    f"<div style='width:40px; height:40px; border-radius:8px; background:#0b1220; display:flex; align-items:center; justify-content:center;'>🤖</div>"
                    f"<div class='bot-bubble'>{escape(msg.get('text',''))}</div></div>"
                )
                st.markdown(bot_html, unsafe_allow_html=True)

                context = msg.get("context") or []
                for chunk in context[:3]:
                    file = chunk.get("file","unknown")
                    fn = chunk.get("fn","")
                    code = chunk.get("code","")
                    preview = code[:350] + ("..." if len(code) > 350 else "")
                    preview_html = (
                        f"<div style='margin-left:56px; margin-top:6px;'>"
                        f"<div class='meta'><strong>{escape(fn)}</strong> - <span>{escape(file)}</span></div>"
                        f"<div class='code-preview' style='white-space:pre-wrap;'>{escape(preview)}</div></div>"
                    )
                    st.markdown(preview_html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="meta" style="padding:6px 0">Quick Actions</div>', unsafe_allow_html=True)
        if st.button("Clear Chat"):
            st.session_state.messages = []
        if st.button("Re-process Repo"):
            if st.session_state.repo_url:
                with st.spinner("Re-processing..."):
                    res = call_process_repo(st.session_state.repo_url)
                    time.sleep(0.35)
                if res.get("error"):
                    st.error("Error: " + res.get("error"))
                else:
                    st.success("Re-processed.")
                    st.session_state.messages.append({"role": "bot", "text": "Re-processed.", "context": []})
            else:
                st.warning("No repo URL saved.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    cols_in = st.columns([8,2])
    with cols_in[0]:
        user_q = st.text_input("", value="", placeholder=("Ask a question..." if st.session_state.processed else "Process a repo first"), key='input_box')
    with cols_in[1]:
        if st.button("Send", key="send_btn"):
            if not st.session_state.processed:
                st.warning("Process a repository first.")
            elif not user_q.strip():
                st.warning("Please type a question.")
            else:
                st.session_state.messages.append({"role": "user", "text": user_q})
                st.session_state.messages.append({"role": "bot", "text": "Thinking...", "context": []})
                st.experimental_rerun()

    if st.session_state.messages and st.session_state.messages[-1].get("text") == "Thinking...":
        last_user = next((m.get("text") for m in reversed(st.session_state.messages) if m.get("role")=="user"), "")
        if last_user:
            with st.spinner("Generating answer..."):
                res = call_query(last_user, top_k=5)
                time.sleep(0.35)

            st.session_state.messages.pop()

            if res.get("error"):
                st.session_state.messages.append({"role": "bot", "text": "Failed: " + res.get("error"), "context": []})
            else:
                st.session_state.messages.append({"role": "bot", "text": res.get("response","No answer."), "context": res.get("context_chunks",[])})
            st.experimental_rerun()

    st.write("")
    st.markdown(
        '<div class="meta"><code>pip install streamlit requests</code> then <code>streamlit run app.py</code></div>',
        unsafe_allow_html=True
    )
