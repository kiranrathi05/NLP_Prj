# # app.py
# """
# Unified Multi-Model Summarizer (single-file Streamlit app)

# Models: google/pegasus-xsum, facebook/bart-large-cnn, t5-small, sshleifer/distilbart-cnn-12-6
# Features:
#  - Paste text or upload TXT/PDF
#  - Chunking for long inputs
#  - Generate summaries from 4 models
#  - Compute word count, compression, ROUGE1-F1, cosine similarity
#  - Table + Bar Charts
#  - Download JSON results
# """

# import streamlit as st
# from io import BytesIO
# import re, math, json
# from typing import List
# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# try:
#     from PyPDF2 import PdfReader
# except:
#     PdfReader = None

# try:
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity
#     SKL_AVAILABLE = True
# except:
#     SKL_AVAILABLE = False

# try:
#     import pandas as pd
#     PANDAS_AVAILABLE = True
# except:
#     PANDAS_AVAILABLE = False


# # ----------------------------------------------------------
# # Utility Functions
# # ----------------------------------------------------------

# def clean_text(text):
#     text = re.sub(r'\s+', ' ', text.replace("\r", " ").replace("\n", " "))
#     return text.strip()

# STOPWORDS = {
#     'the','a','an','and','or','in','on','at','to','for','of','is','are','was','were',
#     'it','this','that','these','those','with','as','by','from','be','been','has','have',
#     'had','but','not','they','their','them','its','he','she','we','you','i','my','me','our',
#     'your','his','her','which','who','what','when','where','why','how','can','could','will',
#     'would','should','may','might','also','such','into','about','over','under','between'
# }

# def generate_heading_from_summary(summary, n_words=3):
#     words = [w for w in re.sub(r'[^\w\s]', '', summary.lower()).split() if w not in STOPWORDS]
#     freq = {}
#     for w in words:
#         freq[w] = freq.get(w, 0) + 1
#     sorted_words = sorted(freq, key=lambda x: (-freq[x], -len(x), x))
#     return " ".join(sorted_words[:n_words]).title()

# def read_txt(file_bytes):
#     try:
#         return file_bytes.decode('utf-8')
#     except:
#         try:
#             return file_bytes.decode('latin-1')
#         except:
#             return ""

# def read_pdf(file_bytes):
#     if PdfReader is None: return ""
#     try:
#         reader = PdfReader(BytesIO(file_bytes))
#         return " ".join([p.extract_text() or "" for p in reader.pages])
#     except:
#         return ""

# def chunk_text(text, max_chars=3500):
#     if len(text) <= max_chars:
#         return [text]
#     parts, cur = [], ""
#     for s in re.split(r'(?<=[.!?])\s+', text):
#         if len(cur) + len(s) <= max_chars:
#             cur += " " + s
#         else:
#             parts.append(cur.strip())
#             cur = s
#     parts.append(cur.strip())
#     return parts

# def rouge1_f1(hyp, ref):
#     h = re.findall(r'\w+', hyp.lower())
#     r = re.findall(r'\w+', ref.lower())
#     if not h or not r: return 0.0
#     from collections import Counter
#     overlap = sum((Counter(h) & Counter(r)).values())
#     prec = overlap / len(h)
#     rec = overlap / len(r)
#     return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

# def cosine_sim(a, b):
#     if not SKL_AVAILABLE: return 0.0
#     vect = TfidfVectorizer().fit_transform([a, b])
#     val = cosine_similarity(vect[0:1], vect[1:2])[0][0]
#     return float(val)


# # ----------------------------------------------------------
# # Model Loading
# # ----------------------------------------------------------

# @st.cache_resource(show_spinner=False)
# def load_pipeline(model_name):
#     device = 0 if torch.cuda.is_available() else -1
#     tok = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return pipeline("summarization", model=model, tokenizer=tok, device=device)


# # ----------------------------------------------------------
# # Page UI Layout + Custom Theme
# # ----------------------------------------------------------

# st.set_page_config(page_title="Multi-Model Summarizer", layout="wide")

# st.markdown("""
# <style>
# body {background-color:white;}
# .nav-button {
#     background-color: #2563eb; color:white; padding:10px 15px; border-radius:8px;
#     border:none; font-weight:600; cursor:pointer; transition:0.2s; margin-right:8px;
# }
# .nav-button:hover {background-color:#1e40af; transform:scale(1.03);}
# .card {
#     background:white; padding:16px; border-radius:10px;
#     border:1px solid #e2e8f0; box-shadow:0 4px 14px rgba(0,0,0,0.04);
#     margin-bottom:12px;
# }
# .model-label {font-weight:700; color:#1e293b; margin-bottom:6px}
# </style>
# """, unsafe_allow_html=True)

# # Beautiful Table Global CSS
# st.markdown("""
# <style>
# .dataframe thead th {
#     background-color:#2563eb !important;
#     color:white !important;
#     font-weight:700 !important;
#     padding:10px !important;
#     text-align:center !important;
# }
# .dataframe tbody td {
#     border:1px solid #e5e7eb !important;
#     padding:8px 10px !important;
#     text-align:center !important;
#     color:#111827 !important;
# }
# .dataframe tbody tr:hover {background-color:#eef2ff !important;}
# </style>
# """, unsafe_allow_html=True)

# st.markdown("### Choose the mode ")

# col1, col2 = st.columns([1,3])
# # with col1:
# #     if st.button("✨ Q&A Mode"):
# #         st.switch_page("pages/applan")
# with col1:
#     if st.button("✨ Q&A Mode"):
#         #st.switch_page("Applan")  # <-- page title, not path
#         st.switch_page("pages/Applan.py")
    
    





# with col2:
#     st.caption("Currently viewing: Multi-Model Summarizer")


# st.title("Multi-Model Summarizer")
# st.write("Generate and compare AI summaries using Pegasus, BART, T5 and DistilBART models.")


# # ----------------------------------------------------------
# # Input + Settings layout
# # ----------------------------------------------------------

# left, right = st.columns([3,1])

# with left:
#     text_input = st.text_area("Enter text", height=300, placeholder="Paste or type text here…")
#     upload = st.file_uploader("Or upload PDF/TXT", type=['pdf','txt'])
#     input_text = clean_text(text_input or "")

# with right:
#     length_choice = st.selectbox("Summary Length", ["Concise","Medium","Detailed"])
#     run = st.button("Generate & Compare", use_container_width=True)
#     st.markdown("---")
#     # st.write("Models included:")
#     # st.caption("Pegasus, BART, T5, DistilBART")
#     st.markdown("""
# <div style='background-color:#f8fafc;padding:12px 18px;border-radius:10px;border:1px solid #e5e7eb;'>

#  <div style='text-align:center; font-weight:bold; font-size:24px; margin-bottom:15px;'>
# 🧠 Models Included
# </div>               
#  <b>Pegasus (XSum):</b> Great for concise, factual summaries.<br>
# <b>BART (CNN):</b> Produces fluent and balanced summaries.<br>
# <b>T5 (Small):</b> Fast, lightweight summarizer for shorter inputs.<br>
#  <b>DistilBART:</b> Efficient and optimized version of BART for speed.
# </div>
# """, unsafe_allow_html=True)



# if upload:
#     file_bytes = upload.read()
#     input_text = read_pdf(file_bytes) if upload.type == "application/pdf" else read_txt(file_bytes)


# # ----------------------------------------------------------
# # Model processing
# # ----------------------------------------------------------

# MODEL_MAP = {
#     "Pegasus (xsum)":"google/pegasus-xsum",
#     "BART (cnn)":"facebook/bart-large-cnn",
#     "T5 (small)":"t5-small",
#     "DistilBART":"sshleifer/distilbart-cnn-12-6"
# }

# if run:
#     if len(input_text) < 30:
#         st.error("Input too short. Provide at least 30 characters.")
#         st.stop()

#     min_len, max_len = (12,60) if length_choice=="Concise" else (30,130) if length_choice=="Medium" else (60,260)
#     parts = chunk_text(input_text)

#     with st.spinner("Loading models…"):
#         pipes = {name: load_pipeline(m) for name,m in MODEL_MAP.items()}

#     results = []
#     for name,pipe in pipes.items():
#         all_sums = []
#         for p in parts:
#             try:
#                 out = pipe(p, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
#                 all_sums.append(out.strip())
#             except:
#                 all_sums.append("")
#         final = clean_text(" ".join(all_sums))
#         heading = generate_heading_from_summary(final)
#         results.append({"Model":name,"Summary":final,"Heading":heading})


#     # Metrics Calculation
#     orig_len = len(re.findall(r'\w+', input_text))
#     metrics = []
#     for r in results:
#         summ = r["Summary"]
#         summ_len = len(re.findall(r'\w+', summ))
#         compression = 100*(1 - summ_len/max(orig_len,1))
#         rouge = rouge1_f1(summ,input_text)
#         cos = cosine_sim(summ,input_text)
#         metrics.append({
#             "Model":r["Model"],
#             "SummaryWords":summ_len,
#             "Compression%":round(compression,2),
#             "ROUGE1-F1":round(rouge,4),
#             "CosineSim":round(cos,4),
#             "Heading":r["Heading"],
#             "Summary":summ
#         })


#     # ------------------------------------------------------
#     # Output Layout
#     # ------------------------------------------------------

#     left2, right2 = st.columns([2,1])

#     with left2:
#         st.subheader("Generated Summaries")
#         for m in metrics:
#             st.markdown(f'<div class="card"><div class="model-label">{m["Model"]}: {m["Heading"]}</div>{m["Summary"]}</div>', unsafe_allow_html=True)

#     if PANDAS_AVAILABLE:
#         df = pd.DataFrame(metrics).drop(columns=["Summary","Heading"]).set_index("Model")
#         styled = df.style
#         st.subheader("Summary Metrics")
#         st.write(styled.to_html(), unsafe_allow_html=True)


#     # Charts
#     st.subheader("Charts")
#     st.bar_chart({m["Model"]:m["SummaryWords"] for m in metrics})
#     st.bar_chart({m["Model"]:m["ROUGE1-F1"] for m in metrics})
#     st.bar_chart({m["Model"]:m["CosineSim"] for m in metrics})


#     # Download
#     result_json = json.dumps({"input_text":input_text,"metrics":metrics}, indent=2)
#     st.download_button("Download Results (JSON)", result_json, "results.json","application/json")


# st.markdown("<br><p style='text-align:center;color:#666'>Models powered by HuggingFace Transformers</p>", unsafe_allow_html=True)
# app.py
"""
Unified Multi-Model Summarizer (single-file Streamlit app)

Models: google/pegasus-xsum, facebook/bart-large-cnn, t5-small, sshleifer/distilbart-cnn-12-6
Features:
 - Paste text or upload TXT/PDF
 - Chunking for long inputs
 - Generate summaries from 4 models
 - Compute word count, compression, ROUGE1-F1, cosine similarity
 - Table + Bar Charts
 - Download JSON results
"""

import streamlit as st
from io import BytesIO
import re, math, json
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------------- Page Config MUST be first ----------------------
st.set_page_config(page_title="Multi-Model Summarizer", layout="wide", initial_sidebar_state="collapsed")

# ---------------------- Optional Imports ----------------------
try:
    from PyPDF2 import PdfReader
except:
    PdfReader = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKL_AVAILABLE = True
except:
    SKL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except:
    PANDAS_AVAILABLE = False

# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.replace("\r", " ").replace("\n", " "))
    return text.strip()

STOPWORDS = {
    'the','a','an','and','or','in','on','at','to','for','of','is','are','was','were',
    'it','this','that','these','those','with','as','by','from','be','been','has','have',
    'had','but','not','they','their','them','its','he','she','we','you','i','my','me','our',
    'your','his','her','which','who','what','when','where','why','how','can','could','will',
    'would','should','may','might','also','such','into','about','over','under','between'
}

def generate_heading_from_summary(summary, n_words=3):
    words = [w for w in re.sub(r'[^\w\s]', '', summary.lower()).split() if w not in STOPWORDS]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq, key=lambda x: (-freq[x], -len(x), x))
    return " ".join(sorted_words[:n_words]).title()

def read_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8')
    except:
        try:
            return file_bytes.decode('latin-1')
        except:
            return ""

def read_pdf(file_bytes):
    if PdfReader is None: return ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        return " ".join([p.extract_text() or "" for p in reader.pages])
    except:
        return ""

def chunk_text(text, max_chars=3500):
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for s in re.split(r'(?<=[.!?])\s+', text):
        if len(cur) + len(s) <= max_chars:
            cur += " " + s
        else:
            parts.append(cur.strip())
            cur = s
    parts.append(cur.strip())
    return parts

def rouge1_f1(hyp, ref):
    h = re.findall(r'\w+', hyp.lower())
    r = re.findall(r'\w+', ref.lower())
    if not h or not r: return 0.0
    from collections import Counter
    overlap = sum((Counter(h) & Counter(r)).values())
    prec = overlap / len(h)
    rec = overlap / len(r)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def cosine_sim(a, b):
    if not SKL_AVAILABLE: return 0.0
    vect = TfidfVectorizer().fit_transform([a, b])
    val = cosine_similarity(vect[0:1], vect[1:2])[0][0]
    return float(val)

# ----------------------------------------------------------
# Model Loading
# ----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name):
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("summarization", model=model, tokenizer=tok, device=device)

# ----------------------------------------------------------
# Custom CSS
# # ----------------------------------------------------------
st.markdown("""
<style>
body {background-color:white;}
* {font-size:17px !important;}
textarea, select, input, button, label, p, div, span {font-size:17px !important;}
.nav-button {
    background-color: #2563eb; color:white; padding:10px 18px; border-radius:8px;
    border:none; font-weight:600; cursor:pointer; transition:0.2s; margin-right:8px;
}
.nav-button:hover {background-color:#1e40af; transform:scale(1.03);}
.card {
    background:white; padding:18px; border-radius:10px;
    border:1px solid #cbd5e1; box-shadow:0 4px 14px rgba(0,0,0,0.05);
    margin-bottom:14px; font-size:17px;
}
.model-label {font-weight:700; color:#1e293b; margin-bottom:6px}

.dataframe thead th {
    background-color:#2563eb !important;
    color:white !important;
    font-weight:700 !important;
    padding:10px !important;
    text-align:center !important;
}
.dataframe tbody td {
    border:1px solid #cbd5e1 !important;
    padding:10px !important;
    text-align:center !important;
    color:#111827 !important;
}
.dataframe {border:2px solid #94a3b8 !important; border-radius:6px;}
.dataframe tbody tr:hover {background-color:#eef2ff !important;}
</style>
""", unsafe_allow_html=True)





# ----------------------------------------------------------
# Top Navigation (Q&A Mode on right)
# ----------------------------------------------------------
top_col1, top_col2 = st.columns([3,1])

# with top_col1:
#     st.markdown("""
#     <div style='font-size:96px !important; font-weight:bold !important; color:#1e293b !important; line-height:1.1 !important; margin:0 !important;'>
#         🧠 Multi-Model Summarizer
#     </div>
#     """, unsafe_allow_html=True)
st.markdown("""
<style>
.big-title {
    font-size:65px !important; 
    font-weight:900 !important; 
    color:white !important; 
    background-color:#2563eb !important;
    padding:18px 28px !important;
    border-radius:14px !important;
    text-align:center !important;
    margin-bottom:15px !important;
}

.sub-title {
    font-size:42px !important; 
    font-weight:800 !important; 
    color:#2563eb !important;
    text-align:center !important;
    padding:10px !important;
    margin-bottom:10px !important;
}
</style>
""", unsafe_allow_html=True)

# APPLY STYLES
with top_col1:
    st.markdown("<div class='big-title'>🧠 Multi-Model Summarizer</div>", unsafe_allow_html=True)

with top_col2:
    st.markdown("<div class='sub-title'>📚 Research Paper Analyser</div>", unsafe_allow_html=True)
    if st.button("✨ Q&A Mode", use_container_width=True):
        st.switch_page("pages/Applan.py")





  





st.write("Generate and compare AI summaries using Pegasus, BART, T5 and DistilBART models.")

# ----------------------------------------------------------
# Input + Settings
# ----------------------------------------------------------
left, right = st.columns([3,1])
# with left:
#     text_input = st.text_area("📝 Enter Text", height=300, placeholder="Paste or type text here…")
#     upload = st.file_uploader("📂 Or upload PDF/TXT", type=['pdf','txt'])
#     input_text = clean_text(text_input or "")
with left:
    # --- Initialize session flags ---
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    if "clear_trigger" not in st.session_state:
        st.session_state.clear_trigger = False

    # --- Handle clear flag BEFORE widget loads ---
    if st.session_state.clear_trigger:
        st.session_state.text_input = ""
        st.session_state.clear_trigger = False

    # --- Label and Clear button in same row ---
    label_col, btn_col = st.columns([4, 1])
    with label_col:
        st.markdown("### 📝 Enter Text")
    with btn_col:
        if st.button(" Clear", use_container_width=True):
            st.session_state.clear_trigger = True
            st.rerun()

    # --- Text area ---
    text_input = st.text_area(
        label="",
        key="text_input",
        height=300,
        placeholder="Paste or type text here…"
    )

    # --- File upload below ---
    upload = st.file_uploader("📂 Or upload PDF/TXT", type=['pdf', 'txt'])

    # --- Clean processed input ---
    input_text = clean_text(st.session_state.text_input or "")


with right:
    length_choice = st.selectbox("📏 Summary Length", ["Concise","Medium","Detailed"])
    run = st.button("🚀 Generate & Compare", use_container_width=True)
    st.markdown("---")
    st.markdown("""
<div style='background-color:#f8fafc;padding:12px 18px;border-radius:10px;border:1px solid #e5e7eb;'>
 <div style='text-align:center; font-weight:bold; font-size:24px; margin-bottom:15px;'>Models Included</div>               
<b>Pegasus (XSum):</b> Great for concise, factual summaries.<br>
<b>BART (CNN):</b> Produces fluent and balanced summaries.<br>
<b>T5 (Small):</b> Fast, lightweight summarizer for shorter inputs.<br>
<b>DistilBART:</b> Efficient and optimized version of BART for speed.
</div>
""", unsafe_allow_html=True)




if upload:
    file_bytes = upload.read()
    input_text = read_pdf(file_bytes) if upload.type == "application/pdf" else read_txt(file_bytes)

# ----------------------------------------------------------
# Model processing
# ----------------------------------------------------------
MODEL_MAP = {
    "Pegasus (xsum)":"google/pegasus-xsum",
    "BART (cnn)":"facebook/bart-large-cnn",
    "T5 (small)":"t5-small",
    "DistilBART":"sshleifer/distilbart-cnn-12-6"
}

if run:
    if len(input_text) < 30:
        st.error("Input too short. Provide at least 30 characters.")
        st.stop()

    min_len, max_len = (12,60) if length_choice=="Concise" else (30,130) if length_choice=="Medium" else (60,260)
    parts = chunk_text(input_text)

    with st.spinner("Loading models…"):
        pipes = {name: load_pipeline(m) for name,m in MODEL_MAP.items()}

    results = []
    for name,pipe in pipes.items():
        all_sums = []
        for p in parts:
            try:
                #out = pipe(p, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
                out = pipe(p, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)[0]["summary_text"]

                all_sums.append(out.strip())
            except:
                all_sums.append("")
        final = clean_text(" ".join(all_sums))
        heading = generate_heading_from_summary(final)
        results.append({"Model":name,"Summary":final,"Heading":heading})

    # Metrics
    orig_len = len(re.findall(r'\w+', input_text))
    metrics = []
    for r in results:
        summ = r["Summary"]
        summ_len = len(re.findall(r'\w+', summ))
        compression = 100*(1 - summ_len/max(orig_len,1))
        rouge = rouge1_f1(summ,input_text)
        cos = cosine_sim(summ,input_text)
        metrics.append({
            "Model":r["Model"],
            "SummaryWords":summ_len,
            "Compression%":round(compression,2),
            "ROUGE1-F1":round(rouge,4),
            "CosineSim":round(cos,4),
            "Heading":r["Heading"],
            "Summary":summ
        })

    # ----------------------- Output -----------------------
    left2, right2 = st.columns([2,1])
    with left2:
        st.subheader("Generated Summaries")
        for m in metrics:
            st.markdown(f'<div class="card"><div class="model-label">{m["Model"]}: {m["Heading"]}</div>{m["Summary"]}</div>', unsafe_allow_html=True)

    if PANDAS_AVAILABLE:
        df = pd.DataFrame(metrics).drop(columns=["Summary","Heading"]).set_index("Model")
        st.subheader("Summary Metrics")
        st.write(df.to_html(escape=False), unsafe_allow_html=True)

    # ----------------------- Charts -----------------------
    st.subheader("📊 Summary Word Count per Model")
    st.bar_chart({m["Model"]:m["SummaryWords"] for m in metrics})

    st.subheader("📈 ROUGE1-F1 Score per Model")
    st.bar_chart({m["Model"]:m["ROUGE1-F1"] for m in metrics})

    st.subheader("🔗 Cosine Similarity per Model")
    st.bar_chart({m["Model"]:m["CosineSim"] for m in metrics})

    # ----------------------- Download -----------------------
    result_json = json.dumps({"input_text":input_text,"metrics":metrics}, indent=2)
    st.download_button("Download Results (JSON)", result_json, "results.json","application/json")

st.markdown("<br><p style='text-align:center;color:#666'>Models powered by HuggingFace Transformers</p>", unsafe_allow_html=True)
