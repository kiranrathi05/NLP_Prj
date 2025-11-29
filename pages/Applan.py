# import streamlit as st
# import tempfile


# st.set_page_config(page_title="Applan")  # <-- this is the page title



# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from groq import Groq

# # ------------------- SETTINGS ------------------- #

# LLAMA_MODEL = "llama-3.1-8b-instant"
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 200

# # ------------------- PAGE CONFIG ------------------- #
# # st.set_page_config(page_title="Research Q&A", layout="centered")

# # ------------------- THEME & STYLES (matches Summarizer) ------------------- #
# st.markdown("""
# <style>
# body {
#     background-color: #ffffff;
# }
# .back-btn, .submit-btn {
#     background-color:#2563eb;
#     color:white;
#     border-radius:10px;
#     padding:10px 25px;
#     font-weight:600;
#     border:none;
#     cursor:pointer;
#     transition:0.2s;
# }
# .back-btn:hover, .submit-btn:hover {
#     background-color:#1e40af;
#     transform:scale(1.03);
# }
# .container-box {
#     background:white;
#     border-radius:14px;
#     padding:28px 25px;
#     width:80%;
#     margin:auto;
#     margin-top:25px;
#     border:1px solid #e5e7eb;
#     box-shadow:0 4px 14px rgba(0,0,0,0.04);
# }
# .upload-style > div > div {
#     border:2px dashed #dbeafe;
#     background-color:#f0f7ff;
#     border-radius:12px;
# }
# .stTextInput>div>div>input {
#     border-radius:10px;
#     border:1px solid #d1d5db !important;
# }
# .answer-box {
#     margin-top:18px;
#     border-left:4px solid #2563eb;
#     padding:12px;
#     font-size:16px;
#     border-radius:4px;
#     background-color:#f9fafb;
# }
# </style>
# """, unsafe_allow_html=True)

# # ------------------- NAVIGATION ------------------- #


# if st.button("⬅ Back to Summarizer", key="back_btn", help="Return to main app"):
#     st.switch_page("app.py")  # <- Use the page_title from app.py




# # ------------------- TITLE ------------------- #
# st.markdown("<h2 style='text-align:center;color:#1e293b;font-weight:700;'>Research Paper Q&A</h2>", unsafe_allow_html=True)
# st.caption("Upload a research paper and ask a question. Powered by Llama + RAG.")

# # ------------------- Interface BOX ------------------- #
# with st.container():
#     #st.markdown('<div class="container-box">', unsafe_allow_html=True)

#     uploaded_file = st.file_uploader("📤 Upload Research PDF", type=["pdf"], key="pdf-file", help="Only text-based PDFs supported.")

#     question = st.text_input("Ask a question based on the paper", placeholder="Example: What methodology is used in this research?")

#     submit = st.button("Get Answer", key="generate_btn")

#     if submit:
#         if uploaded_file and question:
#             with st.spinner("Analyzing document..."):
#                 try:
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#                         temp_pdf.write(uploaded_file.read())
#                         pdf_path = temp_pdf.name

#                     loader = PyPDFLoader(pdf_path)
#                     pages = loader.load_and_split()
#                     splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#                     docs = splitter.split_documents(pages)

#                     # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#                     # vector_store = FAISS.from_documents(docs, embeddings)
#                     # retriever = vector_store.as_retriever()

#                     # # retrieved = retriever.get_relevant_documents(question)
#                     # retrieved = retriever.retrieve(question)


#                     # context = "\n\n".join([d.page_content for d in retrieved])
#                     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#                     vector_store = FAISS.from_documents(docs, embeddings)

# # Instead of as_retriever + get_relevant_documents
#                     retrieved = vector_store.similarity_search(question, k=4)

#                     context = "\n\n".join([d.page_content for d in retrieved])


#                     client = Groq(api_key=GROQ_API_KEY)
#                     prompt = f"""
#                     Use the context to answer the question concisely.
#                     Do not hallucinate or assume unknown facts.

#                     Context:
#                     {context}

#                     Question:
#                     {question}

#                     Answer:
#                     """

#                     completion = client.chat.completions.create(
#                         model=LLAMA_MODEL,
#                         messages=[{"role": "user", "content": prompt}],
#                         temperature=0.2,
#                         max_completion_tokens=512,
#                         top_p=1,
#                         stream=True
#                     )

#                     answer = ""
#                     for chunk in completion:
#                         delta = chunk.choices[0].delta.content or ""
#                         answer += delta

#                     st.markdown('<div class="answer-box">✅ <strong>Answer:</strong><br>' + answer + '</div>', unsafe_allow_html=True)

#                 except Exception as e:
#                     st.error(f"❌ Error: {str(e)}")
#         else:
#             st.warning("⚠️ Please upload a PDF and enter a valid question.")

#     st.markdown("</div>", unsafe_allow_html=True)
import streamlit as st
import tempfile

st.set_page_config(page_title="Applan", layout="centered")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq

# Keys and constants

# LLAMA_MODEL = "llama-3.1-8b-instant"
# CHUNK_SIZE = 800
# Keys and constants
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 800

CHUNK_OVERLAP = 200

# GLOBAL STYLE
st.markdown("""
<style>
/* Background */
body {
    background-color: #ffffff;
}

/* Page Container */
.container-box {
    background: #ffffff;
    border-radius: 18px;
    padding: 30px 35px;
    width: 70%;
    margin: auto;
    margin-top: 25px;
    border: 1px solid #e6e8f0;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton > button {
    background-color: #1f4eff !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border: none !important;
    transition: 0.2s ease-in-out;
}
.stButton > button:hover {
    background-color: #163bcc !important;
    transform: scale(1.05);
}

/* Upload */
.upload-box > div > div {
    border: 2px dashed #cdd8ff !important;
    background-color: #f7f9ff !important;
    border-radius: 12px !important;
}

/* Answer Box */
.answer-box {
    margin-top: 18px;
    border-left: 4px solid #1f4eff;
    padding: 14px;
    font-size: 16px;
    border-radius: 6px;
    background-color: #f8faff;
}

/* Title */
.title-text {
    text-align: center;
    font-weight: 800;
    color: #19223c;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# Back Button
if st.button("⬅ Back to Summarizer", key="back_btn"):
    st.switch_page("app.py")

# Title
st.markdown("<h2 class='title-text'>Research Paper Q&A</h2>", unsafe_allow_html=True)
st.caption("Upload a research paper and ask a question. Powered by Llama + RAG.")

# UI Box
with st.container():
    

    uploaded_file = st.file_uploader(
        "📤 Upload Research PDF",
        type=["pdf"],
        key="pdf-file",
        help="Only text-based PDFs supported."
    )

    question = st.text_input(
        "Ask a question based on the paper:",
        placeholder="What methodology is used in this research?"
    )

    submit = st.button("Get Answer")

    if submit:
        if uploaded_file and question:
            with st.spinner("Analyzing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(uploaded_file.read())
                        pdf_path = temp_pdf.name

                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load_and_split()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    docs = splitter.split_documents(pages)

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vector_store = FAISS.from_documents(docs, embeddings)
                    retrieved = vector_store.similarity_search(question, k=4)

                    context = "\n\n".join([d.page_content for d in retrieved])

                    client = Groq(api_key=GROQ_API_KEY)

                    prompt = f"""
                    Use the context to answer the question concisely.
                    Do not hallucinate or assume unknown facts.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """

                    completion = client.chat.completions.create(
                        model=LLAMA_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_completion_tokens=512,
                        top_p=1,
                        stream=True
                    )

                    answer = ""
                    for chunk in completion:
                        delta = chunk.choices[0].delta.content or ""
                        answer += delta

                    st.markdown(f'<div class="answer-box">✅ <strong>Answer:</strong><br>{answer}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please upload a PDF and enter a valid question.")

    st.markdown("</div>", unsafe_allow_html=True)
