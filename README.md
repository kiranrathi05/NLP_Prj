# Research Paper Analyser

## Description

Research Paper Analyser is an intelligent NLP-based system designed to
automatically **summarize academic research papers** and **answer user
queries** based on their content.\
It leverages **transformer models (Pegasus, BART, T5)** and **LLMs
(Llama 3.1 via RAG)** to provide accurate summaries, contextual answers,
and an interactive interface for research assistance.

## Installation

1.  Clone the repository

    ``` bash
    git clone <https://github.com/kiranrathi05/NLP_Prj>
    ```

2.  Install dependencies

    ``` bash
    pip install -r requirements.txt
    ```

3.  Run the application

    ``` bash
    streamlit run app.py
    ```

## Usage

Visit the local URL provided by Streamlit (typically
`http://localhost:8501`) to:

-   Upload PDF/DOCX research papers\
-   Generate automatic section-wise summaries\
-   Ask interactive questions based on the uploaded paper\
-   View evaluation metrics and model comparisons

## Introduction

### 1.1 Background

The rapid growth of academic research publications makes manual reading,
analysis, and extraction of insights highly challenging. Research papers
often contain dense technical content spanning multiple pages, requiring
significant time and cognitive effort.

With the rise of **transformer-based NLP models** like BERT, Pegasus,
T5, and BART, it is now possible to generate high-quality summaries and
extract contextual information automatically.

This project uses: - **Transformer summarizers** (Pegasus, BART, T5)\
- **A RAG-based QA system** using Sentence-BERT + FAISS + Llama 3.1

to build a system that extracts, condenses, and interprets academic
documents with high accuracy.

### 1.2 Motivation

Researchers and students struggle to manage the massive volume of
scholarly publications. Traditional search engines retrieve documents
but cannot summarize or answer specific questions inside those
documents.

This project bridges the gap by: 1. Generating accurate and concise
summaries\
2. Providing contextual answers to user queries\
3. Reducing time and cognitive load in the research process\
4. Enabling fast literature review through automation

### 1.3 Problem Statement

Design and implement an NLP system that can automatically summarize
academic papers and answer user queries using transformer models and
LLMs.

### 1.4 Aim

To build a **hybrid Research Paper Summarization + QA framework** that
integrates multiple summarization pipelines with an interactive
question-answering interface.

### 1.5 Scope

The project includes:

-   Extraction & preprocessing of PDF/DOCX papers\
-   Section-wise segmentation\
-   Summarization using Pegasus, BART, and T5\
-   Evaluation using ROUGE, BLEU, BERTScore, Flesch Reading Ease\
-   A RAG-based QA system using Sentence-BERT, FAISS, and Llama 3.1\
-   A Streamlit-based interactive web interface

## Objectives

1.  Extract and preprocess text from uploaded research papers\
2.  Identify sections: Abstract, Introduction, Methodology, Results,
    Conclusion\
3.  Generate abstractive summaries using Pegasus, BART, and T5\
4.  Evaluate summaries using ROUGE, BLEU, BERTScore, and readability
    metrics\
5.  Implement Llama 3.1 Q&A using Groq API\
6.  Build a Streamlit-based user interface

## Methodology & Implementation

### 4.1 System Architecture

1.  Document Upload & Extraction\
2.  Preprocessing & Tokenization\
3.  Summarization using Pegasus, BART, T5\
4.  Evaluation Metrics (ROUGE, BLEU, BERTScore, etc.)\
5.  Question Answering using Sentence-BERT + FAISS + Llama 3.1\
6.  Streamlit UI

### 4.2 Data Preprocessing

-   Tokenization using NLTK\
-   Text cleaning\
-   Chunking ≤600 words\
-   Embedding via MiniLM-L6-v2\
-   Normalization

### 4.3 Summarization Pipelines

Models used: - Pegasus\
- BART\
- T5

### 4.4 Evaluation Metrics

-   ROUGE\
-   BLEU\
-   BERTScore\
-   Flesch Reading Ease\
-   Cosine Similarity

### 4.5 Question Answering Component

RAG pipeline: 1. Embedding generation\
2. FAISS retrieval\
3. Llama 3.1 answer generation\
4. Ranking & filtering

### 4.6 Interface

Built using Streamlit with: - File upload\
- Summaries\
- Chat-style QA\
- Clean UI

### 4.7 Model Details

  Model       Type               Task            Dataset
  ----------- ------------------ --------------- ----------------
  Pegasus     Encoder--Decoder   Summarization   XSum
  BART        Seq2Seq            Summarization   CNN/DailyMail
  T5          Text-to-Text       Summarization   Multi-domain
  Llama 3.1   LLM                QA              General corpus

## Software & Hardware Requirements

### Software

-   Python 3.10\
-   Transformers\
-   Streamlit\
-   LangChain\
-   FAISS\
-   Sentence-Transformers\
-   PyPDF2\
-   Groq API

### Hardware

-   Intel i5+\
-   8GB RAM\
-   GPU recommended\
-   10GB storage

## Results & Discussion

### 6.1 Quantitative Evaluation

  ----------------------------------------------------------------------------
  Model     ROUGE-1   ROUGE-L   BLEU   BERTScore   Readability   Cosine Sim
  --------- --------- --------- ------ ----------- ------------- -------------
  Pegasus   0.51      0.45      0.32   0.88        61.4          0.83

  BART      0.54      0.47      0.35   0.90        63.2          0.86

  T5        0.48      0.41      0.28   0.85        65.0          0.80
  ----------------------------------------------------------------------------

### 6.2 QA Performance

The QA system achieved **88--92% factual accuracy** with minimal
hallucination.

### 6.3 Comparative Analysis

  Aspect            Pegasus    BART        T5
  ----------------- ---------- ----------- ----------
  Fluency           High       Very High   Medium
  Informativeness   Moderate   High        Moderate
  Computation       Medium     High        Low

## Future Scope

-   Domain-specific fine-tuning\
-   Multilingual support\
-   Citation-aware summaries\
-   Cloud deployment\
-   Multimodal understanding

## Contributing

Pull requests are welcome.

## License

MIT License


