# DocuMind – Intelligent PDF Question Answering System

An AI-powered PDF Question Answering System built using **Retrieval-Augmented Generation (RAG)**. DocuMind enables users to interact with PDF documents, ask questions, and receive intelligent, context-aware answers.

## ✨ Features

- **PDF Question Answering:** Get context-aware answers from uploaded documents.
- **Hybrid Retrieval:** Combines semantic search and keyword-based retrieval.
- **AI Summarization:** Generate concise summaries of PDF documents.
- **Smart Summarization:** Extract important information from lengthy documents.
- **Quiz Generation:** Generate quizzes based on document content.
- **Multiple AI Modes:** Auto, AI-only, and PDF-only modes.
- **Source Citations:** Provides references supporting generated answers, with separate PDF and web sources.
- **Live Web Search:** Uses Gemini Google Search grounding for current information in Auto and Only AI modes.
- **Current Date & Time:** Provides the current date and time using the Asia/Kolkata timezone in Auto and Only AI modes.
- **Modern UI:** Interactive interface with dark and light themes.

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI
- **AI Model:** Google Gemini
- **Vector Database:** FAISS
- **Embeddings:** Sentence Transformers
- **Retrieval:** Hybrid Search, BM25
- **PDF Processing:** PyMuPDF, OCR

---

*An AI-powered document intelligence system built with Retrieval-Augmented Generation (RAG).*

### Current date and time timezone

DocuMind uses `Asia/Kolkata` by default for server-side current date/time answers. To override it, set the optional `DOCUMIND_TIMEZONE` environment variable to a valid IANA timezone name (for example, `America/New_York`). If the value is invalid or timezone data is unavailable, the application falls back to `Asia/Kolkata`.
