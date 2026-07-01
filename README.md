# RAG Study Assistant

Ask questions about your own notes and get answers pulled directly from them — not from the model's general knowledge.

**Live demo:** https://huggingface.co/spaces/rahul-63/rag-study-assistant

## Problem

Searching through long PDFs or scraped web notes to find one answer is slow. This tool lets you upload material once and query it in plain language.

## How it works

1. Upload notes — PDF or web-scraped HTML/text
2. Text is chunked and embedded with `sentence-transformers`
3. Chunks stored in ChromaDB (local vector DB)
4. On a question, top matching chunks are retrieved and passed to Groq's LLM as context
5. Answer is generated only from retrieved chunks — not the model's open-ended knowledge

## Tech stack

- Streamlit — UI
- ChromaDB — vector storage
- sentence-transformers — embeddings
- Groq API — LLM inference
- pdfplumber, BeautifulSoup — document parsing

## Run locally

```bash
git clone https://github.com/Rahulgupta-63/rag-study-assistant.git
cd rag-study-assistant
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

## Limitations

- Retrieval quality depends on chunk size — long-context questions across multiple sections can lose relevant chunks
- No persistent storage across sessions in the current deployment — notes are re-uploaded each time
- Single-document context only, no cross-document reasoning yet
