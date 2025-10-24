"""
Main server module for the Document Q&A and Summarization ChatGPT plugin.

This FastAPI application exposes two primary endpoints:

1. /summarize – Accepts a URL, uploaded file (PDF or text) or raw text and
   returns a short summary. The number of sentences in the summary can be
   configured via the `sentences` form field.
2. /answer – Accepts a question together with a document and extracts
   relevant sentences from the document to form an answer. The number of
   sentences returned can be controlled with `top_sentences`.

To run locally:

    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 3333
"""

from typing import Optional
import re
from io import BytesIO

import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse

# Optional deps (graceful fallback if missing)
try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:
    PdfReader = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except ImportError:
    TfidfVectorizer = None


app = FastAPI(
    title="Document Q&A and Summarization API",
    description=(
        "An API for summarising documents and answering questions about them. "
        "Documents may be provided via URL, file upload (PDF or plain text) or raw text."
    ),
    version="1.0.0",
)

# Allow the ChatGPT builder (and other origins) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Helpers
# ----------------------------
def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter using punctuation heuristics."""
    cleaned = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if s.strip()]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PdfReader:
        raise RuntimeError("PyPDF2 is required to extract text from PDF files.")
    reader = PdfReader(BytesIO(file_bytes))
    out = []
    for page in reader.pages:
        out.append((page.extract_text() or "").strip())
    return "\n".join(out)


def extract_text_from_url(url: str, timeout: int = 15) -> str:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as err:
        raise RuntimeError(f"Failed to fetch {url}: {err}")
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup is required to extract text from HTML pages.")
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def summarise(text: str, num_sentences: int = 5) -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # frequency-based scoring
    words = re.findall(r"\w+", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    scores: list[float] = []
    for sent in sentences:
        tokens = re.findall(r"\w+", sent.lower())
        if not tokens:
            scores.append(0.0)
        else:
            scores.append(sum(freq.get(t, 0) for t in tokens) / len(tokens))

    top_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:num_sentences]
    top_idx.sort()
    return " ".join(sentences[i] for i in top_idx)


def answer_question(text: str, question: str, top_n: int = 3) -> str:
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn is required to compute TF-IDF for Q&A.")
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    docs = sentences + [question]
    try:
        tfidf = TfidfVectorizer().fit_transform(docs).toarray()
    except Exception as err:
        raise RuntimeError(f"Failed to vectorise text: {err}")
    qvec = tfidf[-1]
    sents = tfidf[:-1]
    sims = sents.dot(qvec)
    top_idx = np.argsort(sims)[::-1][:top_n]
    return " ".join(sentences[i] for i in top_idx)


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "service": "docgpt-plugin"}


@app.post("/summarize")
async def summarize_endpoint(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    sentences: int = Form(5),
) -> JSONResponse:
    # Ensure exactly one source
    provided = sum(bool(x) for x in (url, file, text))
    if provided != 1:
        return JSONResponse(status_code=400, content={"error": "Provide exactly one of url, file or text."})
    try:
        if text:
            content = text
        elif file is not None:
            data = await file.read()
            name = (file.filename or "").lower()
            if name.endswith(".pdf") or (file.content_type and "pdf" in file.content_type.lower()):
                content = extract_text_from_pdf(data)
            else:
                content = data.decode("utf-8", errors="ignore")
        else:  # url
            content = extract_text_from_url(url)  # type: ignore[arg-type]
        summary = summarise(content, num_sentences=max(1, sentences))
        return JSONResponse(content={"summary": summary})
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": str(err)})


@app.post("/answer")
async def answer_endpoint(
    question: str = Form(...),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    top_sentences: int = Form(3),
) -> JSONResponse:
    provided = sum(bool(x) for x in (url, file, text))
    if provided != 1:
        return JSONResponse(status_code=400, content={"error": "Provide exactly one of url, file or text along with the question."})
    try:
        if text:
            content = text
        elif file is not None:
            data = await file.read()
            name = (file.filename or "").lower()
            if name.endswith(".pdf") or (file.content_type and "pdf" in file.content_type.lower()):
                content = extract_text_from_pdf(data)
            else:
                content = data.decode("utf-8", errors="ignore")
        else:
            content = extract_text_from_url(url)  # type: ignore[arg-type]
        answer_text = answer_question(content, question, top_n=max(1, top_sentences))
        return JSONResponse(content={"answer": answer_text})
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": str(err)})


@app.get("/openapi.yaml", include_in_schema=False)
def get_openapi_spec():
    """Serve the OpenAPI spec as raw YAML so GPT Actions can import it."""
    try:
        with open("openapi.yaml", "r", encoding="utf-8") as f:
            return PlainTextResponse(f.read(), media_type="text/yaml")
    except FileNotFoundError:
        return PlainTextResponse("openapi spec not found", status_code=404)


@app.get("/logo.png", include_in_schema=False)
def get_logo():
    """Serve the plugin logo image."""
    try:
        return FileResponse("logo.png", media_type="image/png")
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": f"Failed to load logo: {err}"})
