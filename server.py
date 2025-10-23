"""
Main server module for the Document Q&A and Summarization ChatGPT plugin.

This FastAPI application exposes two primary endpoints:

1. ``/summarize`` – Accepts a URL, uploaded file (PDF or text) or raw text and
   returns a short summary. The number of sentences in the summary can be
   configured via the ``sentences`` form field.
2. ``/answer`` – Accepts a question together with a document and extracts
   relevant sentences from the document to form an answer. The number of
   sentences returned can be controlled with ``top_sentences``.

The plugin is designed to work with ChatGPT's plugin interface. It relies on
basic natural language processing techniques rather than heavy machine learning
models to ensure quick responses and minimal dependencies. Summarisation uses
a simple frequency–based method, and question answering uses TF‑IDF to rank
sentences by relevance to the provided question.

To run this server locally for development:

.. code-block:: bash

   pip install -r requirements.txt
   uvicorn server:app --host 0.0.0.0 --port 3333

The OpenAPI specification for this API is available at ``/openapi.yaml`` once
the server is running.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import requests
import re
import numpy as np
from io import BytesIO

try:
    from PyPDF2 import PdfReader  # type: ignore
except ImportError:
    PdfReader = None  # Fallback if PyPDF2 is not installed

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

# Allow the ChatGPT playground (and potentially other origins) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def split_into_sentences(text: str) -> list[str]:
    """Split a block of text into sentences using a simple regular expression.

    This function avoids external dependencies like NLTK by using a heuristic
    that looks for sentence‑ending punctuation followed by whitespace. It is
    language‑agnostic for languages that use periods, question marks or
    exclamation marks to end sentences.

    Args:
        text: The text to split.

    Returns:
        A list of sentence strings.
    """
    # Normalise newlines to spaces
    cleaned = re.sub(r"\s+", " ", text)
    # Split on punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    # Filter out any empty strings
    return [s.strip() for s in sentences if s.strip()]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyPDF2.

    Args:
        file_bytes: The binary content of the PDF file.

    Returns:
        A string containing all extracted text.
    """
    if not PdfReader:
        raise RuntimeError("PyPDF2 is required to extract text from PDF files.")
    reader = PdfReader(BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        # Some pages may not have extractable text
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def extract_text_from_url(url: str, timeout: int = 10) -> str:
    """Fetch a URL and return its visible text content.

    Args:
        url: The URL to fetch.
        timeout: Timeout in seconds for the HTTP request.

    Returns:
        The visible text content of the page.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as err:
        raise RuntimeError(f"Failed to fetch {url}: {err}")
    if BeautifulSoup is None:
        raise RuntimeError(
            "BeautifulSoup is required to extract text from HTML pages."
        )
    soup = BeautifulSoup(response.content, "html.parser")
    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def summarise(text: str, num_sentences: int = 5) -> str:
    """Generate a simple summary by selecting the most informative sentences.

    The algorithm computes a frequency table of words and scores each sentence
    based on the sum of frequencies of its words. The top‑scoring sentences
    (by default five) are selected and returned in their original order.

    Args:
        text: The input document as a string.
        num_sentences: The desired number of sentences in the summary.

    Returns:
        A summary comprised of the most informative sentences.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Build a frequency table for words (case insensitive)
    words = re.findall(r"\w+", text.lower())
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    # Score sentences by summing word frequencies
    scores = []
    for sent in sentences:
        tokens = re.findall(r"\w+", sent.lower())
        if not tokens:
            scores.append(0.0)
            continue
        scores.append(sum(freq.get(t, 0) for t in tokens) / len(tokens))

    # Get indices of top scoring sentences
    top_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:num_sentences]
    # Sort top sentences back to their original order
    top_indices.sort()
    summary_sentences = [sentences[i] for i in top_indices]
    return " ".join(summary_sentences)


def answer_question(text: str, question: str, top_n: int = 3) -> str:
    """Answer a question by returning the most relevant sentences from the document.

    This function tokenises the document into sentences and uses TF‑IDF to
    compute the similarity between each sentence and the question. The top
    ``top_n`` sentences with the highest cosine similarity scores are returned
    concatenated as the answer.

    Args:
        text: The input document as a string.
        question: The question asked by the user.
        top_n: The number of sentences to include in the answer.

    Returns:
        A string containing the most relevant sentences.
    """
    if TfidfVectorizer is None:
        raise RuntimeError(
            "scikit-learn is required to compute TF-IDF similarities for Q&A."
        )
    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    # Build a list of documents consisting of all sentences plus the question
    docs = sentences + [question]
    try:
        tfidf = TfidfVectorizer().fit_transform(docs)
    except Exception as err:
        raise RuntimeError(f"Failed to vectorise text: {err}")
    # The last vector corresponds to the question
    question_vec = tfidf.toarray()[-1]
    sent_vecs = tfidf.toarray()[:-1]
    # Compute cosine similarities
    # Avoid division by zero for zero vectors
    similarities = sent_vecs.dot(question_vec)
    # Get indices of top sentences by similarity
    top_indices = np.argsort(similarities)[::-1][:top_n]
    # Concatenate the selected sentences
    answer = " ".join([sentences[i] for i in top_indices])
    return answer


@app.post("/summarize")
async def summarize_endpoint(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    sentences: int = Form(5),
) -> JSONResponse:
    """Summarise a document from a URL, uploaded file or raw text.

    You must provide exactly one of ``url``, ``file`` or ``text``. If none
    are provided, a 400 error is returned. The number of sentences in the
    summary can be adjusted via the ``sentences`` field (defaults to 5).
    """
    # Validate that exactly one source is provided
    provided = sum(bool(x) for x in [url, file, text])
    if provided != 1:
        return JSONResponse(status_code=400, content={"error": "Provide exactly one of url, file or text."})
    try:
        content: str
        if text:
            content = text
        elif file is not None:
            data = await file.read()
            filename = file.filename or ""
            # Determine file type by extension and content type
            if filename.lower().endswith(".pdf") or (file.content_type and "pdf" in file.content_type):
                content = extract_text_from_pdf(data)
            else:
                content = data.decode("utf-8", errors="ignore")
        elif url:
            content = extract_text_from_url(url)
        else:
            return JSONResponse(status_code=400, content={"error": "No valid input provided."})
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
    """Answer a question by extracting relevant sentences from a document.

    The user must provide a ``question`` along with exactly one of ``url``,
    ``file`` or ``text``. The ``top_sentences`` field controls how many
    sentences to return (default is 3). Responses include the concatenated
    sentences as ``answer``.
    """
    provided = sum(bool(x) for x in [url, file, text])
    if provided != 1:
        return JSONResponse(status_code=400, content={"error": "Provide exactly one of url, file or text along with the question."})
    try:
        content: str
        if text:
            content = text
        elif file is not None:
            data = await file.read()
            filename = file.filename or ""
            if filename.lower().endswith(".pdf") or (file.content_type and "pdf" in file.content_type):
                content = extract_text_from_pdf(data)
            else:
                content = data.decode("utf-8", errors="ignore")
        elif url:
            content = extract_text_from_url(url)
        else:
            return JSONResponse(status_code=400, content={"error": "No valid input provided."})
        answer_text = answer_question(content, question, top_n=max(1, top_sentences))
        return JSONResponse(content={"answer": answer_text})
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": str(err)})


@app.get("/openapi.yaml", include_in_schema=False)
async def get_openapi_spec() -> JSONResponse:
    """Serve the OpenAPI specification file.

    This endpoint reads the ``openapi.yaml`` file from disk and returns it as
    plain text so the ChatGPT plugin framework can retrieve the API schema.
    """
    try:
        with open("openapi.yaml", "r", encoding="utf-8") as f:
            spec = f.read()
        return JSONResponse(content=spec)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "OpenAPI specification not found."})


@app.get("/logo.png", include_in_schema=False)
async def get_logo() -> JSONResponse:
    """Serve the plugin logo image.

    This endpoint returns the PNG logo stored in the project directory. It is
    used by the ChatGPT plugin framework when displaying the plugin in the
    ChatGPT UI.
    """
    try:
        from fastapi.responses import FileResponse  # Lazy import
        return FileResponse("logo.png", media_type="image/png")
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": f"Failed to load logo: {err}"})
