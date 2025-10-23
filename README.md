# DocGPT – Document Q&A & Summarization Plugin

DocGPT is a simple ChatGPT plugin that makes working with long documents a breeze. It
allows users to upload PDFs or text files, provide URLs, or paste raw text, and
then either generate concise summaries or ask questions about the content. This
project implements a small web service with two endpoints:

* **`POST /summarize`** – Summarises a document into a specified number of
  sentences. Exactly one of `url`, `file` or `text` must be supplied in the
  multipart/form-data request. The optional `sentences` field controls the
  summary length (defaults to 5 sentences).
* **`POST /answer`** – Answers a question about a document by returning the
  most relevant sentences. The request must include `question` and exactly one
  of `url`, `file` or `text`. The `top_sentences` field determines how many
  sentences are returned (default is 3).

The plugin uses basic natural language techniques (word frequency analysis for
summarisation and TF–IDF for question answering). It is intentionally lightweight
to minimise development effort while still delivering helpful results.

## Installation

1. Clone this repository or copy the `document_plugin` folder.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the server using Uvicorn:

   ```bash
   uvicorn server:app --host 0.0.0.0 --port 3333
   ```

4. Once running, the OpenAPI specification can be found at
   `http://localhost:3333/openapi.yaml` and the plugin manifest at
   `http://localhost:3333/ai-plugin.json` (served automatically by the
   ChatGPT plugin environment).

## Usage

### Summarise a document

Send a `POST` request to `/summarize` with one of the following fields:

* `url`: A URL pointing to the document you want to summarise.
* `file`: A PDF or plain text file uploaded via multipart/form-data.
* `text`: Raw text to summarise.

You can optionally include `sentences` to specify the number of sentences in
the summary.

Example using `curl`:

```bash
curl -X POST http://localhost:3333/summarize \
  -F url="https://www.example.com/article" \
  -F sentences=3
```

### Answer a question about a document

Send a `POST` request to `/answer` with the following fields:

* `question` (required): The question you want answered.
* `url`, `file` or `text` (exactly one required): The document to search
  for the answer.
* `top_sentences` (optional): How many sentences to return in the answer
  (default 3).

Example using `curl`:

```bash
curl -X POST http://localhost:3333/answer \
  -F question="What is the termination clause?" \
  -F file=@/path/to/contract.pdf \
  -F top_sentences=2
```

## License

This code is provided as‑is for demonstration purposes. You are free to
modify and use it in your own projects. No warranties or guarantees are
provided.