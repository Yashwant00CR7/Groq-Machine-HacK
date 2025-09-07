# Librarian

"Your personal AI librarian that never stops searching for the documentation you need."

> Status: Core ingestion + self‑correcting pipeline + cloud + stdio MCP integration working. Ongoing: broader MCP feature surface & editor plugins.  
> License: (none yet—add one before public release)

## TL;DR
Give me a library name (PyPI / npm / crates.io). I:  
1. Discover the richest real docs page (not a thin homepage)  
2. Fast-scrape via Jina AI  
3. If weak → deep crawl with Crawl4ai  
4. Embed & refresh Pinecone (cosine, 384D)  
5. Run RAG + confidence scoring  
6. Serve structured results via:  
	 - Local MCP (stdio) server (`mcp_stdio_server.py`)  
	 - Cloud endpoints (`/process_library`, `/ask`) at `LIBRARIAN_CLOUD_URL`our personal AI librarian that never stops searching for the documentation you need.”

> Status: Core ingestion + self‑correcting pipeline + cloud + stdio MCP integration working. Ongoing: broader MCP feature surface & editor plugins.  
> License: (none yet—add one before public release)

## TL;DR
Give me a library name (PyPI / npm / crates.io). I:  
1. Discover the richest real docs page (not a thin homepage)  
2. Fast-scrape via Jina AI  
3. If weak → deep crawl with Crawl4ai  
4. Embed & refresh Pinecone (cosine, 768D)  
5. Run RAG + confidence scoring  
6. Serve structured results via:  
	 - Local MCP (stdio) server (`mcp_stdio_server.py`)  
	 - Cloud endpoints (`/process_library`, `/ask`) at `LIBRARIAN_CLOUD_URL`

---

## 🔧 Implemented Capabilities

- Multi‑ecosystem resolution (PyPI, npm, crates.io)
- Intelligent documentation URL selection (Groq Llama 3.1 + search)
- Fast initial content extraction (Jina AI Reader)
- Confidence scoring + conditional deep crawl fallback (Crawl4ai)
- Deep multi‑page aggregation
- Vector refresh (delete old → upsert new) in Pinecone serverless:
	- Index: `mcp-documentation-index-groq` (cosine, dim=384, us-east-1)
- Groq Llama 3.1 reasoning & structured summarization
- Cache gated by confidence (Medium/High only)
- Dual operation modes: Local MCP stdio + Cloud HTTP service

## 🧱 Architecture Overview

### High-Level Data Flow
1. Request (MCP stdio or Cloud HTTP)
2. Ecosystem metadata fetch
3. Candidate docs discovery (Groq Llama 3.1 + DuckDuckGo + Tavily fallback)
4. Initial scrape (Jina)
5. Content quality + confidence evaluation
6. (Optional) Deep crawl (Crawl4ai)
7. Chunking + Embedding
8. Pinecone index refresh
9. RAG + structuring
10. Response + optional cache

### Core Components

| File | Responsibility |
|------|----------------|
| `config.py` | Env loading, constants (index name), Pydantic schemas |
| `services.py` | Tools (ecosystem lookups, search, scraping), pipeline stages, embeddings, RAG |
| `main.py` | Orchestrates one full ingestion run locally |
| `agent.py` | Legacy/simple launcher |
| `mcp_stdio_server.py` | MCP stdio adapter exposing actions |
| `.vscode/mcp.json` | Local MCP client configuration |
| `requirements.txt` | Dependencies |
| `.env` | Local secrets (never commit real values) |

### Conceptual Data Models

## 🔄 Two-Way Model Communication in API

The API now supports robust two-way communication between the orchestrator and both text and vision models:

- **Text Model (Groq Llama 3.1):** Handles all text-based reasoning, documentation extraction, and RAG workflows. The orchestrator sends queries and receives structured responses, including confidence scores and extracted metadata.
- **Vision Model (Meta Llama 4 Scout):** Used for image-based library identification. The orchestrator can send image data and prompts, receiving back the identified library name or related metadata.
- **Dynamic Model Selection:** The orchestrator chooses the appropriate model (text or vision) based on the input type (text or image), enabling seamless multi-modal support.
- **Structured Output:** Both models return structured data, which is then used for downstream tasks like caching, embedding, and serving answers.
- **Error Handling:** The orchestrator manages model errors and fallbacks, ensuring reliability in both directions.

This two-way communication enables richer, multi-modal documentation search and Q&A, supporting both textual and visual queries in the same pipeline.

---

## 🗺️ Mermaid Diagrams

### Flow
```mermaid
flowchart LR
    A["Client (MCP / Cloud)"] --> B["Request Handler"]
    B --> C["Library Resolver<br/>(PyPI / npm / Crates)"]
    C --> D["URL Discovery Agent<br/>Groq Llama 3.1 + DuckDuckGo + Tavily"]
    D --> E["Jina Fast Scrape"]
    E --> F{"Confidence High?"}
    F -- "Yes" --> H["Chunk & Embed"]
    F -- "No" --> G["Deep Crawl<br/>Crawl4ai"]
    G --> H
    H --> I["Pinecone Refresh<br/>(delete+upsert)"]
    I --> J["RAG + Structuring"]
    J --> K["Cache (if Medium/High)"]
    K --> L["Response"]
```

### Sequence (Ask)
```mermaid
sequenceDiagram
		participant C as Client
		participant H as Handler
		participant P as Pinecone
		participant R as Retriever
		participant G as Groq
		C->>H: ask(library, question)
		H->>P: similarity query
		P-->>H: top-k chunks
		H->>R: assemble context
		R->>G: prompt(context, question)
		G-->>H: answer + reasoning
		H-->>C: structured answer
```

---

## 🧠 Discovery Strategy

- Reasoning model: `llama-3.1-8b-instant` (Groq)
- Search order: Ecosystem metadata → DuckDuckGo queries → Tavily fallback
- URL scoring heuristics: Penalize marketing roots; prefer pages with heading density, code blocks, versioned paths; reject empty stubs.

## 🧪 Confidence & Fallback

| Stage | Trigger |
|-------|---------|
| Deep Crawl | Confidence Low/Unknown OR text below length threshold |
| Re-embed | Always after deep crawl |
| Cache | Only Medium / High |
| Ask pre-processing | If not indexed → run pipeline first |

---

## 🧰 Toolchain / Services

| Concern | Choice |
|---------|--------|
| Reasoning & extraction logic | Groq Llama 3.1 8B Instant |
| Embeddings (384D) | BAAI/bge-small-en-v1.5 (FastEmbed) |
| Fast scrape | Jina AI Reader |
| Deep crawl | Crawl4ai (`crawl4ai-setup` installs Chromium) |
| Search | DuckDuckGo (primary), Tavily (fallback) |
| Vector store | Pinecone serverless (cosine) |
| Protocols | MCP stdio, Cloud HTTP |
| Runtime | Python 3.11 |

---

## ⚙️ Environment Variables

(Do NOT commit real key values. Rotate any leaked keys immediately.)

```
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=mcp-documentation-index-groq
PINECONE_REGION=us-east-1
TAVILY_API_KEY=your_tavily_key
LIBRARIAN_CLOUD_URL=https://librarian-ai-agent-groq-180553692497.us-central1.run.app/
# Optional / future:
# CACHE_DIR=.cache/librarian
```

---

## 📦 Installation (Local)

```cmd
pip install -r requirements.txt
crawl4ai-setup
```

`crawl4ai-setup` handles Playwright + Chromium; no manual `playwright install` needed.

Create `.env` (never commit real values).

---

## ▶️ Local Pipeline Run

If `main.py` supports args:
```cmd
python main.py --library tokio --ecosystem crates
```
Otherwise edit the target inside `main.py` and:
```cmd
python main.py
```

---

## 🔌 MCP (Local Stdio)

`.vscode/mcp.json` points to `mcp_stdio_server.py`.

Example (conceptual inside AI-enabled editor):
```
process_library: { "library_name": "tokio" }
ask: { "library_name": "tokio", "question": "How do I create a TCP server?" }
```

---

## ☁️ Cloud HTTP Mode

**🚀 Live Demo:** [https://librarian-ai-agent-groq-180553692497.us-central1.run.app/docs](https://librarian-ai-agent-groq-180553692497.us-central1.run.app/docs)

Base: `${LIBRARIAN_CLOUD_URL}`

Endpoints:
- `POST /process_library`
	```json
	{ "library_name": "tokio" }
	```
- `POST /ask`
	```json
	{ "library_name": "tokio", "question": "How to use async TCP?" }
	```

Python example:
```python
import os, requests
BASE = os.environ["LIBRARIAN_CLOUD_URL"].rstrip("/")

r = requests.post(f"{BASE}/process_library", json={"library_name":"tokio"})
print(r.json())

q = requests.post(f"{BASE}/ask", json={
		"library_name":"tokio",
		"question":"Show a minimal async TCP server example."
})
print(q.json())
```

(If adding auth later, include an `Authorization: Bearer <token>` header.)

---

## 🔍 Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|-----------|
| Very short summary | Thin initial page | Deep crawl auto-triggers; re-run if still low |
| Crawl errors | Chromium missing | Re-run `crawl4ai-setup` |
| Pinecone auth failure | Wrong key/region/index | Verify env values |
| Repeated Low confidence | Poor URL selection | Refine heuristics / add denylist |
| Slow first run | Model download + warm start | Subsequent runs faster |
| MCP stdio not spawning | Wrong Python path | Fix `.vscode/mcp.json` command |
| Embedding model errors | Model not downloaded | Check Docker build or run download manually |

---

## 🧩 Extending Ecosystems

1. New resolver (API call + candidate docs extraction)  
2. Integrate into discovery ranking  
3. Optional heuristics tuned per ecosystem  
4. Reuse ingestion + fallback unchanged  

---

## 🔐 Security & Hygiene

- Remove any real keys from history; rotate compromised keys.  
- Ensure `.env` is in `.gitignore`.  
- Plan authentication for public cloud endpoints.  
- Avoid logging raw page content in production (use sampling).  

---

## 🚧 Roadmap (Condensed)

| Horizon | Focus |
|---------|-------|
| Short | Solidify MCP surface, answer grounding |
| Mid | VS Code & Neovim plugins, improved relevance scoring |
| Long | Dependency conflict prediction, community doc quality signals |

---

## 📝 License

None yet—add one (e.g., MIT or Apache-2.0) before external contributions.

---

## 🤝 Future Contributing Guidelines (Planned)

- Include reproducible test run output for new ecosystems
- Avoid committing secrets
- Provide small fixture datasets for deterministic tests

---

## 📜 Example End-to-End (`tokio`)

```
process_library(tokio)
 → discovery selects https://docs.rs/tokio
 → jina scrape (enough?) if not → deep crawl
 → vectors refreshed (Pinecone cosine 384D)
 → structured summary (confidence: High)

ask(tokio, "How do I spawn tasks?")
 → retrieve top-k → Groq Llama 3.1 synthesis with source context
```

---

## ✅ Summary

Librarian delivers a resilient, self-correcting documentation ingestion + RAG stack with dual local (MCP stdio) and cloud operation modes, leveraging Groq Llama 3.1 + Jina + Crawl4ai + Pinecone to keep results fresh and high quality.
