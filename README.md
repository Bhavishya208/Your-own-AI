# VectorDB 

A Python vector database built from scratch.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms, plus a **RAG pipeline** powered by a local LLM via Ollama.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

---

## Project Structure

```
VectorDB/
├── main.py        ← Python backend (HNSW, KD-Tree, BruteForce, REST API, RAG)
├── requirements.txt
├── index.html     ← Frontend (same as original — no changes needed)
└── README.md
```

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** (runs the local AI models)

---

## Setup

### Step 1 — Install Python dependencies

```bash
pip install flask requests
```

Or with the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2 — Install Ollama

Go to https://ollama.com and download for your OS, then pull the models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Step 3 — Run

**Terminal 1** — Start Ollama (if not already running):
```bash
ollama serve
```

**Terminal 2** — Start the Python server:
```bash
python main.py
```

You should see:
```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open your browser at **http://localhost:8080**

---

## REST API

Same as the original — all endpoints are identical:

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/doc/insert` | Embed and store document |
| `GET` | `/doc/list` | List all stored documents |
| `DELETE` | `/doc/delete/:id` | Delete document chunk |
| `POST` | `/doc/ask` | RAG: retrieve + generate |
| `GET` | `/status` | Ollama status and model info |

---

## Architecture (main.py)

```
BruteForce      O(N·d)    Exact, baseline
KDTree          O(log N)  Exact, axis-aligned partitioning
HNSW            O(log N)  Approximate, multilayer small-world graph

VectorDB        Unified interface over all 3 (16D demo vectors)
DocumentDB      HNSW-only index for real Ollama embeddings (768D)
OllamaClient    HTTP client → /api/embeddings + /api/generate
```

---

## License

MIT
