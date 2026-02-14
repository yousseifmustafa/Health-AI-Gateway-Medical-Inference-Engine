<div align="center">

# 🩺 SehaTech AI — Medical Inference Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg?style=flat&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Persistence-4169E1.svg?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Zilliz](https://img.shields.io/badge/Vector_DB-Zilliz_Cloud-red.svg?style=flat&logo=zilliz&logoColor=white)](https://zilliz.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Cloudinary](https://img.shields.io/badge/Cloudinary-Media_CDN-3448C5.svg?style=flat&logo=cloudinary&logoColor=white)](https://cloudinary.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/MIT)

**Production-grade Autonomous Medical Triage System** — featuring Self-Healing Adaptive RAG, Multi-Provider LLM Fallback, Persistent Chat History via PostgreSQL, and Agentic Sub-Graph Orchestration for safe, hallucination-resistant diagnostic support.

</div>

---

## 🌟 Project Overview

**SehaTech AI** is a hierarchical **Agentic Medical System** designed to simulate a professional clinical triage process. It acts as a centralized Supervisor Agent that intelligently routes patient queries to specialized sub-systems — a **Deep Diagnostic Doctor** (Self-Healing RAG), **Web Search**, **Vision Analysis** (OCR/X-Ray), or **Emergency Family Notification** — based on real-time intent classification.

### What Makes This Different

| Capability | Traditional Chatbot | SehaTech AI |
|---|---|---|
| Diagnosis Quality | Single LLM call, no verification | **Self-Healing RAG Loop** — generates, grades, retrieves evidence, re-generates |
| Memory | Lost on restart | **PostgreSQL-backed persistent chat history** with per-user thread management |
| Reliability | Single API = single point of failure | **3-Tier LLM Fallback Chain** (HuggingFace → Groq → Gemini) |
| Performance | Blocking, synchronous | **Async-first** with lazy-loaded models and parallel retrieval |
| Language | English-only | **Multilingual** — handles Egyptian Arabic, Gulf Arabic, MSA, and English natively |

---

## 🏗️ System Architecture

### 1. Supervisor Agent — Tool-Based Triage Router

The Supervisor is a **Gemini 2.5 Flash**-powered LangGraph agent that classifies user intent and dispatches to specialized tools. It enforces the **"Golden Triangle"** protocol — collecting Symptoms, Duration, Severity, and Medical History before invoking diagnosis.

```mermaid
flowchart TD
    Start(["User Input"]) --> Router["Supervisor Agent\n Gemini 2.5 Flash + Tool Routing"]

    Router -- "Medical Symptoms" --> DoctorTool["consult_doctor_tool\n Diagnose Sub-Graph"]
    Router -- "General Info / Prices" --> WebTool["web_search_tool\n Tavily or Google Fallback"]
    Router -- "Image Uploaded" --> VisionTool["analyze_medical_image_tool\n Qwen2.5-VL / Gemini Vision"]
    Router -- "Emergency" --> FamilyTool["notify_family_tool\n SMS / Call Gateway"]

    DoctorTool --> Router
    WebTool --> Router
    VisionTool --> Router
    FamilyTool --> Router

    Router -- "Final Response" --> End(["Streamed Output via SSE"])

    DB[("PostgreSQL\nAsyncPostgresSaver")]
    Router -.-> |"checkpoint every turn"| DB
```

### 2. Self-Healing Adaptive RAG — The Diagnostic Engine

Unlike linear RAG pipelines, this system **generates first, judges second, and only retrieves if needed**. This "generate-then-verify" approach skips expensive vector search for simple queries while guaranteeing evidence-backed answers for complex cases.

```mermaid
flowchart TD
    Start(["Start"]) --> Generate["Generate Node\nSelf-Validating Medical LLM"]

    Generate --> GraderNode["Grade Node\nGemini Flash Lite - Structured Output"]

    GraderNode --> Router{"Score >= 0.7?\nOR\nRAG already ran?"}

    Router -- "Confident" --> End(["END - Return Answer"])
    Router -- "Needs Evidence" --> Retrieve

    subgraph RAG_Engine ["RAG Retrieval Engine"]
        direction TB
        Retrieve["Parallel Retrieval\nasyncio.gather x 3-4 queries"] --> Rerank["Cross-Encoder Reranker\nBAAI/bge-reranker-v2-m3"]
    end

    Rerank -- "Inject docs and retry" --> Generate
```

**Key design decisions:**
- **Confidence threshold = 0.7** — calibrated to allow strong parametric answers through while catching vague or unsafe responses
- **Loop protection** — RAG runs at most once; if docs are already present, the system finishes regardless of score
- **State stores text only** — `page_content` strings, not full `Document` objects, to prevent state serialization bloat

### 3. Multimodal Vision Pipeline

A dedicated async sub-graph for medical image analysis — decouples I/O-bound upload from compute-bound vision inference.

```mermaid
flowchart LR
    S(["Image + Query"]) --> UP["Upload Node\nAsync Cloudinary"]
    UP --> VIS["Vision Node\nQwen2.5-VL-7B / Gemini Fallback"]
    VIS --> E(["Analysis Result"])
    UP -.-> |"Upload Failed"| E
```

---

## ⚡ Architectural Highlights

### Singleton ModelManager — Load Once, Serve Forever
A **thread-safe Singleton** with Double-Checked Locking manages all LLM clients, embedding models, and rerankers. One instance serves the entire application — zero redundant initialization.

### Lazy-Loaded Heavy Models
The **embedding model** (Gemma-300M) and **reranker** (BGE-reranker-v2-m3) are **not loaded at startup**. They initialize on first access via `@property` decorators with thread-safe DCL, keeping cold-start time under 2 seconds.

### 3-Tier LLM Fallback Chain
Every LLM call follows: **HuggingFace → Groq → Gemini**. If the primary provider is rate-limited or down, the system silently switches — the user never sees an error. Near-100% uptime for generation.

### Async-First with `asyncio.to_thread()` Bridge
All LangGraph nodes are `async def`. Synchronous SDK calls (HuggingFace, Groq, FlagReranker) are offloaded to thread pools via `asyncio.to_thread()` — the event loop stays free for concurrent users.

### Persistent Chat History (PostgreSQL)
Conversations survive server restarts via **`AsyncPostgresSaver`** backed by an **`AsyncConnectionPool`** (min=2, max=10 connections). A custom `user_threads` table maps MongoDB user IDs to LangGraph thread UUIDs with **atomic upsert** to prevent race conditions.

### Parallel Vector Retrieval
Expanded queries (3-4 variations) are dispatched to Zilliz Cloud **concurrently** via `asyncio.gather()` — completing in the time of a single query.

### API Key Round-Robin
Each provider (HuggingFace, Google, Groq) loads multiple keys from environment variables and rotates through them, effectively multiplying the free-tier rate limit by the number of keys.

### Streaming Response (SSE)
Tokens stream to the frontend in real-time via FastAPI's `StreamingResponse` + LangGraph's `astream_events`. First token appears in ~200ms, with live Arabic status updates ("جاري استشارة الطبيب المختص...").

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|:---|:---|:---|
| **Orchestration** | LangGraph (StateGraph) | Multi-agent workflows with cyclic state and conditional routing |
| **Supervisor LLM** | Gemini 2.5 Flash | Intent classification and tool dispatch |
| **Medical LLM** | II-Medical-8B (HuggingFace) | Primary diagnostic generation |
| **Grading LLM** | Gemini 2.5 Flash Lite | Structured confidence scoring |
| **Fallback LLMs** | Groq (GPT-OSS-20B), Gemini | 3-tier fallback chain |
| **Vision LLM** | Qwen2.5-VL-7B-Instruct | Medical image OCR and analysis |
| **Embeddings** | Gemma-300M (256-dim, normalized) | Semantic vector representation |
| **Reranker** | BAAI/bge-reranker-v2-m3 (FP16) | Cross-encoder relevance scoring |
| **Vector DB** | Zilliz Cloud (Managed Milvus) | Medical document retrieval |
| **Persistence** | PostgreSQL + AsyncPostgresSaver | Durable chat history checkpointing |
| **Backend** | FastAPI | Async API with SSE streaming |
| **Frontend** | Streamlit | Chat UI with image upload |
| **Image CDN** | Cloudinary | Medical image hosting |
| **Web Search** | Tavily → Google Search (Fallback) | Real-time info retrieval |
| **Containerization** | Docker + docker-compose | Production deployment (12GB mem limit) |
| **Validation** | Pydantic | Structured LLM output parsing |

---

## 📁 Project Structure

```
SehaTech/
├── Langgraphs/
│   ├── supervisor_graph.py      # Supervisor Agent + Lazy Factory (make_graph)
│   ├── Diagnose_graph.py        # Self-Healing RAG Sub-Graph (4 nodes)
│   └── analyze_graph.py         # Vision Analysis Sub-Graph
├── Models/
│   └── Model_Manager.py         # Singleton ModelManager + Lazy Loading + Fallback Chain
├── Tools/
│   ├── Query_Optimization_Tool.py   # Translate → Rewrite → Expand (Few-Shot)
│   ├── parallel_retrievs_tool.py    # asyncio.gather fan-out retrieval
│   ├── reranker_tool.py             # Cross-Encoder reranking
│   ├── create_final_prompt_tool.py  # RAG/Memory-mode prompt builder
│   ├── Post_validation_tool.py      # Post-generation QA + style matching
│   └── Summary_tool.py             # Rolling conversation summarization
├── Helper/
│   ├── HF_ApiManager.py         # HuggingFace key round-robin
│   ├── Google_ApiManger.py      # Google key round-robin
│   ├── Groq_ApiManger.py        # Groq key round-robin
│   └── Image_Uploader.py        # Cloudinary async upload
├── vector_db/
│   └── VDB_Conection.py         # Zilliz Cloud retriever factory
├── Database_Manager.py          # PostgreSQL pool + checkpointer + user_threads
├── server.py                    # FastAPI streaming endpoint
├── app.py                       # Streamlit chat UI
├── langgraph.json               # LangGraph Studio entry points
├── docker-compose.yml           # Production container config
├── Dockerfile.backend           # Backend container image
└── requirments.txt              # Python dependencies
```

---

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yousseifmustafa/Health-AI-Gateway-Medical-Inference-Engine-.git
cd Health-AI-Gateway-Medical-Inference-Engine-
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirments.txt
```

### 4. Setup Environment Variables

Create a `.env` file in the root directory:

```env
# --- Model Configuration ---
GROQ_MODEL_NAME=openai/gpt-oss-20b
VALIDATION_MODEL_NAME=openai/gpt-oss-20b
OPTIMIZATION_MODEL_NAME=openai/gpt-oss-20b
GENERATION_MODEL_NAME=Intelligent-Internet/II-Medical-8B
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
OCR_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
EMBEDDING_MODEL_NAME=google/embeddinggemma-300m

# --- PostgreSQL (Persistent Chat History) ---
DATABASE_URL=postgresql://user:password@host:port/database

# --- API Keys (Multiple per provider for round-robin) ---
HUGGINGFACE_API_KEY1=your_hf_key_1
HUGGINGFACE_API_KEY2=your_hf_key_2
HUGGINGFACE_API_KEY3=your_hf_key_3

GOOGLE_API_KEY1=your_google_key_1
GOOGLE_API_KEY2=your_google_key_2
GOOGLE_API_KEY3=your_google_key_3

GROQ_API_KEY1=your_groq_key_1
GROQ_API_KEY2=your_groq_key_2
GROQ_API_KEY3=your_groq_key_3

# --- Vector Database ---
ZILLIZ_URI=your_zilliz_cloud_uri
ZILLIZ_TOKEN=your_zilliz_token
ZILLIZ_COLLECTION=seha_rag_collection

# --- Web Search ---
TAVILY_API_KEY=your_tavily_key
GOOGLE_CSE_ID=your_google_cse_id
GOOGLE_API_KEY=your_google_api_key

# --- Image Hosting ---
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_key
CLOUDINARY_API_SECRET=your_cloudinary_secret
```

### 5. Run the Application

**Option A — Local Development:**

```bash
# Terminal 1: Start FastAPI backend
uvicorn server:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run app.py
```

**Option B — Docker (Production):**

```bash
docker-compose up --build
```

**Option C — LangGraph Studio:**

The project registers three graphs in `langgraph.json` for visual debugging:
- `supervisor_graph` → Full triage agent with tools
- `Diagnose_graph` → RAG diagnostic pipeline (testable independently)
- `Image_Analyzer` → Vision analysis pipeline

---

## 🧪 Testing Scenarios

The system has been tested against complex medical edge cases:

| Test Case | What It Proves |
|---|---|
| 🧬 **The "Fabry Disease" Challenge** | Successfully diagnosed a rare, multi-organ genetic disorder by cross-referencing symptoms across cardiology, dermatology, and nephrology via vector retrieval |
| 🚫 **Hallucination Resistance** | Correctly refused to diagnose fabricated conditions (e.g., "Purple Hiccups Syndrome") — the validation layer rejected the low-confidence answer |
| 🚨 **Emergency Protocol** | Automatically triggered family notification for critical symptom patterns ("Chest pain + radiating to arm") without requiring user confirmation |
| 🌍 **Multilingual Triage** | Accepted Egyptian Arabic slang ("عندي صداع نصفي"), translated it for internal medical processing, and returned the diagnosis in the same dialect |
| 🔄 **Self-Healing Loop** | For ambiguous symptoms, the initial LLM answer scored < 0.7, triggering RAG retrieval → reranking → re-generation with evidence — producing a verified, cited answer |
| 💾 **Persistence Across Restarts** | Conversation history survived server restart via PostgreSQL checkpointing — the user resumed their session without repeating symptoms |

---

## 📐 Design Principles

1. **Never Crash** — Every component follows the pattern: try best option → fall back → graceful error message
2. **Never Block** — All I/O is async or offloaded to thread pools; the event loop stays free
3. **Never Waste** — Lazy loading, conditional RAG, and round-robin keys minimize resource usage and API costs
4. **Never Forget** — PostgreSQL persistence ensures no conversation is lost, even across deployments
5. **Never Hallucinate** — The Self-Healing RAG Loop + structured confidence grading catches unsafe answers

---

<div align="center">

**Built with** ❤️ **by** [Yousseif Mustafa](https://github.com/yousseifmustafa)

*SehaTech AI — Because in healthcare, "I don't know" is better than a wrong answer.*

</div>
