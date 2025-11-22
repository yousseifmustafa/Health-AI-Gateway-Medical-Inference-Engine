# 🩺 Health-AI-Gateway

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg?style=flat&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Zilliz](https://img.shields.io/badge/Vector_DB-Zilliz_Cloud-red.svg?style=flat&logo=zilliz&logoColor=white)](https://zilliz.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Embeddings-FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Lightning AI](https://img.shields.io/badge/Lightning%20AI-Deployment-792EE5.svg?style=flat&logo=lightning&logoColor=white)](https://lightning.ai/)
[![Cloudinary](https://img.shields.io/badge/Cloudinary-Media_Hosting-3448C5.svg?style=flat&logo=cloudinary&logoColor=white)](https://cloudinary.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://opensource.org/licenses/MIT)

</div>
> **State-of-the-art Autonomous Medical Triage System** utilizing Adaptive RAG, Semantic Chunking, and Agentic Orchestration to deliver safe, hallucination-free diagnostic support.

---

## 🌟 Project Overview

**Health-AI-Gateway** is not just a chatbot; it's a hierarchical **Agentic System** designed to simulate a professional medical triage process. It acts as a centralized orchestrator that intelligently routes patient queries to specialized agents (Diagnostic Doctor, Web Search, or Vision Analysis) based on intent.

The core innovation lies in its **Self-Correcting Diagnostic Loop**, where the system continuously evaluates its own confidence levels. If the confidence is below a safety threshold (0.8), it autonomously triggers a Retrieval-Augmented Generation (RAG) loop to fetch verified medical evidence from a vector database, ensuring maximum safety and accuracy.

---

## 🏗️ System Architecture

### 1. Agentic Orchestration (The Triage Router)
The system operates on a **Star Topology**, where a central "Brain" decides the optimal tool for the job.

```mermaid
flowchart TD
    Start(("User Input")) --> Router["🧠 Triage Router (Agent)\n(Intent Classification)"]
    
    Router -- "Medical Triage" --> DoctorTool["🩺 Consult Doctor Tool\n(Diagnostic Agent)"]
    Router -- "General Info/Prices" --> WebTool["🌐 Web Search Tool\n(Tavily API)"]
    Router -- "X-Ray/Prescription" --> VisionTool["👁️ Vision Analysis Tool\n(Gemini Vision/OCR)"]
    Router -- "Emergency" --> FamilyTool["🚨 Notify Family Tool\n(Twilio/Mock SMS)"]
    
    DoctorTool --> Router
    WebTool --> Router
    VisionTool --> Router
    FamilyTool --> Router
    
    Router -- "Safe Response" --> End(("Final Output"))

```



### 2. Self-Correcting Adaptive RAG

Unlike traditional linear RAG, this system "thinks" before it speaks.

```mermaid
flowchart TD
    %% 1. Start & Setup
    Start(("Start")) --> Optimize["Optimize Query<br/>(Translate & Expand)"]
    
    %% 2. The Unified Generator
    Optimize --> Generate["Unified Generator Node<br/>(1st Pass: Memory | 2nd Pass: With Docs)"]
    
    %% 3. The Grading Action (Separated Node)
    Generate --> GraderNode["Grader Node<br/>(Evaluate Answer & Update State with Score)"]
    
    %% 4. The Decision Logic (Conditional Edge/Router)
    GraderNode --> Router{"Router / Decision<br/>Is Score >= 0.8?<br/>OR<br/>Has RAG Run Before?"}

    %% 5. Branching
    Router -- "✅ Yes / Safe" --> Validate
    Router -- "❌ No (Need Context)" --> Retrieve

    %% 6. The RAG Loop (Fetcher)
    subgraph RAG_Engine [RAG Retrieval Engine]
        direction TB
        Retrieve["Retrieve Docs<br/>(Vector DB)"] --> Rerank["Rerank Contexts"]
    end
    
    %% The Critical Loop Back
    Rerank -- "Inject Docs & Retry" --> Generate

    %% 7. Finalization
    Validate["Validator Node<br/>(Style & Safety Check Only)"] --> Summarize["Summarize Conversation"]
    Summarize --> End(("End"))

    %% Styling
    classDef mainNode fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef processNode fill:#e1bee7,stroke:#4a148c,stroke-width:2px;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,stroke-dasharray: 5 5;
    classDef ragNode fill:#e3f2fd,stroke:#2196f3,stroke-width:2px;
    classDef endNode fill:#000,stroke:#000,color:#fff;

    class Optimize,Generate,Validate,Summarize mainNode;
    class GraderNode processNode;
    class Router decision;
    class Retrieve,Rerank ragNode;
    class Start,End endNode;
    
    %% Highlight the loop edge
    linkStyle 6 stroke:#2196f3,stroke-width:3px,color:red;
```

### 3. Multimodal Vision Pipeline (Prescriptions & Drug-Box)
A dedicated, asynchronous pipeline for handling medical imagery. It decouples the upload process (I/O bound) from the vision analysis (Compute bound) to ensure system responsiveness.

```mermaid
flowchart TD
    %% Nodes Definitions
    S(("Start"))
    UP["☁️ Upload Node<br/>(Cloudinary API)"]
    VIS["👁️ Vision Analysis Node<br/>(Groq / HF / Gemini)"]
    E(("End / Output"))

    %% Happy Path
    S -->|User Query + Image Bytes| UP
    UP -->|Secure Image URL| VIS
    VIS -->|Medical Description / OCR| E

    %% Error Handling
    UP -.->|Upload Failed| E
    
    %% Styling
    style S fill:#f9f9f9,stroke:#333,stroke-width:2px
    style E fill:#000,stroke:#000,color:#fff
    style UP fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style VIS fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```
## 🚀 Key Features

* **🤖 Agentic Triage Router:** Uses LLM-based intent classification to dynamically dispatch tasks. It knows the difference between *"I have a headache"* (Doctor) and *"How much is Panadol?"* (Search).
* **🔄 Self-Correcting Loop:** A specialized **Judge Agent** evaluates every answer. If confidence < **80%**, it rejects the answer and forces the system to consult external medical references (Zilliz).
* **🧠 Semantic Data Engineering:** Utilizing **Gemma-300m** for advanced Semantic Chunking during data ingestion, ensuring that retrieved context is coherent and meaningful (solved the *Fabry Disease* challenge).
* **🛡️ Safety Guardrails:** A final validation layer filters output for medical safety and adapts the response tone to match the user's dialect (e.g., Egyptian Slang handling).
* **👁️ Multimodal Support:** Can analyze medical images (X-Rays, Prescriptions) using Vision Language Models.
* **⚡ Resilient Infrastructure:** Built on **FastAPI** with async architecture, Dockerized for easy deployment, and includes automated "Keep-Alive" triggers for the Vector DB.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | **LangGraph** | Managing cyclic state and multi-agent workflows. |
| **LLM Core** | **ModelManager** | Centralized wrapper for high-speed inference routing and caching. |
| **Vector DB** | **Zilliz Cloud (Milvus)** | Storing 2M+ medical semantic chunks. |
| **Embeddings** | **Gemma-300m** | Open-weights model for semantic text representation. |
| **Backend** | **FastAPI** | Asynchronous, high-performance web framework. |
| **Validation** | **Pydantic** | Strict output parsing and structured data validation. |
| **Search** | **Tavily API** | Real-time web search for drug prices and clinics. |
| **Media Storage** | **Cloudinary** | Secure cloud hosting and optimization for uploaded medical images. |
| **Deployment** | **Lightning AI** | Scalable cloud infrastructure for hosting the full AI pipeline. |

```
## 💻 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yousseifmustafa/Health-AI-Gateway.git](https://github.com/yousseifmustafa/Health-AI-Gateway.git)
   cd Health-AI-Gateway
   ```

2. **Creare Virtual Enviroment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. ***Setup Environment Variables Create a .env file in the root directory:***

```Code snippet
GROQ_MODEL_NAME = "openai/gpt-oss-20b"
VALIDATION_MODEL_NAME = "openai/gpt-oss-20b"
OPTIMIZATION_MODEL_NAME = "openai/gpt-oss-20b"
GENERATION_MODEL_NAME="Intelligent-Internet/II-Medical-8B"
RERANKER_MODEL_NAME =  "BAAI/bge-reranker-v2-m3"
OCR_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"

GOOGLE_CSE_ID=your_google_cse_id
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_GROQ_API_KEY
TAVILY_API_KEY=your_tavily_key
ZILLIZ_URI=your_zilliz_uri
ZILLIZ_TOKEN=your_zilliz_token
HF_TOKEN=your_huggingface_token

CLOUDINARY_CLOUD_NAME="your_CLOUDINARY_CLOUD_NAME"
CLOUDINARY_API_KEY="your_CLOUDINARY_API_KEY"
CLOUDINARY_API_SECRET="your-CLOUDINARY_API_SECRET"
```
5. ***Run the Streamlit Interface***

```bash
streamlit run app.py
```



## 🧪 Testing Scenarios

The system has been rigorously tested against complex medical cases to ensure robustness and safety:

* **🧬 The "Fabry Disease" Challenge:** Successfully diagnosed a rare, multi-symptom genetic disorder (connecting symptoms across cardiology, dermatology, and nephrology) using cross-domain retrieval from the vector database.
* **🚫 Hallucination Check:** Correctly identified fabricated conditions (e.g., *"Purple Hiccups Syndrome"*) and strictly refused to invent treatments, proving the effectiveness of the validation layer.
* **🚨 Safety Protocols:** Automatically triggers emergency protocols and family notifications for critical keywords (e.g., *"Chest pain + Radiating to arm"*).

---



<div align="right">
    Made with ❤️ by <a href="https://github.com/yousseifmustafa">Yousseif Mustafa</a>
</div>

