from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END, START
import asyncio
import threading
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from Models.Model_Manager import get_model_manager
    from vector_db.VDB_Conection import create_retriever
    from Tools.reranker_tool import async_rerank_contexts
    from Tools.create_final_prompt_tool import create_final_prompt
    from Tools.parallel_retrievs_tool import parallel_retrieval
except ImportError as e:
    print(f"FATAL ERROR: Could not import helper functions: {e}")
    exit()


# --- Lazy Retriever Holder (ModelManager is shared via get_model_manager) ---
class _RetrieverHolder:
    """Lazy singleton for the vector DB retriever only. ModelManager comes from the shared singleton."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._retriever = None
        return cls._instance

    @property
    def context_retriever(self):
        """Lazy-load the retriever on first access (requires embedding model)."""
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    print("INFO: Lazy-initializing Zilliz Retriever...")
                    mm = get_model_manager()
                    self._retriever = create_retriever(mm.embedding_model)
        return self._retriever


def _get_retriever():
    return _RetrieverHolder().context_retriever


# --- 1. State Definition ---
class DiagnoseState(TypedDict):
    user_query: str
    conversation_summary: str

    # Pre-optimized by the Supervisor — passed as initial inputs
    translated_query: str
    expanded_queries: List[str]

    # Unified Answer Field
    english_medical_answer: str
    confidence_score: float

    # RAG Fields (strings only — no heavy Document objects in state)
    support_contents: List[str]
    final_docs: List[str]

    final_answer: Optional[str]


# --- 2. Helper Models ---
class GradeOutput(BaseModel):
    score: float = Field(description="Confidence score between 0.0 and 1.0 regarding medical accuracy and safety.")


# --- 3. Nodes ---

async def generate_node(state: DiagnoseState):
    """
    Unified Generator with Self-Validation:
    Generates a medically accurate answer, validates it inline, and matches
    the user's language/dialect automatically.
    """
    query = state["translated_query"]
    user_query = state["user_query"]
    summary = state["conversation_summary"]
    docs = state.get("final_docs", [])
    mm = get_model_manager()

    # --- Build evidence section ---
    if docs:
        evidence_block = "## Provided Medical Evidence (Use as primary source *if relevant*):\n"
        for i, doc in enumerate(docs[:4]):
            content = doc if isinstance(doc, str) else str(doc)
            if content:
                evidence_block += f"{i+1}. {' '.join(content.split())}\n"
    else:
        evidence_block = "No external evidence provided. Rely SOLELY on your internal medical knowledge."

    # --- Self-Validating Prompt ---
    system_prompt = f"""You are an expert-level Medical Analyst AI with built-in Quality Assurance.
Your purpose is to provide a detailed, validated, and professionally formatted diagnostic response.

STRICT PROTOCOL — Follow ALL steps in order:

**STEP 1: ANALYZE**
- Assess the clinical picture from the user query.
- If provided evidence exists, cross-reference it. Discard irrelevant or contradictory evidence.
- Formulate a Differential Diagnosis (if applicable).

**STEP 2: SELF-VALIDATE**
- Before outputting, internally verify:
  1. Factual Accuracy: No hallucinated drug names, dosages, or conditions.
  2. Medical Safety: No dangerous or misleading advice.
  3. Completeness: All aspects of the query are addressed.
- If you detect an error in your own reasoning, correct it silently before output.

**STEP 3: FORMAT & LANGUAGE MATCH**
- **Language Detection**: Analyze the "Original User Query" below. Detect its exact language and dialect (e.g., Egyptian Arabic, Gulf Arabic, Formal Arabic MSA, English).
- **Output Language**: Your ENTIRE response MUST be in the SAME language/dialect as the Original User Query.
- **Hybrid Translation Rule**: Keep technical medical terms and drug names in English inside brackets, e.g., (Paracetamol), (Hypertension), then explain them in the user's language.
- **Formatting**: Use structured bullet points (•) and clear section headers. Be professional but warm.

EXAMPLE OUTPUT STRUCTURE (adapt language to match user):
• **التشخيص المحتمل (Possible Diagnosis):** [...]
• **الأسباب المحتملة (Possible Causes):** [...]
• **الخطوات المقترحة (Recommended Actions):** [...]
• **أدوية مقترحة (Suggested Medications):** [...]
• **⚠️ تحذيرات (Warnings):** [...]
"""

    user_prompt = f"""## Previous Conversation Summary:
{summary.strip() if summary else "None"}

## Original User Query (MATCH THIS LANGUAGE/DIALECT):
{user_query}

## Translated Medical Query (for your internal analysis):
{query}

{evidence_block}

## Task:
Provide your expert, self-validated medical analysis following ALL protocol steps above.
"""

    try:
        final_input = f"{system_prompt}\n\n{user_prompt}"
        answer = await mm.agenerate_answer(final_input)
    except Exception as e:
        print(f"Generator Error: {e}")
        answer = "I cannot generate an answer at this moment."

    return {"english_medical_answer": answer, "final_answer": answer}


async def grade_answer_node(state: DiagnoseState):
    """The Judge — Evaluate Confidence. Routes to END or RAG loop."""
    query = state["translated_query"]
    answer = state["english_medical_answer"]
    mm = get_model_manager()

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=mm.Google_key_manger.get_next_api_key(),
            temperature=0
        )
        grader_llm = llm.with_structured_output(GradeOutput)
        system_msg = """You are a strict medical supervisor. Rate the answer based on:
        1. Relevance to the query.
        2. Medical Safety (no dangerous hallucinations).
        3. Completeness.
        Give a score < 0.7 if the answer is vague, says "I don't know", or needs external verification.
        """

        grade = await grader_llm.ainvoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Query: {query}\nAnswer: {answer}")
        ])
        score = grade.score
        print(f"Grade Score is ={score}\n ")
    except Exception as e:
        print(f"Grading Failed: {e}")
        score = 0.0

    return {"confidence_score": score}


async def retrieve_node(state: DiagnoseState):
    """Retrieve Documents (Only runs if needed). Stores only text content, not full objects."""
    expanded_queries = state["expanded_queries"]
    retriever = _get_retriever()
    try:
        docs = await parallel_retrieval(retriever, expanded_queries)
    except Exception:
        docs = []
    # Store only page_content strings — avoids state bloat from full Document objects
    contents = [doc.page_content for doc in docs if hasattr(doc, 'page_content')]
    return {"support_contents": contents}


async def rerank_node(state: DiagnoseState):
    """Filter Documents & Flag Context Presence. Skips reranking if docs ≤ top_n."""
    query = state["translated_query"]
    docs = state.get("support_contents", [])
    mm = get_model_manager()
    top_n = 3

    # Skip reranker if we already have ≤ top_n docs — no point scoring
    if len(docs) <= top_n:
        return {"final_docs": docs}

    try:
        final_docs = await async_rerank_contexts(
            query, docs, mm.reranker_Model, top_n
        )
    except Exception:
        final_docs = docs[:top_n]

    return {"final_docs": final_docs}


# --- 4. Logic & Graph Wiring ---

def decide_next_step(state: DiagnoseState):
    """The Router Logic"""
    score = state.get("confidence_score", 0.0)
    docs = state.get("final_docs", [])

    # High confidence OR already ran RAG loop → done
    if score >= 0.7:
        return "finish"

    if docs:
        return "finish"

    return "retrieve"


# Graph Construction
diagnostic_workflow = StateGraph(DiagnoseState)

# Add Nodes (slim: 4 nodes only)
diagnostic_workflow.add_node("generate", generate_node)
diagnostic_workflow.add_node("grade", grade_answer_node)
diagnostic_workflow.add_node("retrieve", retrieve_node)
diagnostic_workflow.add_node("rerank", rerank_node)

# Edges — START directly to generate (query pre-optimized by Supervisor)
diagnostic_workflow.add_edge(START, "generate")
diagnostic_workflow.add_edge("generate", "grade")

# Conditional: high confidence or RAG done → END, else → retrieve
diagnostic_workflow.add_conditional_edges(
    "grade",
    decide_next_step,
    {
        "finish": END,
        "retrieve": "retrieve"
    }
)

# RAG Loop Back
diagnostic_workflow.add_edge("retrieve", "rerank")
diagnostic_workflow.add_edge("rerank", "generate")

# Compile
diagnose_app = diagnostic_workflow.compile()
__all__ = ["diagnose_app", "DiagnoseState"]