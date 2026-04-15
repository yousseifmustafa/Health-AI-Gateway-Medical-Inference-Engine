from typing import List, Optional, Any
from langgraph.graph import StateGraph, END, START
import asyncio
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from Langgraphs.states import DiagnoseInputState, DiagnoseOutputState, DiagnoseState, GradeOutput
from Langgraphs.prompts import DIAGNOSE_SYSTEM_PROMPT, DIAGNOSE_USER_PROMPT, GRADING_SYSTEM_PROMPT
from Langgraphs.utils import _get_retriever

logger = logging.getLogger("sehatech.diagnose")

try:
    from Models.Model_Manager import get_model_manager
    from Tools.reranker_tool import async_rerank_contexts
    from Tools.parallel_retrievs_tool import parallel_retrieval
except ImportError as e:
    logger.critical("Could not import helper functions: %s", e)
    exit()

# --- Nodes ---

async def generate_node(state: DiagnoseState):
    """
    Unified Generator with Self-Validation:
    Generates a medically accurate answer, validates it inline, and matches
    the user's language/dialect automatically.
    """
    query = state.translated_query
    user_query = state.user_query
    summary = state.conversation_summary
    docs = state.final_docs  
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
    user_prompt = DIAGNOSE_USER_PROMPT.format(
        summary=summary.strip() if summary else "None",
        user_query=user_query,
        query=query,
        evidence_block=evidence_block
    )

    try:
        final_input = f"{DIAGNOSE_SYSTEM_PROMPT}\n\n{user_prompt}"
        answer = await mm.agenerate_answer(final_input)
    except Exception as e:
        logger.exception("Generator Error")
        answer = "I cannot generate an answer at this moment."

    return {"english_medical_answer": answer, "final_answer": answer}


async def grade_answer_node(state: DiagnoseState):
    """The Judge — Evaluate Confidence. Routes to END or RAG loop."""
    query = state.translated_query
    answer = state.english_medical_answer
    mm = get_model_manager()

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=mm.google_key_manager.get_next_api_key(),
            temperature=0
        )
        grader_llm = llm.with_structured_output(GradeOutput)

        grade = await grader_llm.ainvoke([
            SystemMessage(content=GRADING_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}\nAnswer: {answer}")
        ],
        config={"callbacks": []}
        )
        score = grade.score
        logger.info("Grade Score = %s", score)
    except Exception as e:
        logger.warning("Grading Failed: %s", e)
        score = 0.0

    return {"confidence_score": score}


async def retrieve_node(state: DiagnoseState):
    """Retrieve Documents (Only runs if needed). Stores only text content, not full objects."""
    expanded_queries = state.expanded_queries
    try:
        retriever = _get_retriever()
        docs = await parallel_retrieval(retriever, expanded_queries)
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        docs = []
    # Store only page_content strings — avoids state bloat from full Document objects
    contents = [doc.page_content for doc in docs if hasattr(doc, 'page_content')]
    return {"support_contents": contents}


async def rerank_node(state: DiagnoseState):
    """Filter Documents & Flag Context Presence. Skips reranking if docs ≤ top_n."""
    query = state.translated_query
    docs = state.support_contents
    mm = get_model_manager()
    top_n = 3

    # Skip reranker if we already have ≤ top_n docs — no point scoring
    if len(docs) <= top_n:
        return {"final_docs": docs}

    try:
        final_docs = await async_rerank_contexts(
            query, docs, mm.reranker_model, top_n
        )
    except Exception:
        final_docs = docs[:top_n]

    return {"final_docs": final_docs}


# --- Logic & Graph Wiring ---

def decide_next_step(state: DiagnoseState):
    """The Router Logic"""
    score = state.confidence_score
    docs = state.final_docs
    # High confidence OR already ran RAG loop → done
    if score >= 0.9:
        return "finish"

    if docs:
        return "finish"

    return "retrieve"


# Graph Construction
diagnostic_workflow = StateGraph(DiagnoseState, input=DiagnoseInputState, output=DiagnoseOutputState)

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
__all__ = ["diagnose_app", "DiagnoseState", "DiagnoseInputState", "DiagnoseOutputState"]