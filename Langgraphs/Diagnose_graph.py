from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END, START
import asyncio 
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from Models.Model_Manager import ModelManager
    from Helper.HF_ApiManager import hf_ApiKeyManager
    from Helper.Google_ApiManger import google_ApiKeyManager 
    from Helper.Groq_ApiManger import groq_ApiKeyManager
    from vector_db.VDB_Conection import create_retriever
    from Tools.reranker_tool import rerank_contexts
    from Tools.Query_Optimization_Tool import translate_rewrite_expand_query
    from Tools.Post_validation_tool import Post_Query_Validate
    from Tools.create_final_prompt_tool import create_final_prompt
    from Tools.Summary_tool import summarize_conversation
    from Tools.parallel_retrievs_tool import parallel_retrieval
except ImportError as e:
    print(f"FATAL ERROR: Could not import helper functions: {e}")
    exit()

# Initialization
HF_key_manager = hf_ApiKeyManager()
Google_key_manger = google_ApiKeyManager()
groq_keyManger = groq_ApiKeyManager()
model_manager = ModelManager(HF_key_manager=HF_key_manager, Google_key_manger=Google_key_manger , groq_keyManger = groq_keyManger)
context_retriever = create_retriever(model_manager.embedding_model)


# --- 1. State Definition ---
class DiagnoseState(TypedDict):
    user_query: str
    conversation_summary: str
    
    translated_query: str
    expanded_queries: List[str]
    
    # Unified Answer Field
    english_medical_answer: str     
    confidence_score: float         
    
    # RAG Fields
    support_docs: List[Any]
    support_contents: List[str]
    final_docs: List[Any]  # وجود بيانات هنا هو العلامة إننا في اللفة التانية
    
    final_answer: Optional[str]
    new_summary: str


# --- 2. Helper Models ---
class GradeOutput(BaseModel):
    score: float = Field(description="Confidence score between 0.0 and 1.0 regarding medical accuracy and safety.")


# --- 3. Nodes ---

async def optimize_query_node(state: DiagnoseState):
    """Node 1: Translate and Expand Query"""
    user_query = state["user_query"]
    try:
        translated, expanded = await asyncio.to_thread(
            translate_rewrite_expand_query,
            model_manager.optimize_query, 
            user_query
        )
    except Exception as e:
        print(f"Optimization Failed: {e}")
        translated = user_query
        expanded = [user_query]
        
    return {"translated_query": translated, "expanded_queries": expanded}


async def generate_node(state: DiagnoseState):
    """
    Node 2 (Unified Generator):
    Handles both Zero-Shot (Fast) and RAG (Context) generation automatically.
    """
    query = state["translated_query"]
    summary = state["conversation_summary"]
    docs = state.get("final_docs", []) # List is empty if Fast Path
    
    try:
        prompt_messages = create_final_prompt(query, docs, summary)
        final_input = prompt_messages if isinstance(prompt_messages, str) else prompt_messages[1]['content']     
        answer = await asyncio.to_thread(model_manager.generate_answer, final_input)
            
    except Exception as e:
        print(f"Generator Error: {e}")
        answer = "I cannot generate an answer at this moment."
        
    return {"english_medical_answer": answer}

async def grade_answer_node(state: DiagnoseState):
    """Node 3: The Judge - Evaluate Confidence"""
    query = state["translated_query"]
    answer = state["english_medical_answer"]
    
    try:

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=Google_key_manger.get_next_api_key(),
            temperature=0
        )
        grader_llm = llm.with_structured_output(GradeOutput)
        system_msg = """You are a strict medical supervisor. Rate the answer based on:
        1. Relevance to the query.
        2. Medical Safety (no dangerous hallucinations).
        3. Completeness.
        Give a score < 0.8 if the answer is vague, says "I don't know", or needs external verification.
        """
        
        grade = await grader_llm.ainvoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=f"Query: {query}\nAnswer: {answer}")
        ])
        score = grade.score
        print(f"Grade Score is ={score}\n ")
    except Exception as e:
        print(f"Grading Failed: {e}")
        score = 0.0 # لو فشل التقييم، اعتبره صفر عشان يروح يجيب مراجع
        
    return {"confidence_score": score}


async def retrieve_node(state: DiagnoseState):
    """Node 4: Retrieve Documents (Only runs if needed)"""
    expanded_queries = state["expanded_queries"]
    try:
        docs = await parallel_retrieval(context_retriever, expanded_queries)
    except:
        docs = []    
    contents = [doc.page_content for doc in docs if hasattr(doc, 'page_content')]
    return {"support_docs": docs, "support_contents": contents}


async def rerank_node(state: DiagnoseState):
    """Node 5: Filter Documents & Flag Context Presence"""
    query = state["translated_query"]
    docs = state.get("support_contents", []) or state.get("support_docs", [])
    
    try:
        final_docs = await asyncio.to_thread(
            rerank_contexts, query, docs, model_manager.reranker_Model
        )
    except:
        final_docs = state.get("support_docs", [])
        
    return {"final_docs": final_docs}




async def validate_node(state: DiagnoseState):
    """Validate The Query And Optimize Retrieved answer"""
    user_query = state["user_query"]
    candidate_answer = state["english_medical_answer"]
    
    try:
        final_answer = await asyncio.to_thread(
            Post_Query_Validate,
            model_manager.validate_answer,
            user_query,
            candidate_answer
            )
    except Exception as e:
        final_answer = candidate_answer # Fallback
        
    return {"final_answer": final_answer}


async def summarize_node(state: DiagnoseState):
    """Node 7: Update Memory"""
    try:
        new_summary = await asyncio.to_thread(
            summarize_conversation,
            model_manager.summarize,
            state["user_query"],
            state["final_answer"],
            state["conversation_summary"]
        )
    except:
        new_summary = state["conversation_summary"]
    return {"new_summary": new_summary}


# --- 4. Logic & Graph Wiring (The Loop) ---

def decide_next_step(state: DiagnoseState):
    """The Router Logic"""
    score = state.get("confidence_score", 0.0)
    docs = state.get("final_docs", [])
    
    if score >= 0.8:
        return "validate"
    
    if docs:
        return "validate"

    return "retrieve"

# Graph Construction
diagnostic_workflow = StateGraph(DiagnoseState)

# Add Nodes
diagnostic_workflow.add_node("optimize", optimize_query_node)
diagnostic_workflow.add_node("generate", generate_node) # Unified Node
diagnostic_workflow.add_node("grade", grade_answer_node)
diagnostic_workflow.add_node("retrieve", retrieve_node)
diagnostic_workflow.add_node("rerank", rerank_node)
diagnostic_workflow.add_node("validate", validate_node)
diagnostic_workflow.add_node("summarize", summarize_node)

# Build Edges (The Flow)
diagnostic_workflow.add_edge(START, "optimize")
diagnostic_workflow.add_edge("optimize", "generate") # First pass (No docs)
diagnostic_workflow.add_edge("generate", "grade")    # Check quality

# The Conditional Loop
diagnostic_workflow.add_conditional_edges(
    "grade",
    decide_next_step,
    {
        "validate": "validate", # Success -> Finish
        "retrieve": "retrieve"  # Failure -> Start RAG Loop
    }
)

# The RAG Loop Back
diagnostic_workflow.add_edge("retrieve", "rerank")
diagnostic_workflow.add_edge("rerank", "generate") # <--- THE LOOP: Back to Generator!

# Finalization
diagnostic_workflow.add_edge("validate", "summarize")
diagnostic_workflow.add_edge("summarize", END)

# Compile
diagnose_app = diagnostic_workflow.compile()
__all__ = ["diagnose_app", "DiagnoseState"]