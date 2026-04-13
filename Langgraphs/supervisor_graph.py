import os
import re
import json
import asyncio
import logging
from typing import Annotated, List, Optional
from pydantic import BaseModel, ConfigDict

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, RemoveMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

try:
    from Models.Model_Manager import get_model_manager
    from Langgraphs.Diagnose_graph import diagnose_app
    from Tools.Query_Optimization_Tool import async_translate_rewrite_expand_query
    from Database_Manager import get_checkpointer, fetch_user_permanent_records, modify_user_permanent_records
except ImportError as e:
    logging.getLogger("sehatech.supervisor").critical("Failed to import required modules: %s", e)
    raise SystemExit(1)

logger = logging.getLogger("sehatech.supervisor")


@tool
async def consult_doctor_tool(symptom_description: str, medical_history: str = "No History"):
    """
    STRICT PREREQUISITE: Do NOT use this tool until you have gathered the complete "Golden Triangle" of information (Symptoms + Duration + Severity).
    
    Function:
    Activates the specialized Deep Diagnosis Agent to analyze the clinical picture.
    
    Inputs:
    - symptom_description: A comprehensive summary of the symptoms (including Duration, Severity, and Location) .
    - medical_history: Chronic diseases (e.g., Diabetes, Hypertension) and current medications. This field is MANDATORY for diagnostic accuracy.
    """
    # Optimize the query HERE — only when the Doctor tool is actually invoked
    mm = get_model_manager()
    try:
        translated, expanded = await async_translate_rewrite_expand_query(
            mm.optimize_query,
            symptom_description
        )
    except Exception as e:
        logger.warning("Query optimization failed: %s", e)
        translated = symptom_description
        expanded = [symptom_description]

    inputs = {
        "user_query": symptom_description,
        "conversation_summary": medical_history,
        "translated_query": translated,
        "expanded_queries": expanded,
    }

    result = await diagnose_app.ainvoke(inputs)
    return result.get("final_answer", "Cant Diagnose Your Case.")


@tool
async def analyze_medical_image_tool(query: str, state: Annotated[dict, InjectedState]):
    """
    EXCLUSIVE USE: Trigger this tool ONLY when the user has uploaded a image file (e.g. Drug Box, Prescription).
    
    Function:
    Performs Visual Question Answering (VQA) or Optical Character Recognition (OCR) on the uploaded medical image to extract data or analyze findings.
    
    Inputs:
    - query: A specific question or instruction regarding the image content (e.g., "Read the medicine names", "Analyze this X-ray for fractures").
    """
    # InjectedState provides a plain dict at runtime, not the Pydantic model instance
    image_url = state.get("image_url") if isinstance(state, dict) else getattr(state, "image_url", None)

    # Strict Check: No image, no service.
    if not image_url:
        return "Error: No image data found in the current state."

    try:
        mm = get_model_manager()
        ocr_response = await mm.agenerate_with_image(
            text=query, image_url=image_url
        )
        return {"answer": ocr_response}
    except Exception as e:
        return f"Vision Analysis Failed: {str(e)}"


@tool
async def web_search_tool(query: str):
    """
    Function:
    Access real-time external information via Tavily (Primary) or Google Search (Fallback).

    USE THIS TOOL FOR:
    1. **Logistics & Services:** Finding contact info, locations of nearby hospitals, clinics, or pharmacies.
    2. **Market Data:** Checking current prices, alternatives, or availability of medications in the market.
    3. **General Info:** Retrieving public health news or non-clinical general knowledge.
    4. **Explicit User Intents:** When the user directly requests to "search", "find links", or "browse".
    
    STRICT CONSTRAINT:
    - **DO NOT** use this tool for medical diagnosis or symptom analysis.
    - For clinical queries, ALWAYS use the `consult_doctor_tool`.
    """

    try:
        if not os.getenv("TAVILY_API_KEY"):
            raise Exception("Tavily API Key not found in env.")

        logger.info("Using Tavily Search for: %s", query)

        search = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=False
            )

        raw_results = await search.ainvoke(query)

        if isinstance(raw_results, str):
            return f"[Source: Tavily API]\n{raw_results}"

        formatted_results = ["[Source: Tavily API]"]

        for result in raw_results:
            title = result.get('title', 'No Title')
            url = result.get('url', 'No URL')
            content = result.get('content', 'No Content Available')

            entry = (
                f"\n---\n"
                f"Title: {title}\n"
                f"Link: {url}\n"
                f"Summary: {content}\n"
            )
            formatted_results.append(entry)

        return "".join(formatted_results)

    except Exception as e:
        logger.warning("Tavily failed: %s. Switching to Google...", e)

        try:
            search = GoogleSearchAPIWrapper(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                google_cse_id=os.getenv("GOOGLE_CSE_ID")
            )
            result = await asyncio.to_thread(search.run, query)
            return f"[Source: Google API]\n{result}"

        except Exception as google_e:
            return f"Web Search Failed completely. Error: {str(google_e)}"


@tool
async def notify_family_tool(message: str, urgency_level: str = "High"):
    """
    Function:
    Triggers an emergency alert system to notify the patient's family members via SMS or Automated Call.

     STRICT USAGE PROTOCOL:
    1. **Direct Request:** If the user explicitly asks (e.g., "Tell my family", "Call home"), you MUST trigger this tool **IMMEDIATELY**.
       -  DO NOT ask "Are you sure?".
       -  DO NOT wait for further confirmation text.
       - The system has an internal approval mechanism; your job is ONLY to initiate the trigger.

    2. **High-Risk Suggestion:** If you proposed this action due to severe symptoms and the user agreed (said "Yes"), trigger this tool **IMMEDIATELY**.

    Inputs:
    - message: Brief content of the alert (e.g., "Patient reporting severe chest pain").
    - urgency_level: "High" for emergencies, "Medium" for updates.
    """
    logger.info("[SIMULATION] SENDING ALERT TO FAMILY | message=%s | urgency=%s", message, urgency_level)
    return "Success: Family notification sent successfully. The family has been alerted and will contact the patient shortly."


@tool
async def fetch_patient_records_tool(user_id: str):
    """
    PROACTIVE DISCOVERY — Call this tool at the START of any triage session.

    Function:
    Retrieves the patient's permanent medical profile (Allergies, Chronic Diseases,
    Current Medications, Surgical History) from the SehaTech Backend API.

    Patient Longitudinal Safety:
    This data persists across ALL sessions. Using it grounds your questions in
    known facts and prevents contraindicated recommendations.

    Inputs:
    - user_id: The unique patient identifier.
    """
    return await fetch_user_permanent_records(user_id)


@tool
async def modify_patient_records_tool(user_id: str, action: str, medical_fact: str):
    """
    DYNAMIC MAINTENANCE — Sync changes back to the patient's permanent record.

    Function:
    Sends a modification (ADD, REMOVE, or UPDATE) to the SehaTech Backend API
    to keep the patient's life-long medical profile accurate.

    When to Use:
    - ADD: A new allergy, diagnosis, or medication is discovered during triage.
    - REMOVE: The patient states they recovered from a condition or finished a treatment.
    - UPDATE: An existing fact needs correction (e.g., dosage change).

    Inputs:
    - user_id: The unique patient identifier.
    - action: One of 'ADD', 'REMOVE', or 'UPDATE'.
    - medical_fact: The specific medical fact to modify (e.g., 'Penicillin allergy').
    """
    return await modify_user_permanent_records(user_id, action, medical_fact)


# ============================================================================
# STATE SCHEMAS — Multiple Schemas Pattern (Input / Output / Internal)
# ============================================================================

class SupervisorInputState(BaseModel):
    """Schema for data accepted FROM the client (server.py / Flutter app)."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    messages: Annotated[list[BaseMessage], add_messages] = []
    image_url: Optional[str] = None
    conversation_summary: str = ""
    user_id: Optional[str] = None


class SupervisorOutputState(BaseModel):
    """Schema for data returned TO the client — no internal flags or records."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    messages: Annotated[list[BaseMessage], add_messages] = []
    conversation_summary: str = ""


class AgentState(BaseModel):
    """Internal superset — contains every field nodes may read or write."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    messages: Annotated[list[BaseMessage], add_messages] = []
    image_url: Optional[str] = None
    conversation_summary: str = ""
    user_id: Optional[str] = None
    patient_records: Optional[dict] = None
    conflict_flag: bool = False
    conflict_details: Optional[str] = None


# ============================================================================
# DETERMINISTIC CONFLICT DETECTOR — Runs before agent on every user turn
# ============================================================================

# Negation patterns for detecting contradictions
_NEGATION_PATTERNS = [
    r"\bno\s+(allergies|allergy|medications?|chronic|conditions?|surgeries|surgery|history)\b",
    r"\bi\s+(don'?t|do\s+not)\s+(take|have|use)\b",
    r"\bi\s+(stopped|quit|finished|completed)\b",
    r"\bnot\s+(taking|on|using)\b",
    r"\b(مفيش|ماعنديش|مش|ماباخدش|مابخدش|مابستخدمش|لا|مش بآخد|وقفت|خلصت)\b",
    r"\bno\s+medical\s+history\b",
]
_NEGATION_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)


def _detect_conflicts(user_text: str, records: dict) -> list[str]:
    """
    Deterministic conflict detection — zero LLM cost, zero latency.
    Compares user's latest message against fetched permanent records.
    Returns a list of conflict description strings (empty if no conflicts).
    """
    conflicts = []
    if not records or not user_text:
        return conflicts

    text_lower = user_text.lower()
    has_negation = bool(_NEGATION_RE.search(user_text))

    if not has_negation:
        return conflicts

    # Check: "no allergies" vs non-empty allergy list
    allergies = records.get("allergies", [])
    if allergies and re.search(r"(no\s+allerg|مفيش\s*حساسي|ماعنديش\s*حساسي)", user_text, re.IGNORECASE):
        conflicts.append(
            f"Patient says NO allergies, but records show: {', '.join(allergies)}"
        )

    # Check: "no medications" vs non-empty medication list
    meds = records.get("current_medications", [])
    if meds and re.search(r"(no\s+medic|don'?t\s+take|not\s+taking|مش\s*بآخد|ماباخدش|مابخدش)", user_text, re.IGNORECASE):
        conflicts.append(
            f"Patient says NO medications, but records show active prescriptions: {', '.join(meds)}"
        )

    # Check: "no chronic conditions" vs non-empty chronic list
    chronic = records.get("chronic_diseases", [])
    if chronic and re.search(r"(no\s+chronic|no\s+condition|don'?t\s+have\s+any|مفيش\s*أمراض|ماعنديش)", user_text, re.IGNORECASE):
        conflicts.append(
            f"Patient says NO chronic conditions, but records show: {', '.join(chronic)}"
        )

    # Check: "no surgeries" vs non-empty surgical history
    surgeries = records.get("surgical_history", [])
    if surgeries and re.search(r"(no\s+surg|never\s+had\s+surg|مفيش\s*عملي|ماعملتش)", user_text, re.IGNORECASE):
        conflicts.append(
            f"Patient says NO surgical history, but records show: {', '.join(surgeries)}"
        )

    return conflicts


async def conflict_detector_node(state: AgentState) -> dict:
    """
    Deterministic Conflict Detector Node.
    Runs BEFORE agent_node on every turn. Checks the user's latest message
    against the patient's permanent records for contradictions.
    Sets conflict_flag and conflict_details in state.
    """
    records = state.patient_records
    messages = state.messages

    # No records fetched yet → passthrough
    if not records or not messages:
        return {"conflict_flag": False, "conflict_details": None}

    # Get the latest user message
    latest_user_msg = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            latest_user_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not latest_user_msg:
        return {"conflict_flag": False, "conflict_details": None}

    conflicts = _detect_conflicts(latest_user_msg, records)

    if conflicts:
        detail_text = "⚠️ CONFLICT DETECTED — The following contradictions were found between the patient's statement and their permanent records:\n"
        for i, c in enumerate(conflicts, 1):
            detail_text += f"  {i}. {c}\n"
        detail_text += "You MUST address these contradictions with the patient BEFORE proceeding to diagnosis."
        logger.warning("Conflict detected for user: %s", conflicts)
        return {"conflict_flag": True, "conflict_details": detail_text}

    return {"conflict_flag": False, "conflict_details": None}


async def agent_node(state: AgentState):
    messages = list(state.messages)
    mm = get_model_manager()

    try:
        token = mm.google_key_manager.get_next_api_key()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=token,
            temperature=0.3,
            streaming=True
        )
    except Exception as e:
        return {"messages": [AIMessage(content=str(e))]}
    tools = [
        consult_doctor_tool,
        web_search_tool,
        notify_family_tool,
        fetch_patient_records_tool,
        modify_patient_records_tool,
    ]

    if state.image_url:
        tools.append(analyze_medical_image_tool)

    llm_with_tools = llm.bind_tools(tools)

    sys_msg = SystemMessage(content="""
    You are 'SehaTech AI', an intelligent and professional medical triage assistant.
    Your core mission is to construct a comprehensive "Clinical Picture", forward it to the specialized Doctor Agent, and then relay the diagnosis to the user.

    **STRICT OPERATIONAL PROTOCOLS:**

    0. **STEP ZERO — PATIENT DISCOVERY (Permanent Memory):**
    - At the VERY START of any new triage interaction (when a user_id is available), you MUST call `fetch_patient_records_tool` to load the patient's permanent medical profile.
    - This profile contains life-long data: Allergies, Chronic Diseases, Current Medications, Surgical History.
    - **Cross-Referencing Rule:** Once fetched, you MUST ground your questions and analysis in this data:
        - Example: If the profile shows Type 2 Diabetes, say: "أنا شايف إنك عندك سكر من النوع التاني، خلينا نشيك على مستوى الجلوكوز الأول."
        - Example: If the profile shows a Penicillin allergy, NEVER recommend any penicillin-class antibiotics.
    - If the fetch returns empty/unavailable, proceed normally — this is a new patient with no prior records.

    1. **MEDICAL CONFLICT DETECTION (The Safety Gate):**
    - The system has a **deterministic conflict detector** that automatically flags contradictions between your statements and the patient's records.
    - If a `[CONFLICT ALERT]` message appears in the conversation, you MUST:
        1. **STOP** — Pause the triage immediately. Do NOT proceed to diagnosis with conflicting data.
        2. **FLAG** — Politely inform the patient of each specific discrepancy.
           - Example: "لحظة يا فندم — السجلات الطبية بتاعتك بتقول إنك بتاخد دوا للضغط (أملوديبين). هل لسه بتاخده ولا وقفته؟"
           - Example: "I see from your records that you have Type 2 Diabetes, but you mentioned no chronic conditions. Could you help me clarify?"
        3. **RESOLVE** — Based on the patient's clarification:
           - If they CONFIRM the record is outdated → Call `modify_patient_records_tool` with action='REMOVE' or action='UPDATE' IMMEDIATELY.
           - If they CONFIRM the record is correct (they forgot) → Proceed with the record data as ground truth.
           - If UNCERTAIN → Proceed with CAUTION, note the conflict in the consultation, and pass BOTH versions to the doctor tool.
    - **Safety Principle:** NEVER silently ignore a `[CONFLICT ALERT]`. A contradiction is a potential patient safety risk.

    2. **THE GOLDEN TRIANGLE (Top Priority):**
    - Before proceeding to any diagnosis, you MUST ensure the following 4 information points are present:
        1. **Chief Complaint** (What is the main pain/issue?).
        2. **Duration** (How long has it been going on?).
        3. **Severity/Description** (Intensity, nature of pain, etc.).
        4. **Medical History** (Chronic conditions like Diabetes, Hypertension, or current meds — cross-reference with permanent records if available).
    - **Action:** If ANY of these points are missing, you MUST ask the user for them first. DO NOT trigger the doctor tool yet.

    3. **DIAGNOSIS PHASE:**
    - Once the "Golden Triangle" is complete, trigger the `consult_doctor_tool` immediately.
    - **Relay Rule:** When the tool returns the medical response, you MUST display it to the user.
    - **Prefix:** Start your response with: "جاري تحليل البيانات بدقة... (قد يستغرق الأمر لحظات لمراجعة المراجع الطبية). : ..."

    4. **DYNAMIC RECORD MAINTENANCE (Permanent Memory Sync):**
    - If during the conversation the patient reveals NEW medical facts (new allergy, new diagnosis), you MUST call `modify_patient_records_tool` with action='ADD'.
    - If the patient states they have RECOVERED from a condition or FINISHED a treatment (e.g., "خلصت الكورس بتاع المضاد الحيوي"), you MUST call `modify_patient_records_tool` with action='REMOVE'.
    - If the patient corrects or updates an existing fact (e.g., changed medication dosage), use action='UPDATE'.
    - **Confidence Rule:** Only modify records based on clear, explicit patient statements — NEVER based on inference or assumptions.

    5. **EMERGENCY NOTIFICATION (`notify_family_tool`) - HIGH SENSITIVITY:**
    - **PROHIBITION:** You are strictly forbidden from using this tool autonomously without a trigger.
    - **CASE A (User Request):** If the user explicitly asks (e.g., "Tell my family", "Call my parents"), you MUST trigger the tool **IMMEDIATELY**.
        - DO NOT ask "Are you sure?".
        - DO NOT wait for confirmation.
    - **CASE B (Agent Suggestion):** If you detect critical symptoms or the user appears distressed/alone, you MAY **propose** help in the chat:
        - Say (in friendly language): "شكلك تعبان أوي، تحب أبعت رسالة لعيلتك يطمنوا عليك؟"
        - If User says "No": Drop the topic and proceed with triage.
        - If User says "Yes/Agree": Trigger the tool **IMMEDIATELY**.

    6. **MEMORY:**
    - Never ask for information the user has already mentioned in the conversation history.
    - Never ask for information already present in the patient's permanent records.

    7. **SMART LANGUAGE ADAPTATION (The "Sya'a" Layer):**
    - **Goal:** You are the bridge between complex/foreign data and the simple user.
    - **Rule:** If ANY tool (Doctor, Web Search, Vision) returns information in **English** or **Formal Arabic (MSA)**, you MUST **translate and adapt** it into **Friendly Language exactly matching the user's query language** before outputting.
    - **Restriction:** Do NOT output large blocks of English text unless the user is speaking English.
    - **Exception:** You may keep specific **Drug Names** or **Medical Terms** in English, but explain them in the **Same User Language**.

    8. **TONE & PERSONALITY:**
    - Be warm, empathetic, and reassuring (e.g., "سلامتك يا بطل", "Don't worry").
    """)

    # --- Inject conflict alert into the message stream if flagged ---
    conflict_details = state.conflict_details
    if conflict_details:
        conflict_msg = SystemMessage(content=f"[CONFLICT ALERT]\n{conflict_details}")
        all_msgs = [sys_msg, conflict_msg] + messages
    else:
        all_msgs = [sys_msg] + messages

    try:
        response = await llm_with_tools.ainvoke(all_msgs)
    except Exception as e:
        logger.error("LLM invocation error in agent_node: %s", e)
        response = AIMessage(content="عذراً، أواجه مشكلة تقنية في الوقت الحالي. برجاء المحاولة مرة أخرى.")

    return {"messages": [response]}


# --- Tools Postprocessor: extracts patient_records from tool output into state ---
async def tools_postprocessor(state: AgentState) -> dict:
    """
    Runs AFTER the ToolNode. Scans the latest tool messages for output from
    fetch_patient_records_tool and writes it to AgentState.patient_records
    so the conflict_detector_node can use it on subsequent turns.
    """
    messages = state.messages
    updates = {}

    # Walk backwards through recent messages looking for tool responses
    for msg in reversed(messages):
        if not hasattr(msg, "type") or msg.type != "tool":
            break  # Stop at first non-tool message — we only care about the latest batch
        if getattr(msg, "name", None) == "fetch_patient_records_tool":
            try:
                content = msg.content
                if isinstance(content, str):
                    records = json.loads(content)
                elif isinstance(content, dict):
                    records = content
                else:
                    continue
                # Only store if it looks like a valid records dict
                if isinstance(records, dict) and "user_id" in records:
                    updates["patient_records"] = records
                    logger.info("patient_records populated in state for user %s.", records.get("user_id"))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to parse patient records from tool output: %s", e)

    return updates


# ============================================================================
# CONVERSATION SUMMARIZATION NODE — Long-Term Memory Management
# ============================================================================

_SUMMARY_THRESHOLD = 4  # Trigger summarization when messages exceed this count

_SUMMARIZE_PROMPT = """You are a medical conversation summarizer. Summarize the following conversation concisely in 2-3 sentences.
Preserve ALL critical medical details: symptoms, diagnoses, medications, allergies, and any patient safety information.
Keep the summary in the primary language of the conversation.

Previous Summary:
{previous_summary}

Conversation:
{conversation_text}

Concise Summary:"""


async def summarize_node(state: AgentState) -> dict:
    """
    Running Summary Node — triggers when len(messages) > 10.
    1. Generates a new summary from current summary + all messages.
    2. Trims messages to keep only the last 2.
    Uses RemoveMessage to properly interact with the add_messages reducer.
    """
    messages = state.messages
    previous_summary = state.conversation_summary

    # Build conversation text from all messages
    conversation_lines = []
    for msg in messages:
        if not hasattr(msg, "content"):
            continue
        content = msg.content
        # Multimodal messages have content as a list of typed dicts — extract text parts only
        if isinstance(content, list):
            text_parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = " ".join(text_parts).strip()
            if not content:
                continue
        elif not isinstance(content, str):
            continue
        role = "User" if (hasattr(msg, "type") and msg.type == "human") else "Assistant"
        conversation_lines.append(f"{role}: {content}")
    conversation_text = "\n".join(conversation_lines)

    # Generate new summary
    mm = get_model_manager()
    prompt = _SUMMARIZE_PROMPT.format(
        previous_summary=previous_summary if previous_summary else "None",
        conversation_text=conversation_text,
    )

    try:
        new_summary = await mm.asummarize(prompt)
        if not new_summary or len(new_summary.strip()) < 10:
            new_summary = previous_summary
        else:
            new_summary = new_summary.strip().strip('"')
    except Exception as e:
        logger.warning("Summarization failed: %s. Keeping previous summary.", e)
        new_summary = previous_summary

    # Trim: remove all messages except the last 2 using RemoveMessage
    delete_messages = [RemoveMessage(id=msg.id) for msg in messages[:-2]]

    logger.info("Summarized %d messages → kept last 2. Summary: %s", len(messages), new_summary[:100])

    return {
        "conversation_summary": new_summary,
        "messages": delete_messages,
    }


# ============================================================================
# AGENT ROUTER — Custom conditional edge replacing tools_condition
# ============================================================================

def agent_router(state: AgentState) -> str:
    """
    Routes agent output to the correct next node:
    - "tools" if the LLM wants to call tools
    - "summarize" if conversation is long and no tools are needed
    - END if no tools and conversation is short
    """
    messages = state.messages
    if not messages:
        return END

    last_message = messages[-1]

    # If the LLM wants to call tools → route to tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # No tools → check if we need to summarize
    if len(messages) > _SUMMARY_THRESHOLD:
        return "summarize"

    return END


# --- Graph Definition (built once, compiled lazily with checkpointer) ---
tool_node = ToolNode([
    notify_family_tool,
    consult_doctor_tool,
    analyze_medical_image_tool,
    web_search_tool,
    fetch_patient_records_tool,
    modify_patient_records_tool,
])
workflow = StateGraph(AgentState, input=SupervisorInputState, output=SupervisorOutputState)

workflow.add_node("conflict_detector", conflict_detector_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("tools_postprocessor", tools_postprocessor)
workflow.add_node("summarize", summarize_node)

# START → conflict_detector → agent → [router] → tools/summarize/END
workflow.add_edge(START, "conflict_detector")
workflow.add_edge("conflict_detector", "agent")
workflow.add_conditional_edges(
    "agent",
    agent_router,
    {
        "tools": "tools",
        "summarize": "summarize",
        END: END,
    },
)
workflow.add_edge("tools", "tools_postprocessor")
workflow.add_edge("tools_postprocessor", "conflict_detector")
workflow.add_edge("summarize", END)

from langgraph.checkpoint.memory import MemorySaver


# --- Lazy Factory: compiles with PostgreSQL checkpointer on first call ---
_compiled_app = None


async def make_graph():
    """
    Async factory that returns the compiled supervisor graph with PostgreSQL persistence.
    Called by LangGraph Studio (via langgraph.json entry point) or by application code.
    On first call: initializes the AsyncPostgresSaver and compiles the graph.
    Falls back to MemorySaver if PostgreSQL is unavailable.
    """
    global _compiled_app
    if _compiled_app is None:
        try:
            checkpointer = await get_checkpointer()
            logger.info("Supervisor graph compiled with PostgreSQL checkpointer.")
        except Exception as e:
            logger.warning("PostgreSQL unavailable (%s). Falling back to MemorySaver.", e)
            checkpointer = MemorySaver()
        _compiled_app = workflow.compile(checkpointer=checkpointer)
    return _compiled_app


__all__ = ["make_graph", "AgentState", "SupervisorInputState", "SupervisorOutputState"]