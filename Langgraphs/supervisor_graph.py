import os
import json
import asyncio
import logging

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, RemoveMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

from Langgraphs.states import SupervisorInputState, SupervisorOutputState, AgentState
from Langgraphs.prompts import SUPERVISOR_SYSTEM_PROMPT, SUMMARIZE_PROMPT
from Langgraphs.utils import _detect_conflicts

try:
    from Models.Model_Manager import get_model_manager
    from Langgraphs.Diagnose_graph import diagnose_app
    from Tools.Query_Optimization_Tool import async_translate_rewrite_expand_query
    from Database_Manager import get_checkpointer, fetch_user_permanent_records, modify_user_permanent_records
except ImportError as e:
    logging.getLogger("sehatech.supervisor").critical("Failed to import required modules: %s", e)
    raise SystemExit(1)

logger = logging.getLogger("sehatech.supervisor")

from Langgraphs.supervisor_tools import (
    consult_doctor_tool,
    analyze_medical_image_tool,
    web_search_tool,
    notify_family_tool,
    fetch_patient_records_tool,
    modify_patient_records_tool,
)


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
    sys_msg = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)

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


# CONVERSATION SUMMARIZATION NODE — Long-Term Memory Management
_SUMMARY_THRESHOLD = 4  # Trigger summarization when messages exceed this count

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
    prompt = SUMMARIZE_PROMPT.format(
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