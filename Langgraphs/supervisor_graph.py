from typing import Annotated, TypedDict, List
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GoogleSearchAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults
try:
    from Models.Model_Manager import ModelManager
    from Helper.HF_ApiManager import hf_ApiKeyManager
    from Helper.Groq_ApiManger import groq_ApiKeyManager
    from Helper.Google_ApiManger import google_ApiKeyManager 
    from Langgraphs.Diagnose_graph import diagnose_app
except ImportError:
    pass

load_dotenv()

# Initialization
HF_key_manager = hf_ApiKeyManager()
Google_key_manger = google_ApiKeyManager()
groq_keyManger= groq_ApiKeyManager()
model_manager = ModelManager(HF_key_manager=HF_key_manager, Google_key_manger=Google_key_manger , groq_keyManger = groq_keyManger)


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
    inputs = {
        "user_query": symptom_description,
        "conversation_summary": medical_history,
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
    image_bytes = state.get("image_bytes")
    
    # Strict Check: No image, no service.
    if not image_bytes: return "Error: No image data found in the current state."

    try:
        ocr_response = model_manager.generate_with_image(text=query, image_bytes=image_bytes)
        return {"answer": ocr_response}
    except Exception as e:
        return f"Vision Analysis Failed: {str(e)}"


@tool
def web_search_tool(query: str):
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

        print("🚀 Using Tavily Search (High Precision)...")
        search = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=True 
        )
        results = search.invoke(query)
        return f"[Source: Tavily API]\n{str(results)}"

    except Exception as e:
        print(f" Tavily Failed or Limit Reached: {e}")
        
        try:
            search = GoogleSearchAPIWrapper(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                google_cse_id=os.getenv("GOOGLE_CSE_ID")
            )
            return f"[Source: Google API]\n{search.run(query)}"
            
        except Exception as google_e:
            return f"Web Search Failed completely. Google Error: {str(google_e)}"
        
@tool
def notify_family_tool(message: str, urgency_level: str = "High"):
    """
    Function:
    Triggers an emergency alert system to notify the patient's family members via SMS or Automated Call.

    ⚠️ STRICT USAGE PROTOCOL:
    1. **Direct Request:** If the user explicitly asks (e.g., "Tell my family", "Call home"), you MUST trigger this tool **IMMEDIATELY**.
       -  DO NOT ask "Are you sure?".
       -  DO NOT wait for further confirmation text.
       - The system has an internal approval mechanism; your job is ONLY to initiate the trigger.

    2. **High-Risk Suggestion:** If you proposed this action due to severe symptoms and the user agreed (said "Yes"), trigger this tool **IMMEDIATELY**.

    Inputs:
    - message: Brief content of the alert (e.g., "Patient reporting severe chest pain").
    - urgency_level: "High" for emergencies, "Medium" for updates.
    """
    print(f"\n [SIMULATION] SENDING ALERT TO FAMILY")
    print(f" Message: {message}")
    print(f" Urgency: {urgency_level}")
    print(f" Status: Sent Successfully via Mock Gateway\n")    
    return "Success: Family notification sent successfully. The family has been alerted and will contact the patient shortly."


# Agent Logic
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    image_bytes: bytes | None 
    conversation_summary: str
    
async def agent_node(state: AgentState):
    messages = state["messages"]
    try:
        token = Google_key_manger.get_next_api_key()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=token,
            temperature=0.3, 
            streaming=True
        )
    except Exception as e:        
        return {"messages": [AIMessage(content=str(e))]}
    tools = [consult_doctor_tool, web_search_tool, notify_family_tool]
    
    if state.get("image_bytes"):
        tools.append(analyze_medical_image_tool)

    llm_with_tools = llm.bind_tools(tools)
   
    sys_msg = SystemMessage(content="""
    You are 'SehaTech AI', an intelligent and professional medical triage assistant.
    Your core mission is to construct a comprehensive "Clinical Picture", forward it to the specialized Doctor Agent, and then relay the diagnosis to the user.

    **STRICT OPERATIONAL PROTOCOLS:**

    1. **THE GOLDEN TRIANGLE (Top Priority):**
    - Before proceeding to any diagnosis, you MUST ensure the following 4 information points are present:
        1. **Chief Complaint** (What is the main pain/issue?).
        2. **Duration** (How long has it been going on?).
        3. **Severity/Description** (Intensity, nature of pain, etc.).
        4. **Medical History** (Chronic conditions like Diabetes, Hypertension, or current meds).
    - **Action:** If ANY of these points are missing, you MUST ask the user for them first. DO NOT trigger the doctor tool yet.

    2. **DIAGNOSIS PHASE:**
    - Once the "Golden Triangle" is complete, trigger the `consult_doctor_tool` immediately.
    - **Relay Rule:** When the tool returns the medical response, you MUST display it to the user.
    - **Prefix:** Start your response with: "جاري تحليل البيانات بدقة... (قد يستغرق الأمر لحظات لمراجعة المراجع الطبية). : ..."

    3. **EMERGENCY NOTIFICATION (`notify_family_tool`) - HIGH SENSITIVITY:**
    - **PROHIBITION:** You are strictly forbidden from using this tool autonomously without a trigger.
    - **CASE A (User Request):** If the user explicitly asks (e.g., "Tell my family", "Call my parents"), you MUST trigger the tool **IMMEDIATELY**.
        - DO NOT ask "Are you sure?".
        - DO NOT wait for confirmation.
    - **CASE B (Agent Suggestion):** If you detect critical symptoms or the user appears distressed/alone, you MAY **propose** help in the chat:
        - Say (in friendly language): "شكلك تعبان أوي، تحب أبعت رسالة لعيلتك يطمنوا عليك؟"
        - If User says "No": Drop the topic and proceed with triage.
        - If User says "Yes/Agree": Trigger the tool **IMMEDIATELY**.

    4. **MEMORY:**
    - Never ask for information the user has already mentioned in the conversation history.

    5. **SMART LANGUAGE ADAPTATION (The "Sya'a" Layer):**
    - **Goal:** You are the bridge between complex/foreign data and the simple user.
    - **Rule:** If ANY tool (Doctor, Web Search, Vision) returns information in **English** or **Formal Arabic (MSA)**, you MUST **translate and adapt** it into **Friendly Language exactly matching the user's query language** before outputting.
    - **Restriction:** Do NOT output large blocks of English text unless the user is speaking English.
    - **Exception:** You may keep specific **Drug Names** or **Medical Terms** in English, but explain them in the **Same User Language**.

    6. **TONE & PERSONALITY:**
    - Be warm, empathetic, and reassuring (e.g., "سلامتك يا بطل", "Don't worry").
    """)
    all_msgs = [sys_msg] + messages
    
    response = await llm_with_tools.ainvoke(all_msgs)
    
    return {"messages": [response]}


# Workflow
tool_node = ToolNode([notify_family_tool , consult_doctor_tool, analyze_medical_image_tool, web_search_tool])
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

workflow.add_edge("tools", "agent")
memory = MemorySaver()
main_app = workflow.compile(checkpointer=memory)
__all__ = ["main_app", "AgentState"]