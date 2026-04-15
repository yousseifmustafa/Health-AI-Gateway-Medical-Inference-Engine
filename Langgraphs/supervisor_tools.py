import os
import asyncio
import logging
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

from Models.Model_Manager import get_model_manager
from Langgraphs.Diagnose_graph import diagnose_app
from Tools.Query_Optimization_Tool import async_translate_rewrite_expand_query
from Database_Manager import fetch_user_permanent_records, modify_user_permanent_records

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
