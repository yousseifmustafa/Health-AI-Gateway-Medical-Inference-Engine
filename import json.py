import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
try:
    from Langgraphs.supervisor_graph import main_app
    from Models.Model_Manager import ModelManager
    from Helper.HF_ApiManager import hf_ApiKeyManager
    from Helper.Groq_ApiManger import groq_ApiKeyManager
    from Helper.Google_ApiManger import google_ApiKeyManager 
except ImportError as e:
    print(f"Import Error: {e}")
    pass

app = FastAPI(title="SehaTech API", version="1.0")
HF_key_manager = hf_ApiKeyManager()
Google_key_manger = google_ApiKeyManager()
groq_ApiKeyManager = groq_ApiKeyManager()
model_manager = ModelManager(HF_key_manager=HF_key_manager, Google_key_manger=Google_key_manger,groq_keyManger = groq_ApiKeyManager)
finalAnswer= None
summary = None


STATUS_MAPPING = {
    # --- Tools Execution ---
    "consult_doctor_tool": "Consulting with the medical specialist...",
    "web_search_tool": "Searching Web...",
    "speech_to_text_tool": "Processing voice input...",
    "analyze_medical_image_tool": "Analyzing uploaded medical image...",
    "notify_family_tool": "Preparing emergency family alert...",

    # --- RAG Pipeline (The "Thinking" Phase) ---
    "optimize": "Analyzing symptom details...",
    "retrieve": "Retrieving relevant medical references...",
    "rerank": "Filtering results for relevance...",
    
    # --- Generation & Safety ---
    "generate": "Formulating the clinical response...",
    "validate": "Verifying medical accuracy and safety...",
    
    # --- System Operations ---
    "upload": "Securely uploading data...",
    "analyze": "Processing input data..."
}


async def run_full_system_stream(inputs):    
    final_answer_accumulator = ""
    thread_id = inputs.get("thread_id", "1")
    config = {"configurable": {"thread_id": thread_id}}
    
    is_streaming_content = False 

    try:
        async for event in main_app.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            name = event["name"]
    
            if kind in ["on_chain_start", "on_tool_start"]:
                if name in STATUS_MAPPING:
                    yield json.dumps({
                        "type": "status",
                        "node": name,
                        "content": STATUS_MAPPING[name]
                    }) + "\n"

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    final_answer_accumulator += chunk.content
                    yield json.dumps({
                        "type": "token",
                        "content": chunk.content
                    }) + "\n"
            
    except Exception as e:
        print(f" Stream Error: {e}")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    finally:
        current_state = main_app.get_state(config)
        
        if current_state.next and "human_review" in current_state.next:
            yield json.dumps({
                "type": "action_required", 
                "content": "System paused for approval",
                "tool": "notify_family_tool"
            }) + "\n"
        
        else:
            final_summary = current_state.values.get("conversation_summary", "")
            
            if not final_answer_accumulator and current_state.values.get("messages"):
                last_msg = current_state.values["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    final_answer_accumulator = last_msg.content
            yield json.dumps({
                "type": "final",
                "final_answer": final_answer_accumulator,
                "summary": final_summary
            }) + "\n"

@app.get("/")
def root():
    return {"status": "SehaTech System is Online "}

@app.post("/chat")
async def chat_endpoint(
    query: str = Form(...),
    summary: str = Form("No Summary Found."),
    thread_id: str = "1", 
    image: UploadFile = File(None),
):
    image_bytes = None
    
    if image:
        try:
            image_bytes = await image.read()
        except Exception as e:
            print(f" Image Error: {e}")
            raise HTTPException(status_code=500, detail=f"Error handling image: {e}")

    
    final_query = query
    
    if image_bytes:
        final_query += "\n\n[SYSTEM NOTICE: The user has uploaded a medical image. You MUST use the 'analyze_medical_image_tool' to analyze it.]"
    
    inputs = {
        "messages": [HumanMessage(content=final_query)],
        "image_bytes": image_bytes,   
        "thread_id": thread_id,
        "conversation_summary": summary,
    }

    
    return StreamingResponse(
        run_full_system_stream(inputs), 
        media_type="text/event-stream"
    )
