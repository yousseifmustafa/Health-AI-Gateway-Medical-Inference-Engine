import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
try:
    from Langgraphs.supervisor_graph import main_app
except ImportError as e:
    print(f" FATAL IMPORT ERROR: {e}")

app = FastAPI(title="SehaTech API", version="1.0")

STATUS_MAPPING = {
    "consult_doctor_tool": "جاري استشارة الطبيب المختص...",
    "web_search_tool": "جاري البحث في المصادر الطبية...",
    "analyze_medical_image_tool": "جاري تحليل الأشعة/الصورة...",
    "notify_family_tool": "جاري تحضير تنبيه الطوارئ...",
    "optimize": "تحليل الأعراض وصياغة الحالة...",
    "retrieve": "البحث في المراجع الطبية...",
    "generate": "صياغة التشخيص النهائي...",
    "validate": "مراجعة الدقة الطبية...",
}

async def run_full_system_stream(inputs, config):    
    final_answer_accumulator = ""
    
    try:
        async for event in main_app.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            name = event["name"]
            data = event["data"]

            if kind == "on_chain_start" or kind == "on_tool_start":
                if name in STATUS_MAPPING:
                    yield json.dumps({
                        "type": "status",
                        "node": name,
                        "content": STATUS_MAPPING[name]
                    }) + "\n"

            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and chunk.content:
                    final_answer_accumulator += chunk.content
                    yield json.dumps({
                        "type": "token",
                        "content": chunk.content
                    }) + "\n"
            
    except Exception as e:
        print(f"Stream Error: {e}")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    finally:
        state_snapshot = main_app.get_state(config)
        
        if state_snapshot.next and "human_review" in state_snapshot.next:
             yield json.dumps({
                "type": "action_required", 
                "content": "System paused for approval",
                "tool": "notify_family_tool"
            }) + "\n"
        
        else:
            final_values = state_snapshot.values
            final_summary = final_values.get("conversation_summary", "")
            
            if not final_answer_accumulator and final_values.get("messages"):
                last_msg = final_values["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    final_answer_accumulator = last_msg.content
            
            yield json.dumps({
                "type": "final",
                "final_answer": final_answer_accumulator,
                "summary": final_summary
            }) + "\n"

@app.post("/chat")
async def chat_endpoint(
    query: str = Form(...),
    summary: str = Form("No Summary Found."),
    thread_id: str = Form("1"),
    image: UploadFile = File(None),
):
    image_bytes = None
    if image:
        image_bytes = await image.read()

    final_query = query
    if image_bytes:
        final_query += "\n\n[SYSTEM: User uploaded an image. Use 'analyze_medical_image_tool' NOW.]"

    inputs = {
        "messages": [HumanMessage(content=final_query)],
        "image_bytes": image_bytes,   
        "conversation_summary": summary,
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    return StreamingResponse(
        run_full_system_stream(inputs, config), 
        media_type="text/event-stream" # مهم عشان المتصفح يفهم إنه ستريم
    )