import json
from fastapi import APIRouter, Form, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from Database_Manager import get_or_create_thread
from Helper.Image_Uploader import async_upload_to_cloudinary
from Server.config import logger, limiter, MAX_QUERY_LENGTH, _USER_ID_RE
from Server.prompts import STATUS_MAPPING

router = APIRouter()

async def run_full_system_stream(inputs: dict, config: dict, app_graph):
    """Executes the LangGraph app and yields SSE-formatted JSON events."""
    final_answer_accumulator = ""

    try:
        async for event in app_graph.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            name = event["name"]
            data = event["data"]

            if kind in ("on_chain_start", "on_tool_start"):
                if name in STATUS_MAPPING:
                    yield json.dumps({
                        "type": "status",
                        "node": name,
                        "content": STATUS_MAPPING[name]
                    }) + "\n"

            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    final_answer_accumulator += content
                    yield json.dumps({
                        "type": "token",
                        "content": content
                    }) + "\n"

    except Exception as e:
        logger.exception("Stream error during graph execution.")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    finally:
        try:
            state_snapshot = await app_graph.aget_state(config)

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
                        content = last_msg.content
                        if isinstance(content, list):
                            content = "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in content
                            )
                        final_answer_accumulator = content

                yield json.dumps({
                    "type": "final",
                    "final_answer": final_answer_accumulator,
                    "summary": final_summary
                }) + "\n"
        except Exception as e:
            logger.warning("Failed to retrieve final state: %s", e)


@router.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    query: str = Form(...),
    user_id: str = Form(...),
    summary: str = Form("No Summary Found."),
    thread_id: str = Form("1"),
    image: UploadFile = File(None),
):
    app_graph = getattr(request.app.state, "app_graph", None)
    if app_graph is None:
        raise HTTPException(status_code=503, detail="Service unavailable — graph not compiled.")

    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.",
        )

    if not _USER_ID_RE.match(user_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid user_id format. Must be alphanumeric, hyphens, or underscores (1-255 chars).",
        )

    dynamic_thread_id = await get_or_create_thread(user_id)
    
    logger.info("Chat request | user_id=%s | thread=%s (dynamic=%s) | query_len=%d",
                user_id, thread_id, dynamic_thread_id, len(query))

    image_url = None
    if image:
        image_bytes = await image.read()
        logger.info("Image uploaded | user_id=%s | size=%d bytes", user_id, image.size)
        image_url = await async_upload_to_cloudinary(image_bytes)
        if not image_url:
            raise HTTPException(status_code=500, detail="Image upload to Cloudinary failed.")

    if image_url:
        human_message = HumanMessage(content=[
            {"type": "text",      "text": query},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])
        logger.info("Multimodal HumanMessage built | image_url=%s", image_url)
    else:
        human_message = HumanMessage(content=query)

    inputs = {
        "messages": [human_message],
        "image_url": image_url,
        "conversation_summary": summary,
        "user_id": user_id,
    }

    config = {
        "configurable": {
            "thread_id": dynamic_thread_id
        },
        "recursion_limit": 25,
    }

    return StreamingResponse(
        run_full_system_stream(inputs, config, app_graph),
        media_type="text/event-stream",
    )
