"""
server.py — FastAPI Streaming Endpoint for SehaTech AI.

Production-ready with:
- Async graph factory (make_graph) with lifespan caching
- SSE streaming with Arabic status messages
- /health endpoint for load balancer probes
- Rate limiting (slowapi — 10 req/min per IP)
- Input validation (max 5000 chars on query, user_id format check)
- Structured logging (no print statements)
- Image upload → Cloudinary URL (no raw bytes in state)
"""

import re
import json
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# --- Load .env ONCE at the entry point, before any other imports ---
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from Database_Manager import get_or_create_thread
from Helper.Image_Uploader import async_upload_to_cloudinary
from Models.Model_Manager import get_model_manager

# --- Structured Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sehatech.server")

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)

# --- Cached Compiled Graph ---
_app_graph = None

# --- Input Validation ---
MAX_QUERY_LENGTH = 5000
_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,255}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the LangGraph supervisor once at startup, pre-warm models, and start background tasks."""
    global _app_graph
    try:
        from Langgraphs.supervisor_graph import make_graph
        _app_graph = await make_graph()
        logger.info("Supervisor graph compiled and cached at startup.")
    except Exception as e:
        logger.error("Failed to compile supervisor graph at startup: %s", e)
        _app_graph = None

    # Pre-warm heavy models so first request doesn't pay cold-start penalty
    try:
        from Models.Model_Manager import get_model_manager
        mm = get_model_manager()
        _ = mm.embedding_model  # trigger lazy load
        logger.info("Models pre-warmed: embedding model loaded.")
    except Exception as e:
        logger.warning("Model pre-warming failed (non-fatal): %s", e)

    # Start background WAL drain loop
    try:
        from Database_Manager import start_wal_drain_loop
        start_wal_drain_loop()
    except Exception as e:
        logger.warning("WAL drain loop start failed (non-fatal): %s", e)

    yield

    # --- Graceful shutdown ---
    logger.info("Server shutting down — cleaning up resources...")
    try:
        from Database_Manager import shutdown as db_shutdown
        await db_shutdown()
    except Exception as e:
        logger.warning("Shutdown cleanup error: %s", e)
    logger.info("Server shutdown complete.")


app = FastAPI(lifespan=lifespan, title="SehaTech AI Supervisor")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429, content={"detail": "Rate limit exceeded. Try again in a minute."}
))

STATUS_MAPPING = {
    "conflict_detector_node": "جاري التحقق من تعارض المعلومات...",
    "tools_postprocessor": "جاري تحديث السجل الطبي...",
    "agent_node": "جاري صياغة الرد الطبي...",
    "call_tool": "جاري استدعاء الأدوات الطبية...",
    "summarize": "جاري تلخيص المحادثة...",
}


# ============================================================================
# RESPONSE SCHEMAS — Structured output models for stateless analysis endpoints
# ============================================================================

class MedicationEntry(BaseModel):
    """A single medication line extracted from the prescription."""
    name:     str  = Field(description="The medication name exactly as written (trade or generic).")
    dosage:   str  = Field(description="Dosage per administration (e.g. '500mg', '1 tablet').")
    frequency:str  = Field(description="How often to take (e.g. 'twice daily', 'مرتين يومياً').")
    duration: Optional[str] = Field(None, description="Course length if specified (e.g. '7 days').")
    notes:    Optional[str] = Field(None, description="Any special instruction for this drug (e.g. 'take with food').")


class PrescriptionResponse(BaseModel):
    """Structured data extracted from a handwritten or printed medical prescription."""
    patient_name:      Optional[str]             = Field(None, description="Patient name if legible.")
    patient_date:      Optional[str]             = Field(None, description="Date on the prescription if present.")
    doctor_name:       Optional[str]             = Field(None, description="Prescribing doctor name if legible.")
    medications:       List[MedicationEntry]     = Field(default_factory=list, description="All medications listed.")
    general_notes:     Optional[str]             = Field(None, description="Any free-text doctor instructions not tied to a specific drug.")
    confidence:        str                       = Field(description="Overall OCR confidence: 'HIGH', 'MEDIUM', or 'LOW'.")
    unreadable_parts:  Optional[str]             = Field(None, description="Describe any parts of the image that were too unclear to parse.")


class MedicineBoxResponse(BaseModel):
    """Structured data extracted from a medicine box / packaging image."""
    trade_name:          str            = Field(description="Brand/commercial name printed on the box.")
    generic_name:        Optional[str]  = Field(None, description="INN / active ingredient name.")
    active_ingredients:  List[str]      = Field(default_factory=list, description="List of active chemical compounds with concentrations.")
    concentration:       Optional[str]  = Field(None, description="Strength of the formulation (e.g. '500mg/5ml').")
    dosage_form:         Optional[str]  = Field(None, description="Form type (e.g. tablet, syrup, injection).")
    indications:         List[str]      = Field(default_factory=list, description="Stated medical uses / indications.")
    contraindications:   List[str]      = Field(default_factory=list, description="Stated warnings or contraindications.")
    manufacturer:        Optional[str]  = Field(None, description="Manufacturing company name.")
    storage_conditions:  Optional[str]  = Field(None, description="Storage instructions if visible.")
    expiry_date:         Optional[str]  = Field(None, description="Expiry date if legible.")


# ============================================================================
# SYSTEM PROMPTS — Specialized instructions for each analysis type
# ============================================================================

_PRESCRIPTION_SYSTEM_PROMPT = """\
You are a specialized Medical OCR engine. Your ONLY task is to extract structured data 
from the provided prescription image.

RULES:
1. Read ALL text in the image — printed and handwritten — with maximum accuracy.
2. For each medication line, extract: name, dosage, frequency, duration, and any special notes.
3. If a field is not present or illegible, use null — never guess or fabricate data.
4. Preserve the original language of medication names (do not translate drug names).
5. Set `confidence` to:
   - 'HIGH'   if the entire prescription is clearly legible.
   - 'MEDIUM' if some parts are unclear but most data is extractable.
   - 'LOW'    if the image is blurry, rotated, or mostly illegible.
6. Describe any unreadable sections in `unreadable_parts`.
7. DO NOT add medical advice or commentary — extraction only.
"""

_MEDICINE_BOX_SYSTEM_PROMPT = """\
You are a specialized Pharmaceutical Packaging Analyst. Your ONLY task is to extract 
structured information from the provided medicine box or packaging image.

RULES:
1. Identify the TRADE NAME (brand) prominently displayed on the box.
2. Find the GENERIC/INN name (usually in smaller text or parentheses).
3. List ALL active ingredients with their exact concentrations as printed.
4. Extract indications (uses) and contraindications (warnings) from the packaging.
5. Record the dosage form (tablet, capsule, syrup, injection, etc.).
6. Capture manufacturer, storage conditions, and expiry date if visible.
7. If a field cannot be read from the image, use null — never invent data.
8. DO NOT add medical advice or commentary — extraction only.
"""


# ============================================================================
# SHARED UPLOAD HELPER — DRY image upload logic for stateless endpoints
# ============================================================================

async def _require_image_upload(image: UploadFile, endpoint_name: str) -> str:
    """
    Reads the uploaded file, pushes it to Cloudinary, and returns the public URL.
    Raises HTTPException on failure. Common to both stateless endpoints.
    """
    if not image:
        raise HTTPException(status_code=422, detail="An image file is required for this endpoint.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image file is empty.")

    logger.info("%s | Uploading image | size=%d bytes", endpoint_name, len(image_bytes))
    image_url = await async_upload_to_cloudinary(image_bytes)

    if not image_url:
        raise HTTPException(status_code=500, detail="Image upload to Cloudinary failed.")

    logger.info("%s | Image ready | url=%s", endpoint_name, image_url)
    return image_url


@app.get("/health")
async def health_check():
    """Health probe for load balancers."""
    return {"status": "ok", "graph_loaded": _app_graph is not None}


async def run_full_system_stream(inputs: dict, config: dict):
    final_answer_accumulator = ""

    try:
        async for event in _app_graph.astream_events(inputs, config=config, version="v1"):
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
                    # chunk.content can be a list (multimodal) or str — normalize to str
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
            state_snapshot = await _app_graph.aget_state(config)

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


# --- Main Chat Endpoint ---
@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    query: str = Form(...),
    user_id: str = Form(...),
    summary: str = Form("No Summary Found."),
    thread_id: str = Form("1"),
    image: UploadFile = File(None),
):
    # --- Guard: graph must be ready ---
    if _app_graph is None:
        raise HTTPException(status_code=503, detail="Service unavailable — graph not compiled.")

    # --- Input validation ---
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

    # --- Dynamic Session Threading ---
    dynamic_thread_id = await get_or_create_thread(user_id)
    
    logger.info("Chat request | user_id=%s | thread=%s (dynamic=%s) | query_len=%d",
                user_id, thread_id, dynamic_thread_id, len(query))

    # --- Image Upload: convert bytes → Cloudinary URL (CRIT-3 fix) ---
    image_url = None
    if image:
        image_bytes = await image.read()
        logger.info("Image uploaded | user_id=%s | size=%d bytes", user_id, len(image_bytes))
        image_url = await async_upload_to_cloudinary(image_bytes)
        if not image_url:
            raise HTTPException(status_code=500, detail="Image upload to Cloudinary failed.")

    # --- Build HumanMessage: multimodal when image present, plain text otherwise ---
    if image_url:
        # LangChain/Gemini multimodal format: content must be a list of typed dicts
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
        run_full_system_stream(inputs, config),
        media_type="text/event-stream",
    )


# ============================================================================
# STATELESS ANALYSIS ENDPOINTS — No LangGraph, no session, pure vision + schema
# ============================================================================

@app.post("/analyze/prescription", response_model=PrescriptionResponse)
@limiter.limit("20/minute")
async def analyze_prescription(
    request: Request,
    image: UploadFile = File(..., description="Photo of a medical prescription (JPEG/PNG)."),
):
    """
    Stateless OCR endpoint: extract structured medication data from a prescription image.

    - Accepts a single image upload (multipart/form-data).
    - Returns a validated `PrescriptionResponse` JSON object.
    - Does NOT use LangGraph or create a session/thread.
    - Rate limit: 20 requests/minute per IP.
    """
    image_url = await _require_image_upload(image, "/analyze/prescription")

    mm = get_model_manager()
    try:
        result: PrescriptionResponse = await mm.aanalyze_image_structured(
            image_url=image_url,
            system_prompt=_PRESCRIPTION_SYSTEM_PROMPT,
            output_schema=PrescriptionResponse,
        )
    except ValueError as e:
        # Schema validation failed — image is likely not a prescription
        logger.warning("/analyze/prescription | Validation error: %s", e)
        raise HTTPException(
            status_code=400,
            detail="The image does not appear to be a valid prescription, or the content could not be parsed. Please upload a clear prescription photo.",
        )
    except Exception as e:
        logger.exception("/analyze/prescription | Unexpected LLM error.")
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {e}")

    logger.info(
        "/analyze/prescription | Done | medications_found=%d | confidence=%s",
        len(result.medications), result.confidence,
    )
    return result


@app.post("/analyze/medicine-box", response_model=MedicineBoxResponse)
@limiter.limit("20/minute")
async def analyze_medicine_box(
    request: Request,
    image: UploadFile = File(..., description="Photo of a medicine box or packaging (JPEG/PNG)."),
):
    """
    Stateless OCR endpoint: extract structured pharmaceutical data from a medicine box image.

    - Accepts a single image upload (multipart/form-data).
    - Returns a validated `MedicineBoxResponse` JSON object.
    - Does NOT use LangGraph or create a session/thread.
    - Rate limit: 20 requests/minute per IP.
    """
    image_url = await _require_image_upload(image, "/analyze/medicine-box")

    mm = get_model_manager()
    try:
        result: MedicineBoxResponse = await mm.aanalyze_image_structured(
            image_url=image_url,
            system_prompt=_MEDICINE_BOX_SYSTEM_PROMPT,
            output_schema=MedicineBoxResponse,
        )
    except ValueError as e:
        # Schema validation failed — image is likely not a medicine box
        logger.warning("/analyze/medicine-box | Validation error: %s", e)
        raise HTTPException(
            status_code=400,
            detail="The image does not appear to be a valid medicine box, or the content could not be parsed. Please upload a clear photo of the packaging.",
        )
    except Exception as e:
        logger.exception("/analyze/medicine-box | Unexpected LLM error.")
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {e}")

    logger.info(
        "/analyze/medicine-box | Done | trade_name=%s | ingredients=%d",
        result.trade_name, len(result.active_ingredients),
    )
    return result