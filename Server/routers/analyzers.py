import traceback
from fastapi import APIRouter, File, UploadFile, Request, HTTPException

from Server.config import logger, limiter
from Server.schemas import PrescriptionResponse, MedicineBoxResponse
from Server.prompts import _PRESCRIPTION_SYSTEM_PROMPT, _MEDICINE_BOX_SYSTEM_PROMPT
from Server.utils import _require_image_upload
from Models.Model_Manager import get_model_manager

router = APIRouter()

@router.post("/analyze/prescription", response_model=PrescriptionResponse)
@limiter.limit("20/minute")
async def analyze_prescription(
    request: Request,
    image: UploadFile = File(..., description="Photo of a medical prescription (JPEG/PNG)."),
):
    """
    Stateless OCR endpoint: extract structured medication data from a prescription image.
    """
    logger.info("[PIPELINE][STEP-1] /analyze/prescription | Entering endpoint")
    image_url = await _require_image_upload(image, "/analyze/prescription")
    logger.info("[PIPELINE][STEP-2] /analyze/prescription | image_url acquired: %s", image_url)

    logger.info("[PIPELINE][STEP-3] /analyze/prescription | Fetching ModelManager singleton")
    mm = get_model_manager()
    logger.info("[PIPELINE][STEP-4] /analyze/prescription | ModelManager ready: %s", mm)

    logger.info("[PIPELINE][STEP-5] /analyze/prescription | Calling aanalyze_image_structured...")
    try:
        result: PrescriptionResponse = await mm.aanalyze_image_structured(
            image_url=image_url,
            system_prompt=_PRESCRIPTION_SYSTEM_PROMPT,
            output_schema=PrescriptionResponse,
        )
        logger.info("[PIPELINE][STEP-6] /analyze/prescription | LLM raw result type: %s", type(result))
        logger.info("[PIPELINE][STEP-6] /analyze/prescription | LLM raw result value: %s", result)

    except ValueError as e:
        raw_error = str(e)
        tb = traceback.format_exc()
        logger.error("[PIPELINE][STEP-6-FAIL-ValueError] /analyze/prescription | RAW ERROR: %s", raw_error)
        logger.error("[PIPELINE][STEP-6-FAIL-ValueError] /analyze/prescription | TRACEBACK:\n%s", tb)
        logger.error("/analyze/prescription | *** DIAGNOSTIC RAW ERROR ***: %s", raw_error)
        logger.error("/analyze/prescription | Full traceback: %s", tb)
        raise HTTPException(
            status_code=400,
            detail=f"[DIAG] Raw LLM Error: {raw_error[:500]}",
        )
    except Exception as e:
        raw_error = str(e)
        tb = traceback.format_exc()
        logger.error("[PIPELINE][STEP-6-FAIL-Exception] /analyze/prescription | RAW ERROR: %s", raw_error)
        logger.error("[PIPELINE][STEP-6-FAIL-Exception] /analyze/prescription | TRACEBACK:\n%s", tb)
        logger.exception("/analyze/prescription | Unexpected LLM error.")
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {raw_error}")

    logger.info("[PIPELINE][STEP-7] /analyze/prescription | medications_found=%d | confidence=%s", len(result.medications), result.confidence)
    logger.info(
        "/analyze/prescription | Done | medications_found=%d | confidence=%s",
        len(result.medications), result.confidence,
    )
    logger.info("[PIPELINE][STEP-8] /analyze/prescription | Returning response successfully")
    return result


@router.post("/analyze/medicine-box", response_model=MedicineBoxResponse)
@limiter.limit("20/minute")
async def analyze_medicine_box(
    request: Request,
    image: UploadFile = File(..., description="Photo of a medicine box or packaging (JPEG/PNG)."),
):
    """
    Stateless OCR endpoint: extract structured pharmaceutical data from a medicine box image.
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
        raw_error = str(e)
        logger.error("/analyze/medicine-box | *** DIAGNOSTIC RAW ERROR ***: %s", raw_error)
        logger.error("/analyze/medicine-box | Full traceback: %s", traceback.format_exc())
        raise HTTPException(
            status_code=400,
            detail=f"[DIAG] Raw LLM Error: {raw_error[:500]}",
        )
    except Exception as e:
        logger.exception("/analyze/medicine-box | Unexpected LLM error.")
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {e}")

    logger.info(
        "/analyze/medicine-box | Done | trade_name=%s | ingredients=%d",
        result.trade_name, len(result.active_ingredients),
    )
    return result
