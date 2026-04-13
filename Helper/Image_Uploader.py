import os
import asyncio
import logging
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()

logger = logging.getLogger("sehatech.cloudinary")

_cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
_api_key = os.getenv("CLOUDINARY_API_KEY")
_api_secret = os.getenv("CLOUDINARY_API_SECRET")
_is_configured = False

if _cloud_name and _api_key and _api_secret:
    cloudinary.config(
        cloud_name=_cloud_name,
        api_key=_api_key,
        api_secret=_api_secret,
        secure=True,
    )
    _is_configured = True
    logger.info("Cloudinary configured at module load.")
else:
    logger.warning("Cloudinary credentials missing — image uploads will fail.")


def upload_to_cloudinary(image_bytes: bytes) -> str | None:
    if not image_bytes:
        logger.error("No image bytes provided for upload.")
        return None
    
    if not _is_configured:
        logger.error("Cloudinary is not configured. Cannot upload image.")
        return None
    
    try:
        upload_result = cloudinary.uploader.upload(
            image_bytes,
            resource_type="image",
            overwrite=True,
            folder="sehatech/uploads",
            unique_filename=True,
        )
        logger.info("Image uploaded to Cloudinary | url=%s", upload_result.get("secure_url"))
        return upload_result.get("secure_url")

    except Exception as e:
        logger.error("Cloudinary upload failed: %s", str(e))
        return None


async def async_upload_to_cloudinary(image_bytes: bytes) -> str | None:
    """Async wrapper — offloads sync Cloudinary upload to thread pool."""
    return await asyncio.to_thread(upload_to_cloudinary, image_bytes)
