from fastapi import UploadFile, HTTPException
from Helper.Image_Uploader import async_upload_to_cloudinary
from Server.config import ALLOWED_IMAGE_TYPES, MAX_IMAGE_SIZE, _IMAGE_MAGIC_BYTES, logger

async def _require_image_upload(image: UploadFile, endpoint_name: str) -> str:
    """
    Reads the uploaded file, validates type and size, pushes it to Cloudinary,
    and returns the public URL. Raises HTTPException on failure.
    """
    if not image:
        raise HTTPException(status_code=422, detail="An image file is required for this endpoint.")

    # 1) MIME content-type whitelist
    declared_type = (image.content_type or "").lower()
    if declared_type not in ALLOWED_IMAGE_TYPES:
        logger.warning("%s | Rejected upload | declared_type=%s filename=%s", endpoint_name, declared_type, image.filename)
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported image type '{declared_type}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image file is empty.")

    # 2) File size cap — prevents OOM and bandwidth abuse
    if image.size > MAX_IMAGE_SIZE:
        size_mb = image.size / (1024 * 1024)
        logger.warning("%s | Rejected oversized upload | size=%.1fMB filename=%s", endpoint_name, size_mb, image.filename)
        raise HTTPException(
            status_code=422,
            detail=f"Image too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_IMAGE_SIZE // (1024*1024)} MB.",
        )

    # 3) Magic-byte signature verification — prevents disguised payloads
    header = image_bytes[:8]
    magic_match = any(header.startswith(sig) for sig in _IMAGE_MAGIC_BYTES)
    if not magic_match:
        logger.warning("%s | Rejected invalid magic bytes | declared_type=%s filename=%s", endpoint_name, declared_type, image.filename)
        raise HTTPException(
            status_code=422,
            detail="File content does not match a valid image format. Upload a real JPEG, PNG, or WebP image.",
        )

    logger.info("%s | Uploading image | size=%d bytes", endpoint_name, image.size)
    image_url = await async_upload_to_cloudinary(image_bytes)

    if not image_url:
        raise HTTPException(status_code=500, detail="Image upload to Cloudinary failed.")

    logger.info("%s | Image ready | url=%s", endpoint_name, image_url)
    return image_url
