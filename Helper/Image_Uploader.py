import os
import asyncio
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader


load_dotenv()


def upload_to_cloudinary(image_bytes: bytes) -> str | None:
    cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True 
    )

   
   
    try:
        upload_result = cloudinary.uploader.upload(
            image_bytes,
            resource_type="image",
            overwrite=True,
            unique_filename=True 
        )
        
        return upload_result.get("secure_url")
    
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        return None


async def async_upload_to_cloudinary(image_bytes: bytes) -> str | None:
    """Async wrapper — offloads sync Cloudinary upload to thread pool."""
    return await asyncio.to_thread(upload_to_cloudinary, image_bytes)
