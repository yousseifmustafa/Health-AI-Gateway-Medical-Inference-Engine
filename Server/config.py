import re
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

# --- Structured Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sehatech.server")

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)

# --- Input Validation Constants ---
MAX_QUERY_LENGTH = 5000
_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,255}$")

# --- Image Upload Security ---
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB hard cap
_IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff":       "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF":              "image/webp",  # WebP starts with RIFF...WEBP
}
