import os
import threading
import logging
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

class ApiKeyManager:
    """
    Unified API Key Manager for round-robin rotation of API keys.
    Thread-safe and dynamically loads keys from .env based on a prefix.
    """
    def __init__(self, provider_name: str, env_prefix: str):
        """
        Args:
            provider_name: distinct name for logging (e.g., 'google', 'groq', 'hf')
            env_prefix: environment variable prefix (e.g., 'GOOGLE_API_KEY')
        """
        self.provider_name = provider_name
        self.env_prefix = env_prefix
        self.logger = logging.getLogger(f"sehatech.keys.{provider_name}")
        
        self.api_keys: List[str] = self._load_dynamic_api_keys(env_prefix)
        self.current_index: int = 0
        self._lock = threading.Lock()
        
        if not self.api_keys:
            self.logger.warning("No dynamic API keys found with prefix '%s'.", env_prefix)
        else:
            self.logger.info("%s ApiKeyManager initialized. Loaded %d keys.", provider_name.upper(), len(self.api_keys))

    def _load_dynamic_api_keys(self, prefix: str) -> List[str]:
        keys = []
        
        # 1. Try safe loading of the base key (without number)
        base_key = os.getenv(prefix)
        if base_key:
            keys.append(base_key)

        # 2. Load numbered keys (PREFIX1, PREFIX2, etc.)
        i = 1
        while True:
            key_name = f"{prefix}{i}"
            key_value = os.getenv(key_name)
            
            if key_value:
                keys.append(key_value)
                i += 1
            else:
                break 
        
        # Deduplicate keys (keeping order)
        seen = set() # O(1)
        unique_keys = [] # O(n) But Keeping order
        for k in keys:
            if k not in seen:
                unique_keys.append(k)
                seen.add(k)
                
        return unique_keys

    def get_next_api_key(self) -> Optional[str]:
        """Returns the next API key in the rotation, or None if no keys available."""
        
        if not self.api_keys:
            self.logger.error("Attempted to get key for %s but no keys are loaded.", self.provider_name)
            return None
            
        with self._lock:
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
        
        return key
