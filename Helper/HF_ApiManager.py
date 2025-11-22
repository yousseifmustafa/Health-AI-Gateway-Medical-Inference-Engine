import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

class hf_ApiKeyManager:
            
    def __init__(self, env_prefix: str = "HUGGINGFACE_API_KEY"):
        self.api_keys: List[str] = self._load_dynamic_api_keys(env_prefix)
        self.current_index: int = 0
        
        if not self.api_keys:
            print(f"--- WARNING: No dynamic API keys found with prefix '{env_prefix}'. ---")
        else:
            print(f"--- Hf ApiKeyManager initialized. Loaded {len(self.api_keys)} keys. ---")

    def _load_dynamic_api_keys(self, prefix: str) -> List[str]:
        keys_dict = {}
        i = 1
        while True:
            key_name = f"{prefix}{i}"
            key_value = os.getenv(key_name)
            
            if key_value:
                keys_dict[i] = key_value
                i += 1
            else:
                break 
        
        return [keys_dict[k] for k in sorted(keys_dict.keys())]

    def get_next_api_key(self) -> Optional[str]:
        if not self.api_keys:
            return None 
                
        current_key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        
        return current_key
    
    
    