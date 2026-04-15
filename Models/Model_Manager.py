import threading

from Models.core_manager import BaseModelManager
from Models.text_generation import TextGenerationMixin
from Models.vision_generation import VisionGenerationMixin
from Models.task_wrappers import TaskWrappersMixin


class ModelManager(BaseModelManager, TextGenerationMixin, VisionGenerationMixin, TaskWrappersMixin):
    """
    The unified ModelManager class. 
    It inherits core logic and specific capabilities via Mixins.
    """
    pass


# Project-Wide Singleton
_global_mm_instance = None
_global_mm_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Returns the single shared ModelManager instance for the entire project."""
    global _global_mm_instance
    if _global_mm_instance is None:
        with _global_mm_lock:
            if _global_mm_instance is None:
                from Helper.key_manager import ApiKeyManager
                _global_mm_instance = ModelManager(
                    hf_key_manager=ApiKeyManager("hf", "HUGGINGFACE_API_KEY"),
                    google_key_manager=ApiKeyManager("google", "GOOGLE_API_KEY"),
                    groq_key_manager=ApiKeyManager("groq", "GROQ_API_KEY"),
                )
    return _global_mm_instance