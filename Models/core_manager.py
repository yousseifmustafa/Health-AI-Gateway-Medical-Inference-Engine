import os
import torch
import threading
import logging
import google.generativeai as genai
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker

logger = logging.getLogger("sehatech.models")

class BaseModelManager:
    """Core initialization and lazy-loading foundation for the ModelManager."""
    
    def __init__(self, hf_key_manager, google_key_manager, groq_key_manager):
        self.hf_key_manager = hf_key_manager
        self.google_key_manager = google_key_manager
        self.groq_key_manager = groq_key_manager
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Model Configurations ---
        self.groq_model_name = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
        self.opt_model_name = os.getenv("OPTIMIZATION_MODEL_NAME", "openai/gpt-oss-20b")
        self.gen_model_name = os.getenv("GENERATION_MODEL_NAME", "Intelligent-Internet/II-Medical-8B")
        self.val_model_name = os.getenv("VALIDATION_MODEL_NAME", "openai/gpt-oss-20b")
        self.ocr_model_name = os.getenv("OCR_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

        # --- Initialize Clients (Lightweight) ---
        self._init_groq()
        self._init_gemini()

        # --- Lazy-Loaded Heavy Models ---
        self._reranker_model = None
        self._embedding_model = None
        self._lazy_lock = threading.Lock()

    @property
    def reranker_model(self):
        """Lazy-load the reranker model on first access."""
        if self._reranker_model is None:
            with self._lazy_lock:
                if self._reranker_model is None:
                    self._reranker_model = self._load_reranker_model()
        return self._reranker_model

    @property
    def embedding_model(self):
        """Lazy-load the embedding model on first access."""
        if self._embedding_model is None:
            with self._lazy_lock:
                if self._embedding_model is None:
                    self._embedding_model = self._load_local_embedding_model()
                    if not self._embedding_model:
                        raise Exception("CRITICAL: Failed to load Embedding Model.")
        return self._embedding_model

    def _init_groq(self):
        try:
            key = self.groq_key_manager.get_next_api_key()
            self.groq_client = Groq(api_key=key, timeout=10.0) if key else None
        except Exception as e:
            logger.warning("Groq Init Failed: %s", e)
            self.groq_client = None

    def _init_gemini(self):
        """Initialize Gemini fallback model for vision tasks only."""
        if self.google_key_manager:
            try:
                key = self.google_key_manager.get_next_api_key()
                genai.configure(api_key=key)
                self.fallback_model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                logger.warning("Gemini Init Failed: %s", e)
                self.fallback_model = None

    def _load_local_embedding_model(self):
        try:
            return HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m"),
                model_kwargs={'device': self.device, 'truncate_dim': 256},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error("Embedding Load Error: %s", e)
            return None

    def _load_reranker_model(self):
        try:
            return FlagReranker(
                os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
                use_fp16=True
            )
        except Exception as e:
            logger.error("Reranker Load Error: %s", e)
            return None
