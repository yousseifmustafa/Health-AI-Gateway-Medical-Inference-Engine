import os
import torch
import io
import asyncio
import threading
import logging
from huggingface_hub import InferenceClient
from typing import Optional, Type
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from FlagEmbedding import FlagReranker
import google.generativeai as genai
from PIL import Image
from groq import Groq


logger = logging.getLogger("sehatech.models")

# --- Import Helpers ---
try:
    from Helper.key_manager import ApiKeyManager
    from Helper.Image_Uploader import upload_to_cloudinary, async_upload_to_cloudinary
except ImportError:
    logger.critical("Helper modules not found.")
    exit()


class ModelManager:
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
        """Initialize Gemini fallback model for vision tasks only.
        NOTE: We do NOT call genai.configure() here because the LangChain
        ChatGoogleGenerativeAI wrapper in agent_node passes google_api_key
        per-call, which would conflict with a global config.
        """
        if self.google_key_manager:
            try:
                key = self.google_key_manager.get_next_api_key()
                genai.configure(api_key=key)
                self.fallback_model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                logger.warning("Gemini Init Failed: %s", e)
                self.fallback_model = None

    # --- Loaders ---
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

    # --- Synchronous Generation Methods (kept for compatibility) ---
    def _generate_hf(self, prompt: str, model_name: str) -> str:
        client = InferenceClient(api_key=self.hf_key_manager.get_next_api_key(), timeout=10)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )
        return completion.choices[0].message.content

    def _generate_groq(self, prompt: str) -> str:
        if not self.groq_client:
            raise Exception("Groq not available")
        completion = self.groq_client.chat.completions.create(
            model=self.groq_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096
        )
        return completion.choices[0].message.content

    def _generate_gemini(self, prompt: str) -> str:
        if not self.fallback_model:
            raise Exception("Gemini not available")
        response = self.fallback_model.generate_content(prompt)
        return response.text

    # --- The Master Generate Function (Robust Fallback Chain) ---
    def generate(self, prompt: str, hf_model_name: str) -> str:
        try:
            return self._generate_hf(prompt, hf_model_name)
        except Exception as e_hf:
            logger.warning("HF Failed (%s). Switching to Groq...", e_hf)
            try:
                return self._generate_groq(prompt)
            except Exception as e_groq:
                logger.warning("Groq Failed (%s). Switching to Gemini...", e_groq)
                try:
                    return self._generate_gemini(prompt)
                except Exception as e_gemini:
                    return f"CRITICAL SYSTEM FAILURE: All models failed. Last Error: {e_gemini}"

    # --- Async Master Generate (offloads sync SDK to thread pool) ---
    async def agenerate(self, prompt: str, hf_model_name: str) -> str:
        return await asyncio.to_thread(self.generate, prompt, hf_model_name)

    # --- Image Generation ---
    def generate_with_image(self, text: str, image_bytes: bytes = None, image_url: str = None) -> str:
        """
        Handles Vision tasks. Can accept either raw bytes (will upload) or ready URL.
        """
        final_url = image_url
        if not final_url and image_bytes:
            final_url = upload_to_cloudinary(image_bytes)

        if not final_url:
            return "Error: No valid image provided."

        try:
            client = InferenceClient(api_key=self.hf_key_manager.get_next_api_key())
            completion = client.chat.completions.create(
                model=self.ocr_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image_url", "image_url": {"url": final_url}}
                        ]
                    }
                ],
                max_tokens=1024
            )
            return completion.choices[0].message.content

        except Exception as e_hf:
            logger.warning("HF Vision Failed: %s. Switching to Gemini...", e_hf)
            try:
                if not self.fallback_model:
                    raise Exception("Gemini not config")
                if image_bytes:
                    img = Image.open(io.BytesIO(image_bytes))
                    response = self.fallback_model.generate_content([text, img])
                    return response.text
                else:
                    return "Error: Gemini fallback requires image bytes."
            except Exception as e_gemini:
                return f"Vision Failure: {e_gemini}"

    # --- Async Image Generation (Pipelined) ---
    async def agenerate_with_image(self, text: str, image_bytes: bytes = None, image_url: str = None) -> str:
        """
        Fully async vision pipeline.
        Uses asyncio.gather to pipeline Cloudinary upload + API key retrieval
        in parallel, then calls the LLM.
        """
        final_url = image_url

        if not final_url and image_bytes:
            # Pipeline: upload image AND acquire API key in parallel
            async def _get_key():
                return await asyncio.to_thread(self.hf_key_manager.get_next_api_key)

            upload_result, api_key = await asyncio.gather(
                async_upload_to_cloudinary(image_bytes),
                _get_key(),
            )
            final_url = upload_result
        else:
            api_key = await asyncio.to_thread(self.hf_key_manager.get_next_api_key)

        if not final_url:
            return "Error: No valid image provided."

        # HF Inference call (sync SDK → thread pool)
        try:
            def _hf_call():
                client = InferenceClient(api_key=api_key)
                completion = client.chat.completions.create(
                    model=self.ocr_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image_url", "image_url": {"url": final_url}}
                            ]
                        }
                    ],
                    max_tokens=1024
                )
                return completion.choices[0].message.content

            return await asyncio.to_thread(_hf_call)

        except Exception as e_hf:
            logger.warning("HF Vision Failed: %s. Switching to Gemini...", e_hf)
            try:
                if not self.fallback_model:
                    raise Exception("Gemini not configured")

                def _gemini_call():
                    if image_bytes:
                        # Bytes path: open as PIL Image
                        img = Image.open(io.BytesIO(image_bytes))
                        response = self.fallback_model.generate_content([text, img])
                    else:
                        # URL path: pass via file_data part (Gemini SDK >= 0.7)
                        response = self.fallback_model.generate_content([
                            {"role": "user", "parts": [
                                {"text": text},
                                {"file_data": {"mime_type": "image/jpeg", "file_uri": final_url}}
                            ]}
                        ])
                    return response.text

                return await asyncio.to_thread(_gemini_call)

            except Exception as e_gemini:
                return f"Vision Failure: {e_gemini}"

    # -------------------------------------------------------------------------
    # Structured Vision Analysis — stateless, one-shot, schema-validated output
    # -------------------------------------------------------------------------
    async def aanalyze_image_structured(
        self,
        image_url: str,
        system_prompt: str,
        output_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Calls Gemini (via LangChain ChatGoogleGenerativeAI) with a multimodal
        message and returns a validated Pydantic instance matching `output_schema`.

        Uses `with_structured_output` — Gemini's native JSON-mode — so the response
        is always schema-conformant or raises a clear validation error.

        This method is STATELESS: no LangGraph graph, no thread_id, no checkpointer.
        It is the backend engine for the /analyze/* FastAPI endpoints.

        Args:
            image_url:     A publicly accessible URL to the image (e.g. Cloudinary).
            system_prompt: Specialized instruction string for this analysis type.
            output_schema: A Pydantic BaseModel class defining the expected JSON shape.

        Returns:
            A validated instance of `output_schema`.

        Raises:
            Exception: Propagates LLM or validation errors to the caller for
                       translation into appropriate HTTP responses.
        """
        api_key = self.google_key_manager.get_next_api_key()

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1,   # Low temperature → deterministic, structured extraction
        )

        # Bind the output schema — instructs Gemini to return valid JSON only
        structured_llm = llm.with_structured_output(output_schema)

        # Build a multimodal message: image + instruction
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text",      "text": system_prompt},
        ])

        logger.info(
            "aanalyze_image_structured | schema=%s | image_url=%s",
            output_schema.__name__, image_url,
        )

        # ainvoke is natively async on ChatGoogleGenerativeAI
        result = await structured_llm.ainvoke([message])
        return result

    # --- Specific Wrappers (Sync) ---
    def optimize_query(self, prompt: str) -> str:
        return self.generate(prompt, self.opt_model_name)

    def generate_answer(self, prompt: str) -> str:
        return self.generate(prompt, self.gen_model_name)

    def validate_answer(self, prompt: str) -> str:
        return self.generate(prompt, self.val_model_name)

    def summarize(self, prompt: str) -> str:
        return self.generate(prompt, self.opt_model_name)

    # --- Specific Wrappers (Async) ---
    async def aoptimize_query(self, prompt: str) -> str:
        return await asyncio.to_thread(self.optimize_query, prompt)

    async def agenerate_answer(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate_answer, prompt)

    async def avalidate_answer(self, prompt: str) -> str:
        return await asyncio.to_thread(self.validate_answer, prompt)

    async def asummarize(self, prompt: str) -> str:
        return await asyncio.to_thread(self.summarize, prompt)


# --- Project-Wide Singleton ---
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