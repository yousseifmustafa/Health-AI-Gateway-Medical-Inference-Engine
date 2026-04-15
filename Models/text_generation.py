import asyncio
import logging
from huggingface_hub import InferenceClient

logger = logging.getLogger("sehatech.models")

class TextGenerationMixin:
    """Handles text-based generative tasks and fallback logic."""

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
