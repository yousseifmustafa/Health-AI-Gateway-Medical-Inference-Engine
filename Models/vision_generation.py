import io
import asyncio
import logging
from typing import Type
from pydantic import BaseModel
from PIL import Image
from huggingface_hub import InferenceClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

try:
    from Helper.Image_Uploader import upload_to_cloudinary, async_upload_to_cloudinary
except ImportError:
    pass

logger = logging.getLogger("sehatech.models")

class VisionGenerationMixin:
    """Handles vision-based generative tasks and structured OCR."""

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
                if not getattr(self, 'fallback_model', None):
                    raise Exception("Gemini not config")
                if image_bytes:
                    img = Image.open(io.BytesIO(image_bytes))
                    response = self.fallback_model.generate_content([text, img])
                    return response.text
                else:
                    return "Error: Gemini fallback requires image bytes."
            except Exception as e_gemini:
                return f"Vision Failure: {e_gemini}"

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
                if not getattr(self, 'fallback_model', None):
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

    async def aanalyze_image_structured(
        self,
        image_url: str,
        system_prompt: str,
        output_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Calls Gemini (via LangChain ChatGoogleGenerativeAI) with a multimodal
        message and returns a validated Pydantic instance matching `output_schema`.
        """
        api_key = self.google_key_manager.get_next_api_key()

        from langchain_google_genai import HarmCategory, HarmBlockThreshold

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1,   
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        structured_llm = llm.with_structured_output(output_schema)

        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text",      "text": system_prompt},
        ])

        logger.info(
            "aanalyze_image_structured | schema=%s | image_url=%s",
            output_schema.__name__, image_url,
        )
        print(f"[MODEL][STEP-A] aanalyze_image_structured | schema={output_schema.__name__} | image_url={image_url}")
        print(f"[MODEL][STEP-B] aanalyze_image_structured | LangChain model ready, calling ainvoke...")

        try:
            result = await structured_llm.ainvoke([message])
        except Exception as llm_exc:
            import traceback
            print(f"[MODEL][STEP-C-FAIL] aanalyze_image_structured | ainvoke raised {type(llm_exc).__name__}: {llm_exc}")
            print(f"[MODEL][STEP-C-FAIL] Traceback:\n{traceback.format_exc()}")
            raise 

        print(f"[MODEL][STEP-C] aanalyze_image_structured | ainvoke returned type={type(result)}")
        print(f"[MODEL][STEP-D] aanalyze_image_structured | raw result={result}")
        return result
