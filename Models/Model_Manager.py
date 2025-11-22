import os
import torch
import io
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings 
from FlagEmbedding import FlagReranker
import google.generativeai as genai
from PIL import Image 
from groq import Groq
# --- Import Helpers ---
try:
    from Helper.HF_ApiManager import hf_ApiKeyManager
    from Helper.Groq_ApiManger import groq_ApiKeyManager
    from Helper.Google_ApiManger import google_ApiKeyManager
    from Helper.Image_Uploader import upload_to_cloudinary
except ImportError:
    print("FATAL ERROR: Helper modules not found.")
    exit() # Exit if helpers are missing

load_dotenv()

class ModelManager:    
    def __init__(self, HF_key_manager, Google_key_manger, groq_keyManger):    
        self.HF_key_manager = HF_key_manager
        self.Google_key_manger = Google_key_manger
        self.groq_keyManger = groq_keyManger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Model Configurations ---
        self.groq_model_name = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b") # Updated default
        self.opt_model_name = os.getenv("OPTIMIZATION_MODEL_NAME", "openai/gpt-oss-20b")
        self.gen_model_name = os.getenv("GENERATION_MODEL_NAME", "Intelligent-Internet/II-Medical-8B")
        self.val_model_name = os.getenv("VALIDATION_MODEL_NAME", "openai/gpt-oss-20b")
        self.ocr_model_name = os.getenv("OCR_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

        # --- Initialize Clients ---
        self._init_groq()
        self._init_gemini()
        
        # --- Load Local Models (Heavy) ---
        self.reranker_Model = self._load_reranker_model()
        self.embedding_model = self._load_local_embedding_model() 
        
        if not self.embedding_model:
            raise Exception("CRITICAL: Failed to load Embedding Model.")

    def _init_groq(self):
        try:
            key = self.groq_keyManger.get_next_api_key()
            self.groq_client = Groq(api_key=key) if key else None
        except Exception as e:
            print(f" Groq Init Failed: {e}")
            self.groq_client = None

    def _init_gemini(self):
        if self.Google_key_manger:
            try:
                key = self.Google_key_manger.get_next_api_key()
                genai.configure(api_key=key)
                self.fallback_model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                print(f"Gemini Init Failed: {e}")
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
            print(f" Embedding Load Error: {e}")
            return None

    def _load_reranker_model(self):
        try:
            return FlagReranker(
                os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"), 
                use_fp16=True
            )
        except Exception as e:
            print(f"❌ Reranker Load Error: {e}")
            return None

    # --- Generation Methods ---
    def _generate_hf(self, prompt: str, model_name: str) -> str:
        client = InferenceClient(api_key=self.HF_key_manager.get_next_api_key())
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )
        return completion.choices[0].message.content

    def _generate_groq(self, prompt: str) -> str:
        if not self.groq_client: raise Exception("Groq not available")
        completion = self.groq_client.chat.completions.create(
            model=self.groq_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096
        )
        return completion.choices[0].message.content

    def _generate_gemini(self, prompt: str) -> str:
        if not self.fallback_model: raise Exception("Gemini not available")
        response = self.fallback_model.generate_content(prompt)
        return response.text

    # --- The Master Generate Function (Robust Fallback Chain) ---
    def generate(self, prompt: str, hf_model_name: str) -> str:
        try:
            return self._generate_hf(prompt, hf_model_name)
        except Exception as e_hf:
            print(f" HF Failed ({e_hf}). Switching to Groq...")
            
            # 2. Try Groq (Secondary)
            try:
                return self._generate_groq(prompt)
            except Exception as e_groq:
                print(f" Groq Failed ({e_groq}). Switching to Gemini...")
                
                # 3. Try Gemini (Final Resort)
                try:
                    return self._generate_gemini(prompt)
                except Exception as e_gemini:
                    return f"CRITICAL SYSTEM FAILURE: All models failed. Last Error: {e_gemini}"

    # --- Image Generation ---
    def generate_with_image(self, text: str, image_bytes: bytes = None, image_url: str = None) -> str:
        """
        Handles Vision tasks. Can accept either raw bytes (will upload) or ready URL.
        """
        # Logic: Prefer URL if available (faster), else upload bytes
        final_url = image_url
        if not final_url and image_bytes:
            final_url = upload_to_cloudinary(image_bytes)
        
        if not final_url:
            return "Error: No valid image provided."

        try:
            client = InferenceClient(api_key=self.HF_key_manager.get_next_api_key())
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
            print(f" HF Vision Failed: {e_hf}. Switching to Gemini...")
            
            # 2. Fallback to Gemini Vision
            try:
                if not self.fallback_model: raise Exception("Gemini not config")
                if image_bytes:
                    img = Image.open(io.BytesIO(image_bytes))
                    response = self.fallback_model.generate_content([text, img])
                    return response.text
                else:
                    return "Error: Gemini fallback requires image bytes."
            except Exception as e_gemini:
                return f"Vision Failure: {e_gemini}"

    # --- Specific Wrappers ---
    def optimize_query(self, prompt: str) -> str:
        return self.generate(prompt, self.opt_model_name)
    
    def generate_answer(self, prompt: str) -> str:
        return self.generate(prompt, self.gen_model_name)

    def validate_answer(self, prompt: str) -> str:
        return self.generate(prompt, self.val_model_name)
    
    def summarize(self, prompt: str) -> str:
        return self.generate(prompt, self.opt_model_name)