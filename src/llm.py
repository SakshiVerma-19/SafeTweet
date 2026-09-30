import os
import time
import json
import re
import httpx
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    Unified LLM Client supporting:
    1. OpenRouter API (OPENROUTER_API_KEY) - universal access to Llama 3, Gemini, Claude, Mistral
    2. Google Gemini API via official google-genai SDK (GEMINI_API_KEY or GOOGLE_API_KEY)
    3. Groq Cloud API (GROQ_API_KEY)
    4. Local Qwen2.5-1.5B fallback
    """
    _instance = None

    def __init__(self, model_name: str = None):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        if self.openrouter_key and self.openrouter_key.strip():
            self.provider = "openrouter"
            self.client = httpx.Client(timeout=60.0)
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
            self.pipe = None
            print(f"[OK] [LLMClient] Connected to OpenRouter API ({self.model_name})")

        elif self.gemini_key and self.gemini_key.strip():
            from google import genai
            self.provider = "gemini"
            self.client = genai.Client(api_key=self.gemini_key.strip())
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
            self.pipe = None
            self._last_call = 0.0
            print(f"[OK] [LLMClient] Connected to Google Gemini API ({self.model_name})")

        elif self.groq_key and self.groq_key.strip():
            from groq import Groq
            self.provider = "groq"
            self.client = Groq(api_key=self.groq_key.strip())
            self.model_name = model_name or os.getenv("GROQ_MODEL", "groq/compound-mini")
            self.pipe = None
            print(f"[OK] [LLMClient] Connected to Groq Cloud API ({self.model_name})")

        else:
            self.provider = "local"
            self.client = None
            self.pipe = None
            print("[--] [LLMClient] No API key detected; defaulting to local model.")

    @classmethod
    def get_shared_client(cls, model_name: str = None):
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _call_openrouter(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.0, json_mode: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self.openrouter_key.strip()}",
            "HTTP-Referer": "https://github.com/support-agent",
            "X-Title": "Autonomous Support Agent",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(6):
            try:
                response = self.client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return content.strip() if content else ("{}" if json_mode else "")
                
                # Handle rate limit or temporary server issues
                if response.status_code in (429, 502, 503):
                    wait_time = min(3 * (2 ** attempt), 30)
                    print(f"OpenRouter status {response.status_code}. Pausing {wait_time}s (attempt {attempt+1}/6)...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")
            except httpx.RequestError as e:
                wait_time = min(2 * (attempt + 1), 10)
                print(f"Network error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError("Exceeded max retries with OpenRouter API.")

    def _call_gemini(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.0, json_mode: bool = False) -> str:
        from google.genai import types

        system_parts = []
        contents = []

        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if system_parts:
            config_kwargs["system_instruction"] = "\n".join(system_parts)

        config = types.GenerateContentConfig(**config_kwargs)

        for attempt in range(8):
            elapsed = time.time() - getattr(self, "_last_call", 0.0)
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)

            try:
                self._last_call = time.time()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                return response.text.strip() if response.text else ("{}" if json_mode else "")
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    wait_time = min(5 * (2 ** attempt), 60)
                    print(f"Gemini rate limit hit. Pausing {wait_time}s (attempt {attempt+1}/8)...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError("Exceeded max retries with Gemini API.")

    def _call_groq(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.0, json_mode: bool = False) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(5):
            try:
                completion = self.client.chat.completions.create(**kwargs)
                return completion.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "limit" in err_str:
                    wait_time = (attempt + 1) * 5
                    print(f"Rate limit encountered. Pausing for {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError("Exceeded maximum retries on Groq API.")

    def _call_local(self, messages: list[dict], max_tokens: int = 150) -> str:
        if self.pipe is None:
            from transformers import pipeline
            import torch
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.pipe = pipeline(
                "text-generation",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype=torch_dtype,
                device_map=device_map
            )
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()

    def chat_completion(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.0, json_mode: bool = False) -> str:
        if self.provider == "openrouter":
            return self._call_openrouter(messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)
        elif self.provider == "gemini":
            return self._call_gemini(messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)
        elif self.provider == "groq":
            return self._call_groq(messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)
        else:
            return self._call_local(messages, max_tokens=max_tokens)
