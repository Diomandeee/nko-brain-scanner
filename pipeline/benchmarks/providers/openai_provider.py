"""
OpenAI GPT Provider for N'Ko Benchmark.

Implements GPT-5.2 and GPT-5-mini for benchmarking.
"""

import os
import time
from typing import Optional

from .base import (
    BaseProvider,
    TranslationResult,
    ExplanationResult,
    CompletionResult,
    create_translation_prompt,
    create_explanation_prompt,
)
from ..config import get_api_key, MODELS

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIProvider(BaseProvider):
    """
    OpenAI GPT provider for N'Ko benchmarks.
    
    Supports:
    - GPT-5.2 (gpt-5.2)
    - GPT-5 Mini (gpt-5-mini)
    """
    
    def __init__(self, model_id: str = "gpt-5.2", api_key: Optional[str] = None):
        super().__init__(model_id, api_key)
        
        # Get actual model ID from config
        model_config = MODELS.get(model_id)
        self.actual_model_id = model_config.model_id if model_config else model_id
        self.cost_per_1k_input = model_config.cost_per_1k_input if model_config else 10.0
        self.cost_per_1k_output = model_config.cost_per_1k_output if model_config else 30.0
        
        # Initialize client
        self._api_key = api_key or get_api_key("openai")
        self._client = None
        
        if HAS_OPENAI and self._api_key:
            self._client = OpenAI(api_key=self._api_key)
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def is_available(self) -> bool:
        return HAS_OPENAI and self._client is not None
    
    def _call_api(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system: Optional[str] = None,
    ) -> tuple[str, int, int, float]:
        """
        Call the OpenAI API.
        
        Returns:
            Tuple of (response_text, input_tokens, output_tokens, latency_ms)
        """
        if not self.is_available:
            raise RuntimeError("OpenAI client not available")
        
        start_time = time.time()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # NOTE: Some newer OpenAI models (e.g. GPT-5.x) require `max_completion_tokens`
        # instead of `max_tokens`. We switch based on the model family.
        create_kwargs = {
            "model": self.actual_model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if str(self.actual_model_id).startswith("gpt-5") or str(self.actual_model_id).startswith("o"):
            create_kwargs["max_completion_tokens"] = max_tokens
        else:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response text
        response_text = ""
        if response.choices:
            response_text = response.choices[0].message.content or ""
        
        # Token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        return response_text, input_tokens, output_tokens, latency_ms
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """Translate text using GPT."""
        if not self.is_available:
            return TranslationResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="OpenAI client not available",
                source_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        
        prompt = create_translation_prompt(text, source_lang, target_lang)
        system = "You are an expert N'Ko language translator. Provide accurate, natural translations."
        
        try:
            response_text, input_tokens, output_tokens, latency_ms = self._call_api(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.3,
                system=system,
            )
            
            return TranslationResult(
                success=True,
                response=response_text,
                latency_ms=latency_ms,
                model_id=self.model_id,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                source_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                translation=response_text.strip(),
            )
            
        except Exception as e:
            return TranslationResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error=str(e),
                source_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
    
    async def explain_nko(
        self,
        nko_text: str,
        include_examples: bool = True,
    ) -> ExplanationResult:
        """Explain N'Ko text using GPT."""
        if not self.is_available:
            return ExplanationResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="OpenAI client not available",
                nko_text=nko_text,
            )
        
        prompt = create_explanation_prompt(nko_text, include_examples)
        system = "You are an expert in N'Ko language and Manding linguistics. Provide accurate, educational explanations."
        
        try:
            response_text, input_tokens, output_tokens, latency_ms = self._call_api(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.3,
                system=system,
            )
            
            return ExplanationResult(
                success=True,
                response=response_text,
                latency_ms=latency_ms,
                model_id=self.model_id,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                nko_text=nko_text,
                explanation=response_text.strip(),
            )
            
        except Exception as e:
            return ExplanationResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error=str(e),
                nko_text=nko_text,
            )
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> CompletionResult:
        """General completion using GPT."""
        if not self.is_available:
            return CompletionResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="OpenAI client not available",
                prompt=prompt,
            )
        
        try:
            response_text, input_tokens, output_tokens, latency_ms = self._call_api(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            return CompletionResult(
                success=True,
                response=response_text,
                latency_ms=latency_ms,
                model_id=self.model_id,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                prompt=prompt,
                completion=response_text.strip(),
            )
            
        except Exception as e:
            return CompletionResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error=str(e),
                prompt=prompt,
            )

