"""
Google Gemini Provider for N'Ko Benchmark.

Implements Gemini 3 Pro and Gemini 3 Flash for benchmarking.
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
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GoogleProvider(BaseProvider):
    """
    Google Gemini provider for N'Ko benchmarks.
    
    Supports:
    - Gemini 3 Pro (gemini-3-pro)
    - Gemini 3 Flash (gemini-3-flash)
    """
    
    def __init__(self, model_id: str = "gemini-3-pro", api_key: Optional[str] = None):
        super().__init__(model_id, api_key)
        
        # Get actual model ID from config
        model_config = MODELS.get(model_id)
        self.actual_model_id = model_config.model_id if model_config else model_id
        self.cost_per_1k_input = model_config.cost_per_1k_input if model_config else 2.5
        self.cost_per_1k_output = model_config.cost_per_1k_output if model_config else 10.0
        
        # Initialize client
        self._api_key = api_key or get_api_key("google")
        self._model = None
        
        if HAS_GENAI and self._api_key:
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(self.actual_model_id)
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    @property
    def is_available(self) -> bool:
        return HAS_GENAI and self._model is not None
    
    def _call_api(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system: Optional[str] = None,
    ) -> tuple[str, int, int, float]:
        """
        Call the Gemini API.
        
        Returns:
            Tuple of (response_text, input_tokens, output_tokens, latency_ms)
        """
        if not self.is_available:
            raise RuntimeError("Gemini client not available")
        
        start_time = time.time()
        
        # Combine system and user prompts
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        response = self._model.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response text
        response_text = response.text if response.text else ""
        
        # Token usage (estimate based on characters if not provided)
        # Gemini doesn't always provide token counts in the same way
        input_tokens = len(full_prompt) // 4  # Rough estimate
        output_tokens = len(response_text) // 4
        
        # Try to get actual usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                input_tokens = usage.prompt_token_count
            if hasattr(usage, 'candidates_token_count'):
                output_tokens = usage.candidates_token_count
        
        return response_text, input_tokens, output_tokens, latency_ms
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """Translate text using Gemini."""
        if not self.is_available:
            return TranslationResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="Gemini client not available",
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
        """Explain N'Ko text using Gemini."""
        if not self.is_available:
            return ExplanationResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="Gemini client not available",
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
        """General completion using Gemini."""
        if not self.is_available:
            return CompletionResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="Gemini client not available",
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

