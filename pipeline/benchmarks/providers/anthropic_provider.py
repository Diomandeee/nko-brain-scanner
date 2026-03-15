"""
Anthropic Claude Provider for N'Ko Benchmark.

Implements Claude 4.5 Sonnet and Claude Opus 4.1 for benchmarking.
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
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude provider for N'Ko benchmarks.
    
    Supports:
    - Claude 4.5 Sonnet (claude-sonnet-4-5-20250929)
    - Claude Opus 4.1 (claude-opus-4-1-20250828)
    """
    
    def __init__(self, model_id: str = "claude-sonnet-4-5", api_key: Optional[str] = None):
        super().__init__(model_id, api_key)
        
        # Get actual model ID from config
        model_config = MODELS.get(model_id)
        self.actual_model_id = model_config.model_id if model_config else model_id
        self.cost_per_1k_input = model_config.cost_per_1k_input if model_config else 3.0
        self.cost_per_1k_output = model_config.cost_per_1k_output if model_config else 15.0
        
        # Initialize client
        self._api_key = api_key or get_api_key("anthropic")
        self._client = None
        
        if HAS_ANTHROPIC and self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def is_available(self) -> bool:
        return HAS_ANTHROPIC and self._client is not None
    
    def _call_api(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system: Optional[str] = None,
    ) -> tuple[str, int, int, float]:
        """
        Call the Anthropic API.
        
        Returns:
            Tuple of (response_text, input_tokens, output_tokens, latency_ms)
        """
        if not self.is_available:
            raise RuntimeError("Anthropic client not available")
        
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.actual_model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        
        if system:
            kwargs["system"] = system
        
        response = self._client.messages.create(**kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response text
        response_text = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text
        
        # Token usage
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        
        return response_text, input_tokens, output_tokens, latency_ms
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """Translate text using Claude."""
        if not self.is_available:
            return TranslationResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="Anthropic client not available",
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
        """Explain N'Ko text using Claude."""
        if not self.is_available:
            return ExplanationResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="Anthropic client not available",
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
        """General completion using Claude."""
        if not self.is_available:
            return CompletionResult(
                success=False,
                response="",
                latency_ms=0,
                model_id=self.model_id,
                error="Anthropic client not available",
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

