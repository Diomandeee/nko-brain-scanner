"""
Base Provider Interface for N'Ko Benchmark.

Abstract interface that all AI providers must implement.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class TaskType(Enum):
    """Types of benchmark tasks."""
    TRANSLATION = "translation"
    SCRIPT_KNOWLEDGE = "script_knowledge"
    VOCABULARY = "vocabulary"
    CULTURAL = "cultural"
    COMPOSITIONAL = "compositional"


@dataclass
class ProviderResult:
    """Result from an AI provider call."""
    success: bool
    response: str
    latency_ms: float
    model_id: str
    tokens_input: int = 0
    tokens_output: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cost_estimate(self) -> float:
        """Estimate cost based on token usage."""
        # Will be calculated using model-specific pricing
        return 0.0


@dataclass
class TranslationResult(ProviderResult):
    """Result from a translation task."""
    source_text: str = ""
    source_lang: str = ""
    target_lang: str = ""
    translation: str = ""


@dataclass
class ExplanationResult(ProviderResult):
    """Result from an explanation/definition task."""
    nko_text: str = ""
    explanation: str = ""
    examples: List[str] = field(default_factory=list)
    word_class: Optional[str] = None


@dataclass
class CompletionResult(ProviderResult):
    """Result from a general completion task."""
    prompt: str = ""
    completion: str = ""


class BaseProvider(ABC):
    """
    Abstract base class for AI providers.
    
    All providers (Anthropic, OpenAI, Google) must implement this interface
    to ensure consistent benchmarking across models.
    """
    
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key
        self._total_tokens_input = 0
        self._total_tokens_output = 0
        self._total_calls = 0
        self._total_latency_ms = 0.0
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'openai', 'google')."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass
    
    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translate text between languages.
        
        Args:
            text: Source text to translate
            source_lang: Source language code (nko, en, fr)
            target_lang: Target language code (nko, en, fr)
            
        Returns:
            TranslationResult with translation and metadata
        """
        pass
    
    @abstractmethod
    async def explain_nko(
        self,
        nko_text: str,
        include_examples: bool = True,
    ) -> ExplanationResult:
        """
        Explain/define N'Ko text.
        
        Args:
            nko_text: N'Ko text to explain
            include_examples: Whether to include usage examples
            
        Returns:
            ExplanationResult with explanation and examples
        """
        pass
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> CompletionResult:
        """
        General text completion.
        
        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            CompletionResult with completion text
        """
        pass
    
    async def run_task(
        self,
        task_type: TaskType,
        task_data: Dict[str, Any],
    ) -> ProviderResult:
        """
        Run a benchmark task of the specified type.
        
        Args:
            task_type: Type of task to run
            task_data: Task-specific data
            
        Returns:
            ProviderResult with task results
        """
        start_time = time.time()
        
        try:
            if task_type == TaskType.TRANSLATION:
                result = await self.translate(
                    text=task_data.get("text", ""),
                    source_lang=task_data.get("source_lang", "nko"),
                    target_lang=task_data.get("target_lang", "en"),
                )
            
            elif task_type == TaskType.SCRIPT_KNOWLEDGE:
                # Script knowledge uses explanation
                result = await self.explain_nko(
                    nko_text=task_data.get("nko_text", ""),
                    include_examples=False,
                )
            
            elif task_type == TaskType.VOCABULARY:
                result = await self.explain_nko(
                    nko_text=task_data.get("word", ""),
                    include_examples=True,
                )
            
            elif task_type == TaskType.CULTURAL:
                # Cultural uses general completion
                result = await self.complete(
                    prompt=task_data.get("prompt", ""),
                )
            
            elif task_type == TaskType.COMPOSITIONAL:
                result = await self.complete(
                    prompt=task_data.get("prompt", ""),
                )
            
            else:
                result = CompletionResult(
                    success=False,
                    response="",
                    latency_ms=0,
                    model_id=self.model_id,
                    error=f"Unknown task type: {task_type}",
                )
            
            # Update statistics
            self._total_calls += 1
            self._total_latency_ms += result.latency_ms
            self._total_tokens_input += result.tokens_input
            self._total_tokens_output += result.tokens_output
            
            return result
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ProviderResult(
                success=False,
                response="",
                latency_ms=elapsed_ms,
                model_id=self.model_id,
                error=str(e),
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get provider usage statistics."""
        return {
            "provider": self.provider_name,
            "model_id": self.model_id,
            "total_calls": self._total_calls,
            "total_tokens_input": self._total_tokens_input,
            "total_tokens_output": self._total_tokens_output,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": self._total_latency_ms / max(1, self._total_calls),
        }
    
    def reset_statistics(self):
        """Reset usage statistics."""
        self._total_tokens_input = 0
        self._total_tokens_output = 0
        self._total_calls = 0
        self._total_latency_ms = 0.0


def create_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Create a standardized translation prompt."""
    lang_names = {
        "nko": "N'Ko",
        "en": "English",
        "fr": "French",
    }
    
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    return f"""Translate this {source_name} text to {target_name}.

{source_name}: {text}

Provide only the {target_name} translation, nothing else."""


def create_explanation_prompt(nko_text: str, include_examples: bool = True) -> str:
    """Create a standardized explanation prompt for N'Ko text."""
    examples_instruction = ""
    if include_examples:
        examples_instruction = "\n- 2-3 example sentences using this word/phrase"
    
    return f"""Explain this N'Ko text: {nko_text}

Provide:
- The meaning in English
- The word class (noun, verb, adjective, etc.) if applicable
- The Latin transliteration{examples_instruction}

Be concise and accurate."""


def create_script_knowledge_prompt(nko_text: str) -> str:
    """Create a prompt for script knowledge tasks."""
    return f"""Analyze this N'Ko script: {nko_text}

Provide:
1. Latin transliteration
2. Meaning in English
3. Any notable features of the N'Ko characters used

Be concise and accurate."""

