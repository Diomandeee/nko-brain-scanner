#!/usr/bin/env python3
"""
N'Ko AI Model Benchmark Pipeline

Evaluates state-of-the-art AI models (Claude 4.5, GPT-5.2, Gemini 3) on N'Ko
language tasks to determine the optimal base model for the translation system.

Usage:
    python -m training.benchmarks.nko_benchmark --all
    python -m training.benchmarks.nko_benchmark --providers anthropic openai google
    python -m training.benchmarks.nko_benchmark --quick
    python -m training.benchmarks.nko_benchmark --tasks translation vocabulary
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import (
    MODELS,
    PROVIDERS,
    BenchmarkConfig,
    get_quick_config,
    get_medium_config,
    get_full_config,
    REPORTS_DIR,
)
from .data.sampler import TestDataSampler, create_test_dataset
from .data.complex_tests import ComplexTestGenerator, generate_compositional_tests
from .providers.base import BaseProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider import OpenAIProvider
from .providers.google_provider import GoogleProvider
from .tasks.translation import TranslationTask
from .tasks.script_knowledge import ScriptKnowledgeTask
from .tasks.vocabulary import VocabularyTask
from .tasks.cultural import CulturalTask
from .tasks.compositional import CompositionalTask
from .metrics.composite_scorer import CompositeScorer, ModelScore, TaskScore
from .reports.generator import ReportGenerator


class NkoBenchmark:
    """
    Main N'Ko AI Model Benchmark orchestrator.
    
    Coordinates:
    - Data loading and sampling
    - Provider initialization
    - Task execution
    - Metrics calculation
    - Report generation
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.scorer = CompositeScorer()
        self.report_generator = ReportGenerator()
        
        # Results storage
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "translation_samples": self.config.translation_samples,
                "vocabulary_samples": self.config.vocabulary_samples,
                "script_samples": self.config.script_samples,
                "cultural_samples": self.config.cultural_samples,
                "compositional_samples": self.config.compositional_samples,
            },
            "models": {},
            "rankings": [],
            "recommendation": None,
        }
    
    def _create_provider(self, model_key: str) -> Optional[BaseProvider]:
        """Create provider instance for a model."""
        if model_key not in MODELS:
            print(f"  Warning: Unknown model '{model_key}'")
            return None
        
        model_config = MODELS[model_key]
        
        if model_config.provider == "anthropic":
            provider = AnthropicProvider(model_key)
        elif model_config.provider == "openai":
            provider = OpenAIProvider(model_key)
        elif model_config.provider == "google":
            provider = GoogleProvider(model_key)
        else:
            print(f"  Warning: Unknown provider '{model_config.provider}'")
            return None
        
        if not provider.is_available:
            print(f"  Warning: {model_key} not available (check API key)")
            return None
        
        return provider
    
    def _progress_callback(self, **kwargs):
        """Print progress updates."""
        task = kwargs.get("task", "")
        current = kwargs.get("current", 0)
        total = kwargs.get("total", 0)
        
        if total > 0:
            pct = (current / total) * 100
            print(f"    [{task}] {current}/{total} ({pct:.0f}%)", end="\r")
    
    async def run_model_benchmark(
        self,
        model_key: str,
        dataset,
        tasks: List[str],
    ) -> Optional[ModelScore]:
        """
        Run benchmark for a single model.
        
        Args:
            model_key: Model identifier
            dataset: Test dataset
            tasks: List of tasks to run
            
        Returns:
            ModelScore with results, or None if failed
        """
        print(f"\n  Benchmarking: {MODELS[model_key].name}")
        
        provider = self._create_provider(model_key)
        if not provider:
            return None
        
        task_scores = []
        all_latencies = []
        
        # Run translation task
        if "translation" in tasks:
            print("    Running translation tests...")
            trans_task = TranslationTask()
            trans_results = await trans_task.run(
                provider=provider,
                nko_to_en=dataset.nko_to_en,
                nko_to_fr=dataset.nko_to_fr,
                en_to_nko=dataset.en_to_nko,
                fr_to_nko=dataset.fr_to_nko,
                progress_callback=self._progress_callback,
            )
            print()  # New line after progress
            
            # Calculate translation score
            predictions = [r.predicted for r in trans_results.results if r.success]
            references = [r.expected for r in trans_results.results if r.success]
            trans_score = self.scorer.score_translation(predictions, references)
            task_scores.append(trans_score)
            
            if trans_results.avg_latency_ms > 0:
                all_latencies.append(trans_results.avg_latency_ms)
            
            print(f"      Translation: {trans_score.raw_score:.1f} (BLEU: {trans_score.metrics.get('bleu', 0):.1f}, chrF: {trans_score.metrics.get('chrf', 0):.1f})")
        
        # Run script knowledge task
        if "script_knowledge" in tasks:
            print("    Running script knowledge tests...")
            script_task = ScriptKnowledgeTask()
            script_results = await script_task.run(
                provider=provider,
                script_samples=dataset.script_samples,
                progress_callback=self._progress_callback,
            )
            print()
            
            predictions = [r.predicted for r in script_results.results]
            references = [r.expected or "" for r in script_results.results]
            successes = [r.success for r in script_results.results]
            script_score = self.scorer.score_script_knowledge(predictions, references, successes)
            task_scores.append(script_score)
            
            if script_results.avg_latency_ms > 0:
                all_latencies.append(script_results.avg_latency_ms)
            
            print(f"      Script Knowledge: {script_score.raw_score:.1f}")
        
        # Run vocabulary task
        if "vocabulary" in tasks:
            print("    Running vocabulary tests...")
            vocab_task = VocabularyTask()
            vocab_results = await vocab_task.run(
                provider=provider,
                vocabulary_samples=dataset.vocabulary,
                progress_callback=self._progress_callback,
            )
            print()
            
            predictions = [r.predicted_meaning for r in vocab_results.results]
            references = [r.expected_meaning or "" for r in vocab_results.results]
            successes = [r.success for r in vocab_results.results]
            vocab_score = self.scorer.score_vocabulary(predictions, references, successes)
            task_scores.append(vocab_score)
            
            if vocab_results.avg_latency_ms > 0:
                all_latencies.append(vocab_results.avg_latency_ms)
            
            print(f"      Vocabulary: {vocab_score.raw_score:.1f}")
        
        # Run cultural task
        if "cultural" in tasks:
            print("    Running cultural tests...")
            cultural_task = CulturalTask()
            cultural_results = await cultural_task.run(
                provider=provider,
                cultural_samples=dataset.cultural_samples,
                progress_callback=self._progress_callback,
            )
            print()
            
            predictions = [r.predicted_explanation for r in cultural_results.results]
            references = [r.expected_meaning or "" for r in cultural_results.results]
            successes = [r.success for r in cultural_results.results]
            cultural_score = self.scorer.score_cultural(predictions, references, successes)
            task_scores.append(cultural_score)
            
            if cultural_results.avg_latency_ms > 0:
                all_latencies.append(cultural_results.avg_latency_ms)
            
            print(f"      Cultural: {cultural_score.raw_score:.1f}")
        
        # Run compositional task
        if "compositional" in tasks:
            print("    Running compositional tests...")
            comp_task = CompositionalTask()
            comp_results = await comp_task.run(
                provider=provider,
                compositional_samples=dataset.compositional_samples,
                progress_callback=self._progress_callback,
            )
            print()
            
            predictions = [r.predicted for r in comp_results.results]
            references = [r.expected for r in comp_results.results]
            successes = [r.success for r in comp_results.results]
            comp_score = self.scorer.score_compositional(predictions, references, successes)
            task_scores.append(comp_score)
            
            if comp_results.avg_latency_ms > 0:
                all_latencies.append(comp_results.avg_latency_ms)
            
            print(f"      Compositional: {comp_score.raw_score:.1f}")
        
        # Calculate overall score
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        stats = provider.get_statistics()
        
        # Estimate cost
        model_config = MODELS[model_key]
        cost = (
            (stats["total_tokens_input"] / 1000) * model_config.cost_per_1k_input +
            (stats["total_tokens_output"] / 1000) * model_config.cost_per_1k_output
        )
        
        model_score = self.scorer.create_model_score(
            model_id=model_key,
            provider=model_config.provider,
            task_scores=task_scores,
            avg_latency_ms=avg_latency,
            total_cost=cost,
        )
        
        print(f"    Overall: {model_score.overall_score:.1f} | Latency: {avg_latency:.0f}ms | Est. Cost: ${cost:.2f}")
        
        return model_score
    
    async def run(
        self,
        model_keys: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run full benchmark.
        
        Args:
            model_keys: List of model identifiers to test
            tasks: List of tasks to run
            
        Returns:
            Complete benchmark results
        """
        print("=" * 60)
        print("N'Ko AI Model Benchmark")
        print("=" * 60)
        
        # Default to all models and tasks
        if model_keys is None:
            model_keys = list(MODELS.keys())
        
        if tasks is None:
            tasks = ["translation", "script_knowledge", "vocabulary", "cultural", "compositional"]
        
        print(f"\nModels: {', '.join(model_keys)}")
        print(f"Tasks: {', '.join(tasks)}")
        
        # Create test dataset
        print("\nPreparing test data...")
        sampler = TestDataSampler(self.config)
        dataset = sampler.create_dataset()
        
        # Add compositional tests
        if "compositional" in tasks:
            comp_generator = ComplexTestGenerator(self.config)
            dataset.compositional_samples = comp_generator.to_benchmark_format()
            print(f"  Compositional samples: {len(dataset.compositional_samples)}")
        
        print(f"\nTotal test samples: {dataset.total_samples()}")
        
        # Run benchmarks for each model
        model_scores = []
        
        for model_key in model_keys:
            try:
                score = await self.run_model_benchmark(model_key, dataset, tasks)
                if score:
                    model_scores.append(score)
                    self.results["models"][model_key] = score.to_dict()
            except Exception as e:
                print(f"  Error benchmarking {model_key}: {e}")
        
        # Generate rankings and recommendation
        if model_scores:
            recommendation = self.scorer.get_recommendation(model_scores)
            self.results["rankings"] = recommendation["all_rankings"]
            self.results["recommendation"] = recommendation
            
            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            
            print("\nRankings:")
            for rank_info in recommendation["all_rankings"]:
                print(f"  {rank_info['rank']}. {rank_info['model_id']} ({rank_info['provider']}): {rank_info['overall_score']:.1f}")
            
            print(f"\nRecommendation: {recommendation['recommendation']}")
            print(f"  Reason: {recommendation['best_overall']['reason']}")
        
        return self.results
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save benchmark results to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = REPORTS_DIR / f"nko_benchmark_{timestamp}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        
        # Also generate markdown report
        md_path = output_path.with_suffix('.md')
        self.report_generator.generate_markdown(self.results, md_path)
        print(f"Markdown report: {md_path}")
        
        return str(output_path)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="N'Ko AI Model Benchmark Pipeline"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models on all tasks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark (50 samples)",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Medium benchmark (500 samples)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["anthropic", "openai", "google"],
        help="Providers to test",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Specific models to test",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["translation", "script_knowledge", "vocabulary", "cultural", "compositional"],
        help="Tasks to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )
    
    args = parser.parse_args()
    
    # Determine config
    if args.quick:
        config = get_quick_config()
    elif args.medium:
        config = get_medium_config()
    else:
        config = get_full_config()
    
    # Determine models to test
    model_keys = None
    if args.models:
        model_keys = args.models
    elif args.providers:
        model_keys = []
        for provider in args.providers:
            if provider in PROVIDERS:
                model_keys.extend(PROVIDERS[provider])
    elif args.all:
        model_keys = list(MODELS.keys())
    else:
        # Default: one model per provider (latest 2025 models)
        model_keys = ["claude-4.5-sonnet", "gpt-5.2", "gemini-3-flash"]
    
    # Run benchmark
    benchmark = NkoBenchmark(config)
    results = await benchmark.run(
        model_keys=model_keys,
        tasks=args.tasks,
    )
    
    # Save results
    benchmark.save_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())

