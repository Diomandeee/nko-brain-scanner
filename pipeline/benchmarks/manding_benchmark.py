#!/usr/bin/env python3
"""
Comprehensive Manding Language AI Model Benchmark Pipeline

Extends the N'Ko benchmark to evaluate AI models across the full Manding language family:
- Bidirectional translation between all language pairs (12 directions)
- Cross-Manding evaluation (N'Ko ↔ Bambara, dialectal variations)
- Curriculum-based progressive difficulty (CEFR A1-C2)
- Complex tests (proverbs, compound words, cognates)

Usage:
    # Full multilingual benchmark
    python -m training.benchmarks.manding_benchmark --all --multilingual
    
    # Specific language pairs
    python -m training.benchmarks.manding_benchmark --pairs nko-bambara bambara-french
    
    # Curriculum mode
    python -m training.benchmarks.manding_benchmark --curriculum --levels A1 A2 B1
    
    # Quick cross-language test
    python -m training.benchmarks.manding_benchmark --quick --cross-manding
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .config import (
    MODELS,
    PROVIDERS,
    LANGUAGES,
    TRANSLATION_PAIRS,
    CROSS_MANDING_PAIRS,
    CURRICULUM_LEVELS,
    BenchmarkConfig,
    get_quick_config,
    get_medium_config,
    get_full_config,
    get_multilingual_config,
    get_curriculum_config,
    REPORTS_DIR,
)
from .data.manding_loader import (
    MandingDataLoader,
    MandingTestSet,
    Language,
    create_manding_test_set,
)
from .data.complex_tests import ComplexTestGenerator
from .providers.base import BaseProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider import OpenAIProvider
from .providers.google_provider import GoogleProvider
from .tasks.translation import TranslationTask
from .tasks.cross_language import (
    CrossMandingTranslationTask,
    ScriptTransliterationTask,
    DialectIdentificationTask,
    CognateRecognitionTask,
    calculate_cross_language_metrics,
)
from .tasks.curriculum import (
    CurriculumTask,
    CurriculumTestGenerator,
    CEFRLevel,
    calculate_curriculum_metrics,
)
from .metrics.composite_scorer import CompositeScorer
from .reports.manding_report import (
    MandingReportGenerator,
    generate_manding_benchmark_report,
)


class MandingBenchmark:
    """
    Comprehensive Manding Language AI Model Benchmark orchestrator.
    
    Coordinates:
    - Multilingual data loading (nicolingua, Bayelemabaga, unified corpus, Supabase)
    - Cross-language task execution
    - Curriculum-based evaluation
    - Multi-model comparison
    - Report generation with language pair breakdowns
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or get_multilingual_config()
        self.scorer = CompositeScorer()
        self.report_generator = MandingReportGenerator()
        self.data_loader = MandingDataLoader()
        
        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {}
        self.testset: Optional[MandingTestSet] = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_provider(self, model_key: str) -> Optional[BaseProvider]:
        """Create provider instance for a model."""
        if model_key not in MODELS:
            print(f"  Warning: Unknown model '{model_key}'")
            return None
        
        model_config = MODELS[model_key]
        
        try:
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
        except Exception as e:
            print(f"  Warning: Failed to create provider for {model_key}: {e}")
            return None
    
    def _progress_callback(self, current: int, total: int, task: str = ""):
        """Print progress updates."""
        if total > 0:
            pct = (current / total) * 100
            print(f"    [{task}] {current}/{total} ({pct:.0f}%)", end="\r")
    
    async def load_data(self) -> MandingTestSet:
        """Load and prepare all Manding language test data."""
        print("\n📂 Loading Manding language data...")
        
        self.testset = self.data_loader.load_all(
            nko_translation_limit=self.config.translation_samples,
            bambara_limit=self.config.translation_samples // 2,
            vocab_limit=self.config.vocabulary_samples,
            # IMPORTANT: `load_data()` runs inside an event loop, so we must not call
            # any sync wrapper that uses `asyncio.run()` (it will raise and drop data).
            include_supabase=False,
        )

        # Load Supabase data asynchronously (safe inside event loop)
        try:
            supabase_vocab, supabase_trans = await self.data_loader.load_supabase_data()
            self.testset.nko_vocabulary.extend(supabase_vocab)
            self.testset.nko_translations.extend(supabase_trans)
        except Exception as e:
            print(f"    Warning: Supabase load failed (continuing without it): {e}")
        
        return self.testset
    
    async def run_translation_tasks(
        self,
        provider: BaseProvider,
        model_key: str,
        language_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Run translation tasks for specified language pairs using CrossMandingTranslationTask."""
        results = {}
        pairs = language_pairs or TRANSLATION_PAIRS
        
        cross_task = CrossMandingTranslationTask()
        
        for source, target in pairs:
            pair_key = f"{source}_{target}"
            print(f"    Testing {source} → {target}...")
            
            # Get appropriate test data for this pair - USE FULL DATASET
            # Sample size based on config, not hardcoded 50
            sample_size = self.config.translation_samples
            test_pairs = []
            
            if source == "nko" and target in ["english", "french"]:
                test_pairs = self.testset.nko_translations[:sample_size]
            elif source in ["english", "french"] and target == "nko":
                # Reverse the pairs for back-translation
                test_pairs = self.testset.nko_translations[:sample_size]
            elif source == "bambara" and target == "french":
                test_pairs = self.testset.bambara_french_pairs[:sample_size]
            elif source == "french" and target == "bambara":
                test_pairs = self.testset.bambara_french_pairs[:sample_size]
            elif source == "bambara" and target == "english":
                test_pairs = self.testset.bambara_english_pairs[:sample_size]
            elif source == "english" and target == "bambara":
                test_pairs = self.testset.bambara_english_pairs[:sample_size]
            elif source == "nko" and target == "bambara":
                # Use cognates for N'Ko-Bambara
                test_pairs = self.testset.nko_bambara_cognates[:sample_size]
            elif source == "bambara" and target == "nko":
                test_pairs = self.testset.nko_bambara_cognates[:sample_size]
            
            if not test_pairs:
                print(f"      No data available for {source} → {target}")
                continue
            
            try:
                # Map string language names to Language enum
                lang_map = {
                    "nko": Language.NKO,
                    "bambara": Language.BAMBARA,
                    "english": Language.ENGLISH,
                    "french": Language.FRENCH,
                    "malinke": Language.MALINKE,
                    "jula": Language.JULA,
                }
                
                source_lang = lang_map.get(source)
                target_lang = lang_map.get(target)
                
                if not source_lang or not target_lang:
                    print(f"      Unknown language: {source} or {target}")
                    continue
                
                # Run cross-language translation task
                task_results = await cross_task.run(
                    provider=provider,
                    translation_pairs=test_pairs,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    progress_callback=self._progress_callback,
                )
                
                metrics = calculate_cross_language_metrics(task_results)
                
                # SAVE DETAILED RESULTS including all predictions!
                detailed_samples = []
                # task_results is a List[CrossLanguageResult]
                result_list = task_results.results if hasattr(task_results, 'results') else task_results
                if isinstance(result_list, list):
                    for r in result_list:
                        detailed_samples.append({
                            "source_text": getattr(r, 'source_text', ''),
                            "expected": getattr(r, 'reference_text', getattr(r, 'expected', '')),
                            "predicted": getattr(r, 'prediction', getattr(r, 'predicted', '')),
                            "success": getattr(r, 'is_correct', getattr(r, 'success', False)),
                            "partial_match": getattr(r, 'partial_match', 0.0),
                            "latency_ms": getattr(r, 'latency_ms', 0),
                        })
                
                results[pair_key] = {
                    "source_lang": source,
                    "target_lang": target,
                    "accuracy": metrics.get("accuracy", 0),
                    "partial_match": metrics.get("partial_match", 0),
                    "chrf": metrics.get("partial_match", 0),
                    "count": metrics.get("count", 0),
                    "errors": metrics.get("error_count", 0),
                    "detailed_samples": detailed_samples,  # SAVE ALL PREDICTIONS!
                }
                print(f"      {pair_key}: accuracy={metrics.get('accuracy', 0):.1f}%, partial={metrics.get('partial_match', 0):.1f}%")
                
            except Exception as e:
                print(f"      Error in {pair_key}: {e}")
                results[pair_key] = {"error": str(e)}
        
        print()  # New line after progress
        return results
    
    async def run_cross_language_tasks(
        self,
        provider: BaseProvider,
        model_key: str,
    ) -> Dict[str, Any]:
        """Run cross-Manding language tasks."""
        results = {}
        
        # Script transliteration
        print("    Testing N'Ko ↔ Latin transliteration...")
        script_task = ScriptTransliterationTask()
        vocab_with_latin = [v for v in self.testset.nko_vocabulary if v.latin_transcription][:50]
        
        if vocab_with_latin:
            try:
                script_results = await script_task.run(
                    provider=provider,
                    vocabulary=vocab_with_latin,
                    direction="nko_to_latin",
                    progress_callback=self._progress_callback,
                )
                results["script_transliteration"] = calculate_cross_language_metrics(script_results)
                print(f"      Script: {results['script_transliteration']['accuracy']:.1f}%")
            except Exception as e:
                print(f"      Script error: {e}")
        
        # Dialect identification
        print("    Testing dialect identification...")
        dialect_task = DialectIdentificationTask()
        try:
            dialect_results = await dialect_task.run(
                provider=provider,
                pairs=self.testset.bambara_french_pairs[:50],
                progress_callback=self._progress_callback,
            )
            results["dialect_identification"] = calculate_cross_language_metrics(dialect_results)
            print(f"      Dialect ID: {results['dialect_identification']['accuracy']:.1f}%")
        except Exception as e:
            print(f"      Dialect ID error: {e}")
        
        # Cognate recognition
        if self.testset.nko_bambara_cognates:
            print("    Testing cognate recognition...")
            cognate_task = CognateRecognitionTask()
            try:
                cognate_results = await cognate_task.run(
                    provider=provider,
                    cognates=self.testset.nko_bambara_cognates[:50],
                    progress_callback=self._progress_callback,
                )
                results["cognate_recognition"] = calculate_cross_language_metrics(cognate_results)
                print(f"      Cognates: {results['cognate_recognition']['accuracy']:.1f}%")
            except Exception as e:
                print(f"      Cognate error: {e}")
        
        print()
        return results
    
    async def run_curriculum_tasks(
        self,
        provider: BaseProvider,
        model_key: str,
        levels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run curriculum-based progressive difficulty tests."""
        results = {}
        levels = levels or self.config.curriculum_levels
        
        # Generate curriculum tests
        curriculum_gen = CurriculumTestGenerator(self.testset)
        curriculum_task = CurriculumTask()
        
        for level in levels:
            try:
                cefr_level = CEFRLevel(level)
                print(f"    Testing level {level} ({CURRICULUM_LEVELS[level]['name']})...")
                
                test_items = curriculum_gen.generate_level_tests(
                    level=cefr_level,
                    count=self.config.curriculum_samples_per_level,
                )
                
                if not test_items:
                    print(f"      No tests generated for {level}")
                    continue
                
                level_results = await curriculum_task.run(
                    provider=provider,
                    test_items=test_items,
                    progress_callback=self._progress_callback,
                )
                
                metrics = calculate_curriculum_metrics(level_results)
                results[level] = {
                    "accuracy": metrics.get("overall_accuracy", 0),
                    "partial_score": metrics.get("overall_partial", 0),
                    "count": metrics.get("count", 0),
                    "errors": metrics.get("error_count", 0),
                }
                print(f"      {level}: {results[level]['accuracy']:.1f}%")
                
            except Exception as e:
                print(f"      Error in {level}: {e}")
                results[level] = {"error": str(e)}
        
        print()
        return results
    
    async def run_model_benchmark(
        self,
        model_key: str,
        tasks: List[str],
        language_pairs: Optional[List[Tuple[str, str]]] = None,
        curriculum_levels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run complete benchmark for a single model."""
        model_config = MODELS[model_key]
        print(f"\n🤖 Benchmarking: {model_config.name}")
        
        provider = self._create_provider(model_key)
        if not provider:
            return {"error": "Provider not available"}
        
        results = {
            "model_name": model_config.name,
            "model_id": model_config.model_id,
            "provider": model_config.provider,
        }
        
        # Run requested tasks
        if "translation" in tasks:
            print("  📝 Translation tasks:")
            results["translation"] = await self.run_translation_tasks(
                provider, model_key, language_pairs
            )
        
        if "cross_language" in tasks:
            print("  🔄 Cross-language tasks:")
            results["cross_language"] = await self.run_cross_language_tasks(
                provider, model_key
            )
        
        if "curriculum" in tasks:
            print("  📚 Curriculum tasks:")
            results["curriculum"] = await self.run_curriculum_tasks(
                provider, model_key, curriculum_levels
            )
        
        # Calculate aggregate scores
        total_samples = 0
        total_errors = 0
        
        for task_results in results.values():
            if isinstance(task_results, dict):
                for pair_results in task_results.values():
                    if isinstance(pair_results, dict):
                        total_samples += pair_results.get("count", 0)
                        total_errors += pair_results.get("errors", pair_results.get("error_count", 0))
        
        results["total_samples"] = total_samples
        results["total_errors"] = total_errors
        
        # Estimate cost
        stats = provider.get_statistics() if hasattr(provider, "get_statistics") else {}
        input_tokens = stats.get("total_tokens_input", 0)
        output_tokens = stats.get("total_tokens_output", 0)
        
        results["estimated_cost"] = (
            (input_tokens / 1000) * model_config.cost_per_1k_input +
            (output_tokens / 1000) * model_config.cost_per_1k_output
        )
        results["avg_latency_ms"] = stats.get("avg_latency_ms", 0)
        
        return results
    
    async def run(
        self,
        model_keys: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        language_pairs: Optional[List[Tuple[str, str]]] = None,
        curriculum_levels: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run full multilingual Manding language benchmark.
        
        Args:
            model_keys: List of model identifiers to test
            tasks: List of tasks to run
            language_pairs: Specific language pairs to test
            curriculum_levels: CEFR levels to test
            
        Returns:
            Complete benchmark results for all models
        """
        print("=" * 70)
        print("   MANDING LANGUAGE AI MODEL BENCHMARK")
        print("   Testing N'Ko, Bambara, and Cross-Manding capabilities")
        print("=" * 70)
        
        # Default values
        if model_keys is None:
            # Default: one strong model per provider (API-verified IDs in config)
            model_keys = ["claude-4.5-sonnet", "gpt-5.2", "gemini-3-flash"]
        
        if tasks is None:
            tasks = ["translation", "cross_language", "curriculum"]
        
        print(f"\n📊 Configuration:")
        print(f"   Models: {', '.join(model_keys)}")
        print(f"   Tasks: {', '.join(tasks)}")
        
        if language_pairs:
            pairs_str = ", ".join([f"{s}→{t}" for s, t in language_pairs])
            print(f"   Language pairs: {pairs_str}")
        
        if curriculum_levels:
            print(f"   Curriculum levels: {', '.join(curriculum_levels)}")
        
        # Load data
        await self.load_data()
        
        print(f"\n📈 Test data loaded:")
        print(f"   N'Ko translations: {len(self.testset.nko_translations):,}")
        print(f"   N'Ko vocabulary: {len(self.testset.nko_vocabulary):,}")
        print(f"   Bambara-French pairs: {len(self.testset.bambara_french_pairs):,}")
        print(f"   Cognate pairs: {len(self.testset.nko_bambara_cognates):,}")
        
        # Run benchmark for each model
        for model_key in model_keys:
            try:
                model_results = await self.run_model_benchmark(
                    model_key=model_key,
                    tasks=tasks,
                    language_pairs=language_pairs,
                    curriculum_levels=curriculum_levels,
                )
                self.results[model_key] = model_results
            except Exception as e:
                print(f"  ❌ Error benchmarking {model_key}: {e}")
                self.results[model_key] = {"error": str(e)}
        
        return self.results
    
    def save_results(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save all benchmark results and generate reports."""
        output_dir = output_dir or REPORTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to {output_dir}...")
        
        # Generate comprehensive reports
        reports = generate_manding_benchmark_report(self.results, output_dir)
        
        print(f"   JSON report: {reports['json']}")
        print(f"   Markdown report: {reports['markdown']}")
        print(f"   Heatmap data: {reports['heatmap']}")
        
        # SAVE DETAILED SAMPLES (all predictions) to JSONL
        detailed_path = self._save_detailed_samples(output_dir)
        if detailed_path:
            reports['detailed_samples'] = detailed_path
            print(f"   Detailed samples: {detailed_path}")
        
        return reports
    
    def _save_detailed_samples(self, output_dir: Path) -> Optional[Path]:
        """Save all model predictions to a JSONL file for analysis/training."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"manding_samples_{timestamp}.jsonl"
        
        sample_count = 0
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for model_key, model_results in self.results.items():
                    if isinstance(model_results, dict) and "error" not in model_results:
                        # Extract translation samples
                        if "translation" in model_results:
                            for pair_key, pair_data in model_results["translation"].items():
                                if isinstance(pair_data, dict) and "detailed_samples" in pair_data:
                                    for sample in pair_data["detailed_samples"]:
                                        record = {
                                            "model": model_key,
                                            "task": "translation",
                                            "language_pair": pair_key,
                                            "source_lang": pair_data.get("source_lang"),
                                            "target_lang": pair_data.get("target_lang"),
                                            **sample
                                        }
                                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                        sample_count += 1
            
            if sample_count > 0:
                print(f"   📊 Saved {sample_count} detailed samples to {filepath}")
                return filepath
            else:
                filepath.unlink(missing_ok=True)  # Remove empty file
                return None
                
        except Exception as e:
            print(f"   Warning: Could not save detailed samples: {e}")
            return None
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("   BENCHMARK SUMMARY")
        print("=" * 70)
        
        if not self.results:
            print("   No results to summarize.")
            return
        
        # Create rankings
        model_scores = []
        for model_key, results in self.results.items():
            if "error" in results and isinstance(results["error"], str):
                continue
            
            # Calculate aggregate score
            score = 0.0
            count = 0
            
            # Translation scores
            if "translation" in results:
                for pair_results in results["translation"].values():
                    if isinstance(pair_results, dict) and "partial_match" in pair_results:
                        score += pair_results["partial_match"]
                        count += 1
                    elif isinstance(pair_results, dict) and "chrf" in pair_results:
                        score += pair_results["chrf"]
                        count += 1
            
            # Cross-language scores
            if "cross_language" in results:
                for task_results in results["cross_language"].values():
                    if isinstance(task_results, dict) and "accuracy" in task_results:
                        score += task_results["accuracy"]
                        count += 1
            
            # Curriculum scores
            if "curriculum" in results:
                for level_results in results["curriculum"].values():
                    if isinstance(level_results, dict) and "accuracy" in level_results:
                        score += level_results["accuracy"]
                        count += 1
            
            avg_score = score / count if count > 0 else 0
            model_scores.append((model_key, avg_score, results.get("estimated_cost", 0)))
        
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\n   Model Rankings:")
        print("   " + "-" * 50)
        for rank, (model_key, score, cost) in enumerate(model_scores, 1):
            model_name = MODELS[model_key].name if model_key in MODELS else model_key
            print(f"   {rank}. {model_name}")
            print(f"      Score: {score:.1f} | Est. Cost: ${cost:.2f}")
        
        if model_scores:
            best_model = model_scores[0][0]
            best_name = MODELS[best_model].name if best_model in MODELS else best_model
            print(f"\n   🏆 Recommended: {best_name}")
        
        print("=" * 70)


async def main():
    """Main entry point for Manding benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Manding Language AI Model Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full multilingual benchmark
  python -m training.benchmarks.manding_benchmark --all --multilingual
  
  # Quick test with specific models
  python -m training.benchmarks.manding_benchmark --quick --models claude-4.5-sonnet gpt-4o
  
  # Test specific language pairs
  python -m training.benchmarks.manding_benchmark --pairs nko-english bambara-french
  
  # Curriculum-focused evaluation
  python -m training.benchmarks.manding_benchmark --curriculum --levels A1 A2 B1 B2
        """
    )
    
    # Config presets
    parser.add_argument(
        "--all", action="store_true",
        help="Run all models on all tasks",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick benchmark (~200 samples)",
    )
    parser.add_argument(
        "--medium", action="store_true",
        help="Medium benchmark (~1000 samples)",
    )
    parser.add_argument(
        "--multilingual", action="store_true",
        help="Focus on cross-language evaluation",
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Focus on curriculum-based evaluation",
    )
    parser.add_argument(
        "--comprehensive", action="store_true",
        help="COMPREHENSIVE evaluation using full datasets (nicolingua 130K+, Supabase, Bayelemabaga)",
    )
    
    # Model selection
    parser.add_argument(
        "--providers", nargs="+",
        choices=["anthropic", "openai", "google"],
        help="Providers to test",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODELS.keys()),
        help="Specific models to test",
    )
    
    # Task selection
    parser.add_argument(
        "--tasks", nargs="+",
        choices=["translation", "cross_language", "curriculum"],
        help="Tasks to run",
    )
    
    # Language pair selection
    parser.add_argument(
        "--pairs", nargs="+",
        help="Language pairs to test (e.g., nko-english bambara-french)",
    )
    
    # Curriculum levels
    parser.add_argument(
        "--levels", nargs="+",
        choices=["A1", "A2", "B1", "B2", "C1", "C2"],
        help="CEFR levels to test",
    )
    
    # Output
    parser.add_argument(
        "--output", type=str,
        help="Output directory for reports",
    )
    
    args = parser.parse_args()
    
    # Determine config
    if args.comprehensive:
        from .config import get_comprehensive_config
        config = get_comprehensive_config()
        print("🚀 Running COMPREHENSIVE evaluation with full datasets")
    elif args.quick:
        config = get_quick_config()
    elif args.medium:
        config = get_medium_config()
    elif args.multilingual:
        config = get_multilingual_config()
    elif args.curriculum:
        config = get_curriculum_config()
    else:
        config = get_full_config()
    
    # Determine models
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
        # Default: one model per provider
        model_keys = ["claude-4.5-sonnet", "gpt-5.2", "gemini-3-flash"]
    
    # Parse language pairs
    language_pairs = None
    if args.pairs:
        language_pairs = []
        for pair in args.pairs:
            parts = pair.split("-")
            if len(parts) == 2:
                language_pairs.append((parts[0], parts[1]))
    
    # Run benchmark
    benchmark = MandingBenchmark(config)
    
    await benchmark.run(
        model_keys=model_keys,
        tasks=args.tasks,
        language_pairs=language_pairs,
        curriculum_levels=args.levels,
    )
    
    # Print summary and save results
    benchmark.print_summary()
    
    output_dir = Path(args.output) if args.output else None
    benchmark.save_results(output_dir)


if __name__ == "__main__":
    asyncio.run(main())

