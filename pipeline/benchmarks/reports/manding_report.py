"""
Manding Language Benchmark Report Generator.

Generates comprehensive reports for cross-language AI model evaluation:
- Per-language-pair BLEU/chrF++ scores
- Curriculum progression analysis
- Model comparison heatmaps
- Best model recommendation by task type
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..config import (
    LANGUAGES,
    TRANSLATION_PAIRS,
    CURRICULUM_LEVELS,
    REPORTS_DIR,
    LanguageCode,
)


@dataclass
class LanguagePairScore:
    """Score for a specific language pair."""
    source_lang: str
    target_lang: str
    bleu: float = 0.0
    chrf: float = 0.0
    accuracy: float = 0.0
    sample_count: int = 0
    error_count: int = 0


@dataclass
class CurriculumScore:
    """Score for a CEFR curriculum level."""
    level: str
    accuracy: float = 0.0
    partial_score: float = 0.0
    sample_count: int = 0
    error_count: int = 0


@dataclass
class ModelBenchmarkScore:
    """Complete benchmark score for a model."""
    model_name: str
    model_id: str
    provider: str
    
    # Overall scores
    overall_score: float = 0.0
    overall_translation_score: float = 0.0
    overall_cross_language_score: float = 0.0
    overall_curriculum_score: float = 0.0
    
    # Per-language-pair scores
    language_pair_scores: Dict[str, LanguagePairScore] = field(default_factory=dict)
    
    # Curriculum scores by level
    curriculum_scores: Dict[str, CurriculumScore] = field(default_factory=dict)
    
    # Task-specific scores
    task_scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    total_samples: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    estimated_cost: float = 0.0


class MandingReportGenerator:
    """
    Generates comprehensive reports for Manding language benchmark results.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_model_score(
        self,
        model_name: str,
        model_id: str,
        provider: str,
        results: Dict[str, Any],
    ) -> ModelBenchmarkScore:
        """
        Generate comprehensive score for a single model.
        
        Args:
            model_name: Display name of the model
            model_id: API model identifier
            provider: Provider name (anthropic, openai, google)
            results: Raw results dictionary from benchmark run
            
        Returns:
            ModelBenchmarkScore with all metrics
        """
        score = ModelBenchmarkScore(
            model_name=model_name,
            model_id=model_id,
            provider=provider,
        )
        
        # Extract translation scores by language pair
        if "translation" in results:
            for pair_key, pair_results in results["translation"].items():
                if isinstance(pair_results, dict):
                    score.language_pair_scores[pair_key] = LanguagePairScore(
                        source_lang=pair_results.get("source_lang", ""),
                        target_lang=pair_results.get("target_lang", ""),
                        bleu=pair_results.get("bleu", 0.0),
                        chrf=pair_results.get("chrf", 0.0),
                        accuracy=pair_results.get("accuracy", 0.0),
                        sample_count=pair_results.get("count", 0),
                        error_count=pair_results.get("errors", 0),
                    )
        
        # Extract curriculum scores by level
        if "curriculum" in results:
            for level, level_results in results["curriculum"].items():
                if isinstance(level_results, dict):
                    score.curriculum_scores[level] = CurriculumScore(
                        level=level,
                        accuracy=level_results.get("accuracy", 0.0),
                        partial_score=level_results.get("partial_score", 0.0),
                        sample_count=level_results.get("count", 0),
                        error_count=level_results.get("errors", 0),
                    )
        
        # Extract task-specific scores
        for task in ["script_knowledge", "vocabulary", "cultural", "compositional", "cross_language"]:
            if task in results:
                task_data = results[task]
                if isinstance(task_data, dict):
                    score.task_scores[task] = task_data.get("accuracy", task_data.get("score", 0.0))
                elif isinstance(task_data, (int, float)):
                    score.task_scores[task] = float(task_data)
        
        # Calculate overall scores
        score.overall_translation_score = self._calculate_translation_average(score.language_pair_scores)
        score.overall_curriculum_score = self._calculate_curriculum_average(score.curriculum_scores)
        score.overall_cross_language_score = score.task_scores.get("cross_language", 0.0)
        
        # Weighted overall score
        score.overall_score = self._calculate_weighted_overall(score)
        
        # Metadata
        score.total_samples = results.get("total_samples", 0)
        score.total_errors = results.get("total_errors", 0)
        score.avg_latency_ms = results.get("avg_latency_ms", 0.0)
        score.estimated_cost = results.get("estimated_cost", 0.0)
        
        return score
    
    def _calculate_translation_average(
        self, pair_scores: Dict[str, LanguagePairScore]
    ) -> float:
        """Calculate average translation score across all language pairs."""
        if not pair_scores:
            return 0.0
        
        total_chrf = sum(ps.chrf for ps in pair_scores.values())
        return total_chrf / len(pair_scores)
    
    def _calculate_curriculum_average(
        self, curriculum_scores: Dict[str, CurriculumScore]
    ) -> float:
        """Calculate weighted average curriculum score (harder levels weighted more)."""
        if not curriculum_scores:
            return 0.0
        
        level_weights = {"A1": 1.0, "A2": 1.2, "B1": 1.5, "B2": 1.8, "C1": 2.0, "C2": 2.5}
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for level, cs in curriculum_scores.items():
            weight = level_weights.get(level, 1.0)
            weighted_sum += cs.accuracy * weight
            weight_total += weight
        
        return weighted_sum / weight_total if weight_total > 0 else 0.0
    
    def _calculate_weighted_overall(self, score: ModelBenchmarkScore) -> float:
        """Calculate weighted overall score."""
        weights = {
            "translation": 0.35,
            "cross_language": 0.20,
            "curriculum": 0.15,
            "script_knowledge": 0.10,
            "vocabulary": 0.10,
            "cultural": 0.05,
            "compositional": 0.05,
        }
        
        total = 0.0
        total += score.overall_translation_score * weights["translation"]
        total += score.overall_cross_language_score * weights["cross_language"]
        total += score.overall_curriculum_score * weights["curriculum"]
        
        for task, weight in weights.items():
            if task in score.task_scores:
                total += score.task_scores[task] * weight
        
        return total
    
    def generate_comparison_report(
        self,
        model_scores: List[ModelBenchmarkScore],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report across all models.
        
        Args:
            model_scores: List of ModelBenchmarkScore for each tested model
            
        Returns:
            Dictionary with complete comparison data
        """
        # Sort models by overall score
        ranked_models = sorted(model_scores, key=lambda x: x.overall_score, reverse=True)
        
        report = {
            "timestamp": self.timestamp,
            "total_models": len(model_scores),
            "rankings": [],
            "by_language_pair": {},
            "by_curriculum_level": {},
            "by_task": {},
            "recommendations": {},
        }
        
        # Overall rankings
        for rank, ms in enumerate(ranked_models, 1):
            report["rankings"].append({
                "rank": rank,
                "model_name": ms.model_name,
                "model_id": ms.model_id,
                "provider": ms.provider,
                "overall_score": round(ms.overall_score, 2),
                "translation_score": round(ms.overall_translation_score, 2),
                "cross_language_score": round(ms.overall_cross_language_score, 2),
                "curriculum_score": round(ms.overall_curriculum_score, 2),
            })
        
        # Best model per language pair
        for pair in TRANSLATION_PAIRS:
            pair_key = f"{pair[0]}_{pair[1]}"
            best_model = None
            best_score = -1
            
            for ms in model_scores:
                if pair_key in ms.language_pair_scores:
                    score = ms.language_pair_scores[pair_key].chrf
                    if score > best_score:
                        best_score = score
                        best_model = ms.model_name
            
            if best_model:
                report["by_language_pair"][pair_key] = {
                    "best_model": best_model,
                    "chrf_score": round(best_score, 2),
                }
        
        # Best model per curriculum level
        for level in CURRICULUM_LEVELS:
            best_model = None
            best_score = -1
            
            for ms in model_scores:
                if level in ms.curriculum_scores:
                    score = ms.curriculum_scores[level].accuracy
                    if score > best_score:
                        best_score = score
                        best_model = ms.model_name
            
            if best_model:
                report["by_curriculum_level"][level] = {
                    "best_model": best_model,
                    "accuracy": round(best_score, 2),
                }
        
        # Best model per task
        task_types = ["script_knowledge", "vocabulary", "cultural", "compositional", "cross_language"]
        for task in task_types:
            best_model = None
            best_score = -1
            
            for ms in model_scores:
                if task in ms.task_scores:
                    score = ms.task_scores[task]
                    if score > best_score:
                        best_score = score
                        best_model = ms.model_name
            
            if best_model:
                report["by_task"][task] = {
                    "best_model": best_model,
                    "score": round(best_score, 2),
                }
        
        # Recommendations
        if ranked_models:
            report["recommendations"] = {
                "overall_best": ranked_models[0].model_name,
                "best_for_nko": self._get_best_for_nko(model_scores),
                "best_for_bambara": self._get_best_for_bambara(model_scores),
                "best_for_translation": self._get_best_for_translation(model_scores),
                "best_value": self._get_best_value(model_scores),
            }
        
        return report
    
    def _get_best_for_nko(self, model_scores: List[ModelBenchmarkScore]) -> str:
        """Find best model specifically for N'Ko tasks."""
        best = None
        best_score = -1
        
        for ms in model_scores:
            nko_score = 0.0
            nko_count = 0
            
            for pair_key, ps in ms.language_pair_scores.items():
                if "nko" in pair_key:
                    nko_score += ps.chrf
                    nko_count += 1
            
            if nko_count > 0:
                avg = nko_score / nko_count
                if avg > best_score:
                    best_score = avg
                    best = ms.model_name
        
        return best or "N/A"
    
    def _get_best_for_bambara(self, model_scores: List[ModelBenchmarkScore]) -> str:
        """Find best model specifically for Bambara tasks."""
        best = None
        best_score = -1
        
        for ms in model_scores:
            bam_score = 0.0
            bam_count = 0
            
            for pair_key, ps in ms.language_pair_scores.items():
                if "bambara" in pair_key:
                    bam_score += ps.chrf
                    bam_count += 1
            
            if bam_count > 0:
                avg = bam_score / bam_count
                if avg > best_score:
                    best_score = avg
                    best = ms.model_name
        
        return best or "N/A"
    
    def _get_best_for_translation(self, model_scores: List[ModelBenchmarkScore]) -> str:
        """Find best model for translation tasks."""
        best = None
        best_score = -1
        
        for ms in model_scores:
            if ms.overall_translation_score > best_score:
                best_score = ms.overall_translation_score
                best = ms.model_name
        
        return best or "N/A"
    
    def _get_best_value(self, model_scores: List[ModelBenchmarkScore]) -> str:
        """Find best value model (score/cost ratio)."""
        best = None
        best_ratio = -1
        
        for ms in model_scores:
            if ms.estimated_cost > 0:
                ratio = ms.overall_score / ms.estimated_cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = ms.model_name
        
        return best or model_scores[0].model_name if model_scores else "N/A"
    
    def generate_json_report(
        self,
        comparison: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """Save comparison report as JSON."""
        filename = filename or f"manding_benchmark_{self.timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_markdown_report(
        self,
        comparison: Dict[str, Any],
        model_scores: List[ModelBenchmarkScore],
        filename: Optional[str] = None,
    ) -> Path:
        """Generate comprehensive Markdown report."""
        filename = filename or f"manding_benchmark_{self.timestamp}.md"
        filepath = self.output_dir / filename
        
        lines = []
        
        # Header
        lines.append("# Manding Language AI Model Benchmark Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Models Tested:** {comparison['total_models']}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append("### Overall Rankings")
        lines.append("")
        lines.append("| Rank | Model | Provider | Overall | Translation | Cross-Lang | Curriculum |")
        lines.append("|------|-------|----------|---------|-------------|------------|------------|")
        
        for entry in comparison["rankings"]:
            lines.append(
                f"| {entry['rank']} | {entry['model_name']} | {entry['provider']} | "
                f"{entry['overall_score']:.1f} | {entry['translation_score']:.1f} | "
                f"{entry['cross_language_score']:.1f} | {entry['curriculum_score']:.1f} |"
            )
        
        lines.append("")
        
        # Recommendations
        lines.append("### Recommendations")
        lines.append("")
        recs = comparison.get("recommendations", {})
        lines.append(f"- **Overall Best Model:** {recs.get('overall_best', 'N/A')}")
        lines.append(f"- **Best for N'Ko:** {recs.get('best_for_nko', 'N/A')}")
        lines.append(f"- **Best for Bambara:** {recs.get('best_for_bambara', 'N/A')}")
        lines.append(f"- **Best for Translation:** {recs.get('best_for_translation', 'N/A')}")
        lines.append(f"- **Best Value:** {recs.get('best_value', 'N/A')}")
        lines.append("")
        
        # Per-Language-Pair Results
        lines.append("## Translation Results by Language Pair")
        lines.append("")
        lines.append("| Language Pair | Best Model | chrF++ Score |")
        lines.append("|---------------|------------|--------------|")
        
        for pair_key, data in comparison.get("by_language_pair", {}).items():
            pair_display = pair_key.replace("_", " → ")
            lines.append(f"| {pair_display} | {data['best_model']} | {data['chrf_score']:.1f} |")
        
        lines.append("")
        
        # Curriculum Results
        lines.append("## Curriculum Progression Analysis")
        lines.append("")
        lines.append("| CEFR Level | Best Model | Accuracy |")
        lines.append("|------------|------------|----------|")
        
        for level, data in comparison.get("by_curriculum_level", {}).items():
            level_info = CURRICULUM_LEVELS.get(level, {})
            level_name = level_info.get("name", level)
            lines.append(f"| {level} ({level_name}) | {data['best_model']} | {data['accuracy']:.1f}% |")
        
        lines.append("")
        
        # Detailed Model Results
        lines.append("## Detailed Model Results")
        lines.append("")
        
        for ms in model_scores:
            lines.append(f"### {ms.model_name}")
            lines.append("")
            lines.append(f"- **Provider:** {ms.provider}")
            lines.append(f"- **Model ID:** `{ms.model_id}`")
            lines.append(f"- **Overall Score:** {ms.overall_score:.2f}")
            lines.append(f"- **Total Samples:** {ms.total_samples:,}")
            lines.append(f"- **Error Rate:** {(ms.total_errors / max(ms.total_samples, 1) * 100):.1f}%")
            lines.append(f"- **Avg Latency:** {ms.avg_latency_ms:.0f}ms")
            lines.append(f"- **Estimated Cost:** ${ms.estimated_cost:.2f}")
            lines.append("")
            
            # Language pair breakdown
            if ms.language_pair_scores:
                lines.append("#### Translation Scores")
                lines.append("")
                lines.append("| Direction | BLEU | chrF++ | Samples |")
                lines.append("|-----------|------|--------|---------|")
                
                for pair_key, ps in ms.language_pair_scores.items():
                    pair_display = pair_key.replace("_", " → ")
                    lines.append(f"| {pair_display} | {ps.bleu:.1f} | {ps.chrf:.1f} | {ps.sample_count} |")
                
                lines.append("")
            
            # Curriculum breakdown
            if ms.curriculum_scores:
                lines.append("#### Curriculum Scores")
                lines.append("")
                lines.append("| Level | Accuracy | Partial | Samples |")
                lines.append("|-------|----------|---------|---------|")
                
                for level, cs in ms.curriculum_scores.items():
                    lines.append(f"| {level} | {cs.accuracy:.1f}% | {cs.partial_score:.1f}% | {cs.sample_count} |")
                
                lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by LearnN'Ko Manding Language Benchmark Pipeline*")
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return filepath
    
    def generate_heatmap_data(
        self,
        model_scores: List[ModelBenchmarkScore],
    ) -> Dict[str, Any]:
        """
        Generate data for visualization heatmaps.
        
        Returns structured data suitable for plotting with matplotlib/seaborn.
        """
        heatmap_data = {
            "models": [ms.model_name for ms in model_scores],
            "language_pairs": [],
            "curriculum_levels": list(CURRICULUM_LEVELS.keys()),
            "translation_matrix": [],
            "curriculum_matrix": [],
        }
        
        # Collect all language pairs
        all_pairs = set()
        for ms in model_scores:
            all_pairs.update(ms.language_pair_scores.keys())
        heatmap_data["language_pairs"] = sorted(all_pairs)
        
        # Translation matrix (models x language pairs)
        for ms in model_scores:
            row = []
            for pair in heatmap_data["language_pairs"]:
                if pair in ms.language_pair_scores:
                    row.append(ms.language_pair_scores[pair].chrf)
                else:
                    row.append(0.0)
            heatmap_data["translation_matrix"].append(row)
        
        # Curriculum matrix (models x levels)
        for ms in model_scores:
            row = []
            for level in heatmap_data["curriculum_levels"]:
                if level in ms.curriculum_scores:
                    row.append(ms.curriculum_scores[level].accuracy)
                else:
                    row.append(0.0)
            heatmap_data["curriculum_matrix"].append(row)
        
        return heatmap_data
    
    def generate_all_reports(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Path]:
        """
        Generate all report types from benchmark results.
        
        Args:
            results: Dictionary mapping model_key -> results dict
            
        Returns:
            Dictionary of report type -> file path
        """
        from ..config import MODELS
        
        # Generate model scores
        model_scores = []
        for model_key, model_results in results.items():
            if model_key in MODELS:
                model_config = MODELS[model_key]
                score = self.generate_model_score(
                    model_name=model_config.name,
                    model_id=model_config.model_id,
                    provider=model_config.provider,
                    results=model_results,
                )
                model_scores.append(score)
        
        # Generate comparison
        comparison = self.generate_comparison_report(model_scores)
        
        # Generate all reports
        reports = {}
        reports["json"] = self.generate_json_report(comparison)
        reports["markdown"] = self.generate_markdown_report(comparison, model_scores)
        
        # Save heatmap data
        heatmap_data = self.generate_heatmap_data(model_scores)
        heatmap_path = self.output_dir / f"manding_heatmap_{self.timestamp}.json"
        with open(heatmap_path, "w") as f:
            json.dump(heatmap_data, f, indent=2)
        reports["heatmap"] = heatmap_path
        
        return reports


def generate_manding_benchmark_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Convenience function to generate all Manding benchmark reports.
    
    Args:
        results: Dictionary mapping model_key -> results dict
        output_dir: Optional output directory
        
    Returns:
        Dictionary of report type -> file path
    """
    generator = MandingReportGenerator(output_dir)
    return generator.generate_all_reports(results)

