"""
Report Generator for N'Ko Benchmark.

Generates JSON and Markdown reports from benchmark results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class ReportGenerator:
    """
    Generate benchmark reports in various formats.
    
    Supports:
    - JSON (machine-readable, complete data)
    - Markdown (human-readable summary)
    - Console (quick summary)
    """
    
    def generate_json(
        self,
        results: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Generate JSON report.
        
        Args:
            results: Benchmark results dictionary
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    def generate_markdown(
        self,
        results: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Generate Markdown report.
        
        Args:
            results: Benchmark results dictionary
            output_path: Path to save Markdown file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        
        # Header
        lines.append("# N'Ko AI Model Benchmark Report")
        lines.append("")
        lines.append(f"**Generated:** {results.get('timestamp', datetime.now().isoformat())}")
        lines.append("")
        
        # Configuration
        config = results.get("config", {})
        lines.append("## Configuration")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for key, value in config.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
        lines.append("")
        
        # Rankings
        rankings = results.get("rankings", [])
        if rankings:
            lines.append("## Model Rankings")
            lines.append("")
            lines.append("| Rank | Model | Provider | Overall Score |")
            lines.append("|------|-------|----------|---------------|")
            for rank_info in rankings:
                lines.append(
                    f"| {rank_info['rank']} | {rank_info['model_id']} | "
                    f"{rank_info['provider']} | {rank_info['overall_score']:.1f} |"
                )
            lines.append("")
        
        # Recommendation
        recommendation = results.get("recommendation", {})
        if recommendation:
            lines.append("## Recommendation")
            lines.append("")
            best = recommendation.get("best_overall", {})
            if best:
                lines.append(f"**Recommended Model:** `{best.get('model_id', 'N/A')}`")
                lines.append("")
                lines.append(f"- **Provider:** {best.get('provider', 'N/A')}")
                lines.append(f"- **Overall Score:** {best.get('overall_score', 0):.1f}")
                lines.append(f"- **Reason:** {best.get('reason', 'N/A')}")
            lines.append("")
            
            value = recommendation.get("best_value", {})
            if value and value.get("model_id") != best.get("model_id"):
                lines.append("### Best Value Option")
                lines.append("")
                lines.append(f"**Model:** `{value.get('model_id', 'N/A')}`")
                lines.append(f"- **Score:** {value.get('overall_score', 0):.1f}")
                lines.append(f"- **Estimated Cost:** ${value.get('cost', 0):.2f}")
                lines.append(f"- **Reason:** {value.get('reason', 'N/A')}")
            lines.append("")
        
        # Detailed Results
        models = results.get("models", {})
        if models:
            lines.append("## Detailed Results")
            lines.append("")
            
            for model_id, model_data in models.items():
                lines.append(f"### {model_id}")
                lines.append("")
                lines.append(f"- **Provider:** {model_data.get('provider', 'N/A')}")
                lines.append(f"- **Overall Score:** {model_data.get('overall_score', 0):.1f}")
                lines.append(f"- **Avg Latency:** {model_data.get('avg_latency_ms', 0):.0f}ms")
                lines.append(f"- **Est. Cost:** ${model_data.get('total_cost_estimate', 0):.2f}")
                lines.append("")
                
                task_scores = model_data.get("task_scores", {})
                if task_scores:
                    lines.append("#### Task Scores")
                    lines.append("")
                    lines.append("| Task | Weight | Raw Score | Weighted Score |")
                    lines.append("|------|--------|-----------|----------------|")
                    for task_name, task_data in task_scores.items():
                        lines.append(
                            f"| {task_name.replace('_', ' ').title()} | "
                            f"{task_data.get('weight', 0):.0%} | "
                            f"{task_data.get('raw_score', 0):.1f} | "
                            f"{task_data.get('weighted_score', 0):.1f} |"
                        )
                    lines.append("")
                    
                    # Key metrics for each task
                    lines.append("#### Key Metrics")
                    lines.append("")
                    for task_name, task_data in task_scores.items():
                        metrics = task_data.get("metrics", {})
                        if metrics:
                            lines.append(f"**{task_name.replace('_', ' ').title()}:**")
                            for metric_name, metric_value in metrics.items():
                                if isinstance(metric_value, float):
                                    lines.append(f"- {metric_name}: {metric_value:.2f}")
                                else:
                                    lines.append(f"- {metric_name}: {metric_value}")
                            lines.append("")
                
                lines.append("---")
                lines.append("")
        
        # Comparison Chart (text-based)
        if len(models) > 1:
            lines.append("## Score Comparison")
            lines.append("")
            lines.append("```")
            
            # Find max score for scaling
            max_score = max(
                m.get("overall_score", 0) for m in models.values()
            ) or 100
            
            for model_id, model_data in sorted(
                models.items(),
                key=lambda x: x[1].get("overall_score", 0),
                reverse=True
            ):
                score = model_data.get("overall_score", 0)
                bar_len = int((score / max_score) * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                lines.append(f"{model_id:25} {bar} {score:.1f}")
            
            lines.append("```")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by N'Ko AI Model Benchmark Pipeline*")
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        rankings = results.get("rankings", [])
        if rankings:
            print("\nModel Rankings:")
            for rank_info in rankings:
                print(
                    f"  {rank_info['rank']}. {rank_info['model_id']} "
                    f"({rank_info['provider']}): {rank_info['overall_score']:.1f}"
                )
        
        recommendation = results.get("recommendation", {})
        if recommendation:
            print(f"\nRecommendation: {recommendation.get('recommendation', 'N/A')}")
            best = recommendation.get("best_overall", {})
            if best:
                print(f"  Reason: {best.get('reason', 'N/A')}")
        
        print("=" * 60)


def generate_comparison_table(
    results: Dict[str, Any],
) -> str:
    """Generate a comparison table in Markdown format."""
    models = results.get("models", {})
    if not models:
        return "No models to compare."
    
    lines = []
    lines.append("| Model | Provider | Overall | Translation | Vocabulary | Cultural | Latency (ms) | Cost ($) |")
    lines.append("|-------|----------|---------|-------------|------------|----------|--------------|----------|")
    
    for model_id, model_data in sorted(
        models.items(),
        key=lambda x: x[1].get("overall_score", 0),
        reverse=True
    ):
        task_scores = model_data.get("task_scores", {})
        trans_score = task_scores.get("translation", {}).get("raw_score", 0)
        vocab_score = task_scores.get("vocabulary", {}).get("raw_score", 0)
        cultural_score = task_scores.get("cultural", {}).get("raw_score", 0)
        
        lines.append(
            f"| {model_id} | {model_data.get('provider', 'N/A')} | "
            f"{model_data.get('overall_score', 0):.1f} | "
            f"{trans_score:.1f} | {vocab_score:.1f} | {cultural_score:.1f} | "
            f"{model_data.get('avg_latency_ms', 0):.0f} | "
            f"{model_data.get('total_cost_estimate', 0):.2f} |"
        )
    
    return '\n'.join(lines)

