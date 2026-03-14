"""
Heatmap generator for the (i,j) layer duplication sweep.

Runs all valid (i,j) configurations, evaluates probes at each,
and stores results as numpy arrays for visualization.

Usage on Vast.ai::

    from scanner.heatmap_generator import HeatmapGenerator

    gen = HeatmapGenerator(model, tokenizer, probes)
    results = gen.run_sweep(configs, script="english")
    gen.save_results(results, "results/heatmaps/english_sweep.npz")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .layer_duplicator import LayerDuplicator


@dataclass
class SweepResult:
    """Result of evaluating one (i,j) configuration."""
    i: int
    j: int
    math_score: float
    semantic_score: float
    combined_score: float
    duration_s: float
    num_layers_total: int


class HeatmapGenerator:
    """
    Orchestrates the full (i,j) sweep for heatmap generation.

    For each configuration:
    1. Duplicate layers via LayerDuplicator
    2. Run math probes -> score
    3. Run semantic probes -> score
    4. Restore original layers
    5. Record scores

    Results are stored in an 80x80 numpy array (or whatever the
    model's layer count is).
    """

    def __init__(self, model, tokenizer, math_probes: list, semantic_probes: list):
        self.model = model
        self.tokenizer = tokenizer
        self.math_probes = math_probes
        self.semantic_probes = semantic_probes
        self.duplicator = LayerDuplicator(model)

    def _run_probe(self, prompt: str, max_tokens: int = 50) -> str:
        """Run a single probe and return the model's text response."""
        if torch is None:
            raise ImportError("torch is required")

        # With device_map="auto", use first parameter's device for input placement
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def evaluate_config(
        self, i: int, j: int, script: str = "english",
        max_math_probes: int = 0, max_sem_probes: int = 0,
    ) -> SweepResult:
        """
        Evaluate a single (i,j) configuration.

        Parameters
        ----------
        i, j : int
            Layer duplication range.
        script : str
            Which script version to use ("english" or "nko").
        max_math_probes : int
            Max math probes to run (0 = all).
        max_sem_probes : int
            Max semantic probes to run (0 = all).

        Returns
        -------
        SweepResult
        """
        from probes.scoring import score_math, score_semantic

        start = time.time()

        # Duplicate layers
        total = self.duplicator.duplicate(i, j)

        # Subsample probes if requested
        math_subset = self.math_probes[:max_math_probes] if max_math_probes > 0 else self.math_probes
        sem_subset = self.semantic_probes[:max_sem_probes] if max_sem_probes > 0 else self.semantic_probes

        # Run math probes
        math_scores = []
        for probe in math_subset:
            prompt = probe[script] if script in probe else probe["english"]
            response = self._run_probe(prompt)
            score = score_math(response, probe["answer"])
            math_scores.append(score)

        # Run semantic probes
        sem_scores = []
        for probe in sem_subset:
            prompt = probe[script] if script in probe else probe["english"]
            response = self._run_probe(prompt)
            score = score_semantic(response, probe["answer_keywords"])
            sem_scores.append(score)

        # Restore
        self.duplicator.restore()

        duration = time.time() - start
        math_avg = np.mean(math_scores) if math_scores else 0.0
        sem_avg = np.mean(sem_scores) if sem_scores else 0.0

        return SweepResult(
            i=i, j=j,
            math_score=float(math_avg),
            semantic_score=float(sem_avg),
            combined_score=float((math_avg + sem_avg) / 2),
            duration_s=duration,
            num_layers_total=total,
        )

    def run_sweep(
        self,
        configs: List[Tuple[int, int]],
        script: str = "english",
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 100,
        max_math_probes: int = 0,
        max_sem_probes: int = 0,
    ) -> List[SweepResult]:
        """
        Run the full (i,j) sweep.

        Parameters
        ----------
        configs : List[Tuple[int, int]]
            List of (i, j) configurations to evaluate.
        script : str
            Which script version to use.
        checkpoint_dir : str, optional
            Directory to save checkpoints.
        checkpoint_every : int
            Save checkpoint every N configs.
        max_math_probes : int
            Max math probes per config (0 = all).
        max_sem_probes : int
            Max semantic probes per config (0 = all).

        Returns
        -------
        List[SweepResult]
        """
        results = []
        total = len(configs)

        import sys
        for idx, (i, j) in enumerate(configs):
            print(f"[{idx+1}/{total}] Evaluating ({i}, {j}) for {script}...", end=" ", flush=True)
            try:
                result = self.evaluate_config(
                    i, j, script,
                    max_math_probes=max_math_probes,
                    max_sem_probes=max_sem_probes,
                )
                results.append(result)
                print(f"math={result.math_score:.3f} sem={result.semantic_score:.3f} "
                      f"({result.duration_s:.1f}s)", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                results.append(SweepResult(
                    i=i, j=j, math_score=0.0, semantic_score=0.0,
                    combined_score=0.0, duration_s=0.0, num_layers_total=0,
                ))
            sys.stdout.flush()

            # Checkpoint
            if checkpoint_dir and (idx + 1) % checkpoint_every == 0:
                self.save_results(results, f"{checkpoint_dir}/checkpoint_{script}_{idx+1}.npz")

        return results

    def results_to_heatmap(self, results: List[SweepResult], metric: str = "combined_score") -> np.ndarray:
        """
        Convert sweep results to an NxN heatmap array.

        Parameters
        ----------
        results : List[SweepResult]
        metric : str
            Which metric to use ("math_score", "semantic_score", "combined_score").

        Returns
        -------
        np.ndarray
            NxN array where N is the number of layers.
        """
        n = self.duplicator.num_layers
        heatmap = np.full((n, n), np.nan)

        for r in results:
            if 0 <= r.i < n and 0 <= r.j <= n:
                heatmap[r.i, r.j - 1] = getattr(r, metric)

        return heatmap

    @staticmethod
    def save_results(results: List[SweepResult], path: str) -> None:
        """Save sweep results to compressed numpy format."""
        data = {
            "i": np.array([r.i for r in results]),
            "j": np.array([r.j for r in results]),
            "math_score": np.array([r.math_score for r in results]),
            "semantic_score": np.array([r.semantic_score for r in results]),
            "combined_score": np.array([r.combined_score for r in results]),
            "duration_s": np.array([r.duration_s for r in results]),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        print(f"Saved {len(results)} results to {path}")

    @staticmethod
    def load_results(path: str) -> List[SweepResult]:
        """Load sweep results from numpy format."""
        data = np.load(path)
        results = []
        for idx in range(len(data["i"])):
            results.append(SweepResult(
                i=int(data["i"][idx]),
                j=int(data["j"][idx]),
                math_score=float(data["math_score"][idx]),
                semantic_score=float(data["semantic_score"][idx]),
                combined_score=float(data["combined_score"][idx]),
                duration_s=float(data["duration_s"][idx]),
                num_layers_total=0,
            ))
        return results
