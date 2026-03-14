"""
Per-layer hidden state activation profiling.

Extracts hidden states from all layers of a transformer model
and computes statistical metrics per layer:
- L2 norm (activation magnitude)
- Shannon entropy (distribution spread)
- Sparsity (fraction of near-zero neurons)
- Kurtosis (peakedness of activation distribution)

Usage on Vast.ai::

    from scanner.activation_profiler import ActivationProfiler

    profiler = ActivationProfiler(model, tokenizer)
    profile = profiler.profile("ߒ ߓߍ߬ ߕߊ߬ߡߌ߲")
    # profile = {"layer_0": {"l2_norm": ..., "entropy": ..., ...}, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None  # type: ignore


@dataclass
class LayerMetrics:
    """Statistical metrics for a single layer's hidden state."""
    layer_idx: int
    l2_norm: float = 0.0
    entropy: float = 0.0
    sparsity: float = 0.0
    kurtosis: float = 0.0
    mean_activation: float = 0.0
    std_activation: float = 0.0

    def to_dict(self) -> dict:
        return {
            "layer": self.layer_idx,
            "l2_norm": self.l2_norm,
            "entropy": self.entropy,
            "sparsity": self.sparsity,
            "kurtosis": self.kurtosis,
            "mean": self.mean_activation,
            "std": self.std_activation,
        }


class ActivationProfiler:
    """
    Profiles per-layer activation patterns for a transformer model.

    Requires a HuggingFace model loaded with output_hidden_states=True.
    """

    def __init__(self, model, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        # With device_map="auto", model is split across GPUs.
        # Use first parameter's device for input placement.
        if device == "auto":
            self.device = next(model.parameters()).device
        else:
            self.device = device
        self._sparsity_threshold = 1e-3

    def profile(self, text: str) -> List[LayerMetrics]:
        """
        Run a single text through the model and compute per-layer metrics.

        Parameters
        ----------
        text : str
            Input text (NKO or English).

        Returns
        -------
        List[LayerMetrics]
            One entry per layer (including embedding layer).
        """
        if torch is None:
            raise ImportError("torch is required for profiling")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # Tuple of (num_layers+1) tensors

        metrics = []
        for layer_idx, hs in enumerate(hidden_states):
            # hs shape: [batch, seq_len, hidden_dim]
            # Take mean over sequence length for aggregate metrics
            h = hs[0].float().cpu().numpy()  # [seq_len, hidden_dim]
            h_flat = h.mean(axis=0)  # [hidden_dim] — average over positions

            m = LayerMetrics(layer_idx=layer_idx)
            m.l2_norm = float(np.linalg.norm(h_flat))
            m.mean_activation = float(np.mean(h_flat))
            m.std_activation = float(np.std(h_flat))

            # Shannon entropy (on absolute values, normalized)
            abs_h = np.abs(h_flat)
            abs_sum = abs_h.sum()
            if abs_sum > 0:
                p = abs_h / abs_sum
                p = p[p > 0]  # Remove zeros for log
                m.entropy = float(-np.sum(p * np.log2(p)))

            # Sparsity (fraction near zero)
            m.sparsity = float(np.mean(np.abs(h_flat) < self._sparsity_threshold))

            # Kurtosis
            if m.std_activation > 0:
                centered = (h_flat - m.mean_activation) / m.std_activation
                m.kurtosis = float(np.mean(centered ** 4) - 3.0)  # Excess kurtosis

            metrics.append(m)

        return metrics

    def profile_batch(self, texts: List[str]) -> Dict[str, List[LayerMetrics]]:
        """
        Profile multiple texts and return per-text metrics.

        Parameters
        ----------
        texts : List[str]
            Input texts.

        Returns
        -------
        Dict[str, List[LayerMetrics]]
            Mapping of text -> per-layer metrics.
        """
        results = {}
        for text in texts:
            results[text] = self.profile(text)
        return results

    def average_profiles(self, all_metrics: List[List[LayerMetrics]]) -> List[LayerMetrics]:
        """
        Average metrics across multiple inputs per layer.

        Parameters
        ----------
        all_metrics : List[List[LayerMetrics]]
            List of per-input metric lists (each from profile()).

        Returns
        -------
        List[LayerMetrics]
            Averaged metrics per layer.
        """
        if not all_metrics:
            return []

        num_layers = len(all_metrics[0])
        averaged = []

        for layer_idx in range(num_layers):
            layer_data = [m[layer_idx] for m in all_metrics if layer_idx < len(m)]
            avg = LayerMetrics(layer_idx=layer_idx)
            n = len(layer_data)
            if n > 0:
                avg.l2_norm = sum(d.l2_norm for d in layer_data) / n
                avg.entropy = sum(d.entropy for d in layer_data) / n
                avg.sparsity = sum(d.sparsity for d in layer_data) / n
                avg.kurtosis = sum(d.kurtosis for d in layer_data) / n
                avg.mean_activation = sum(d.mean_activation for d in layer_data) / n
                avg.std_activation = sum(d.std_activation for d in layer_data) / n
            averaged.append(avg)

        return averaged
