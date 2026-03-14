"""
Layer duplication engine for circuit analysis.

Implements the (i,j) layer duplication technique from the RYS paper (Ng):
duplicate layers i through j-1 by slicing and extending nn.ModuleList.

Usage on Vast.ai::

    from scanner.layer_duplicator import LayerDuplicator

    dup = LayerDuplicator(model)
    dup.duplicate(45, 52)   # Duplicate layers 45-51
    # ... run evaluation ...
    dup.restore()           # Restore original layers
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


class LayerDuplicator:
    """
    Duplicates a contiguous block of transformer layers.

    The RYS technique: given layers [0, 1, ..., N-1], config (i, j)
    produces [0, ..., i-1, i, ..., j-1, i, ..., j-1, j, ..., N-1].
    """

    def __init__(self, model):
        self.model = model
        self._original_layers: Optional[nn.ModuleList] = None
        self._original_count: int = 0

        # Detect model architecture
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self._layers_attr = model.model.layers
            self._layers_parent = model.model
            self._layers_name = "layers"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self._layers_attr = model.transformer.h
            self._layers_parent = model.transformer
            self._layers_name = "h"
        else:
            raise ValueError(
                "Cannot find layer list. Supported: model.model.layers (Qwen2/Llama) "
                "or model.transformer.h (GPT-2 style)"
            )

        self._original_count = len(self._layers_attr)

    @property
    def num_layers(self) -> int:
        """Number of layers in the original model."""
        return self._original_count

    def valid_configs(self) -> list[Tuple[int, int]]:
        """
        Generate all valid (i, j) configurations.

        Valid: 0 <= i < j <= num_layers, and j - i >= 2 (at least 2 layers).
        """
        configs = []
        n = self._original_count
        for i in range(n):
            for j in range(i + 2, n + 1):
                configs.append((i, j))
        return configs

    def coarse_configs(self, step: int = 4) -> list[Tuple[int, int]]:
        """
        Generate coarse-grained configs (every `step`th layer).

        For quick scanning before committing to the full sweep.
        """
        configs = []
        n = self._original_count
        for i in range(0, n, step):
            for j in range(i + step, n + 1, step):
                configs.append((i, j))
        return configs

    def duplicate(self, i: int, j: int) -> int:
        """
        Duplicate layers [i, j) by appending copies after the original block.

        Parameters
        ----------
        i : int
            Start layer (inclusive).
        j : int
            End layer (exclusive).

        Returns
        -------
        int
            New total number of layers.
        """
        if torch is None:
            raise ImportError("torch is required")

        assert 0 <= i < j <= self._original_count, \
            f"Invalid config ({i}, {j}) for {self._original_count} layers"

        # Save original if not already saved
        if self._original_layers is None:
            self._original_layers = copy.copy(self._layers_attr)

        original = list(self._original_layers)
        # Build new layer list: [0..i-1] + [i..j-1] + [i..j-1] + [j..N-1]
        # NOTE: We reuse the same layer objects (shared weights). No deepcopy needed.
        # The forward pass simply runs through these layers twice, which is the
        # RYS technique — same circuit fires twice in the computational graph.
        # This avoids the massive overhead of deep-copying BnB quantized layers.
        new_layers = (
            original[:i]
            + original[i:j]
            + original[i:j]
            + original[j:]
        )

        new_module_list = nn.ModuleList(new_layers)
        setattr(self._layers_parent, self._layers_name, new_module_list)

        return len(new_layers)

    def restore(self) -> None:
        """Restore the original layer configuration."""
        if self._original_layers is not None:
            setattr(self._layers_parent, self._layers_name, self._original_layers)
            self._original_layers = None

    def get_config_info(self, i: int, j: int) -> dict:
        """Return metadata about a duplication config."""
        duplicated_count = j - i
        total_after = self._original_count + duplicated_count
        return {
            "i": i,
            "j": j,
            "duplicated_layers": duplicated_count,
            "original_total": self._original_count,
            "new_total": total_after,
            "duplicated_fraction": duplicated_count / self._original_count,
        }
