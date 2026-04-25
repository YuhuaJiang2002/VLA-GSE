# Copyright 2025 VLA-GSE contributors. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
Adaptive temporal action ensembler.

Given a stream of overlapping predicted action chunks (shape ``[chunk_size, action_dim]``),
``AdaptiveEnsembler`` returns a single ``[action_dim]`` action per time step by
exponentially weighting recent predictions.

Weights are computed from the cosine similarity between the most recent and the
current first-step predictions (``cur_weight``), modulated by a sensitivity
parameter ``adaptive_ensemble_alpha``. This mirrors the ensembling logic used in
open VLA evaluation pipelines.
"""

from collections import deque
from typing import Optional

import numpy as np


class AdaptiveEnsembler:
    """Temporal ensemble of overlapping VLA action chunks."""

    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.0) -> None:
        self.pred_action_horizon = int(pred_action_horizon)
        self.adaptive_ensemble_alpha = float(adaptive_ensemble_alpha)
        self.action_history: deque = deque(maxlen=self.pred_action_horizon)

    def reset(self) -> None:
        self.action_history.clear()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        """Append ``cur_action`` (shape ``[chunk_size, action_dim]``) and return the ensembled action."""
        cur_action = np.asarray(cur_action)
        if cur_action.ndim == 1:
            cur_action = cur_action[None, :]
        self.action_history.append(cur_action)

        num_actions = len(self.action_history)
        curr_act_preds = np.stack(
            [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
        )

        if num_actions == 1:
            return curr_act_preds[0]

        ref = curr_act_preds[0]
        norms = np.linalg.norm(curr_act_preds, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-7
        cos_sim = np.sum(curr_act_preds * ref, axis=-1) / norms

        weights = np.exp(self.adaptive_ensemble_alpha * cos_sim)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * curr_act_preds, axis=0)
