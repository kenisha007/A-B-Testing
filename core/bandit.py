"""
Thompson Sampling Engine — Beta-Bernoulli Multi-Armed Bandit
============================================================
Implements the Beta-Bernoulli conjugate update for binary reward signals.
Used by Netflix, Uber, and Meta for adaptive traffic allocation during experiments.

Key insight: instead of equal 50/50 splits, we sample from each arm's posterior
Beta(α, β) and route traffic to the arm with the highest sampled value.
This naturally explores uncertain arms and exploits known winners.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArmPosterior:
    """Beta distribution posterior for a single variant arm."""
    variant_id: str
    alpha: float = 1.0   # successes + 1  (uniform prior)
    beta: float  = 1.0   # failures  + 1

    @property
    def mean(self) -> float:
        """Posterior mean = E[θ] = α/(α+β)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance = αβ / ((α+β)²(α+β+1))"""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Draw one sample from Beta(α, β) — the Thompson sampling step."""
        if rng is None:
            rng = np.random.default_rng()
        return float(rng.beta(self.alpha, self.beta))

    def credible_interval(self, ci: float = 0.95) -> Tuple[float, float]:
        """HPD credible interval via Beta percent-point function."""
        from scipy.stats import beta as beta_dist
        lower = (1 - ci) / 2
        upper = 1 - lower
        return (
            float(beta_dist.ppf(lower, self.alpha, self.beta)),
            float(beta_dist.ppf(upper, self.alpha, self.beta))
        )

    def update(self, reward: int) -> None:
        """Bayesian update: reward=1 → increment α, reward=0 → increment β."""
        if reward == 1:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class ThompsonSamplingEngine:
    """
    Multi-arm Thompson Sampler with exact Bayesian posteriors.

    Supports:
      - N arms (not just A/B — could be A/B/C/D/…)
      - Beta-Bernoulli conjugate updates (O(1) per event)
      - Probability of being best via Monte Carlo (configurable samples)
      - Expected loss computation for ROPE-based stopping
      - Arm allocation weights for logging / observability

    Usage:
        engine = ThompsonSamplingEngine(variant_ids=["control", "treatment"])
        chosen = engine.select_arm()
        engine.update(chosen, reward=1)
        results = engine.compute_metrics(n_samples=50_000)
    """

    def __init__(
        self,
        variant_ids: List[str],
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.arms: Dict[str, ArmPosterior] = {
            vid: ArmPosterior(variant_id=vid) for vid in variant_ids
        }

    def select_arm(self) -> str:
        """
        Thompson Sampling arm selection.
        Draw θ̃ᵢ ~ Beta(αᵢ, βᵢ) for each arm, return argmax.
        Expected to converge to optimal arm in O(log T) regret.
        """
        samples = {
            vid: arm.sample(self._rng)
            for vid, arm in self.arms.items()
        }
        chosen = max(samples, key=samples.__getitem__)
        logger.debug("Thompson samples=%s → selected=%s", samples, chosen)
        return chosen

    def update(self, variant_id: str, reward: int) -> None:
        """Posterior update after observing a reward signal."""
        if variant_id not in self.arms:
            raise ValueError(f"Unknown variant: {variant_id}")
        self.arms[variant_id].update(reward)

    def bulk_update(self, variant_id: str, successes: int, failures: int) -> None:
        """Bulk-load historical data without per-event iteration."""
        arm = self.arms[variant_id]
        arm.alpha += successes
        arm.beta  += failures

    def compute_prob_best(self, n_samples: int = 50_000) -> Dict[str, float]:
        """
        P(arm i is the best arm) via Monte Carlo.
        For 2 arms this can be done analytically, but MC generalizes to N arms.

        Returns dict mapping variant_id → probability of being globally best.
        """
        samples = np.stack(
            [self._rng.beta(arm.alpha, arm.beta, size=n_samples)
             for arm in self.arms.values()],
            axis=1  # shape: (n_samples, n_arms)
        )
        winners = np.argmax(samples, axis=1)  # shape: (n_samples,)
        arm_ids = list(self.arms.keys())
        counts = np.bincount(winners, minlength=len(arm_ids))
        return {arm_ids[i]: float(counts[i]) / n_samples for i in range(len(arm_ids))}

    def compute_expected_loss(
        self,
        n_samples: int = 50_000,
    ) -> Dict[str, float]:
        """
        Expected loss if we choose arm i = E[max_j θ_j − θ_i].
        Used for ROPE-based early stopping: stop when max expected loss < ε.
        """
        samples = np.stack(
            [self._rng.beta(arm.alpha, arm.beta, size=n_samples)
             for arm in self.arms.values()],
            axis=1
        )
        best = samples.max(axis=1, keepdims=True)  # (n_samples, 1)
        losses = best - samples                      # (n_samples, n_arms)
        arm_ids = list(self.arms.keys())
        return {arm_ids[i]: float(losses[:, i].mean()) for i in range(len(arm_ids))}

    def allocation_weights(self) -> Dict[str, float]:
        """
        Current traffic allocation weights based on posterior means.
        Used to dynamically skew traffic toward the winning arm.
        """
        means = {vid: arm.mean for vid, arm in self.arms.items()}
        total = sum(means.values())
        return {vid: v / total for vid, v in means.items()}

    def state_dict(self) -> Dict:
        """Serializable snapshot of all posteriors (for persistence / audit)."""
        return {
            vid: {"alpha": arm.alpha, "beta": arm.beta}
            for vid, arm in self.arms.items()
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "ThompsonSamplingEngine":
        """Reconstruct engine from persisted state."""
        instance = cls(variant_ids=list(state.keys()))
        for vid, params in state.items():
            instance.arms[vid].alpha = params["alpha"]
            instance.arms[vid].beta  = params["beta"]
        return instance
