"""
RecursionRouter — learned recursion depth for RVLM.

Treats the number of iterations as a task-adaptive latent variable rather than
a fixed hyperparameter.  The router has two responsibilities:

1. **Pre-flight** (`recommended_max_iterations`): uses complexity features derived
   from segmentation mask statistics to predict how many iterations the task needs
   *before* the loop starts.

2. **Per-iteration** (`should_continue`): inspects the REPL state after each
   iteration to detect early completion (the `report` variable appeared) or
   unproductive stalling (two consecutive iterations produced no sub-LM calls and
   no meaningful stdout), and terminates early when appropriate.

Complexity features (medical imaging context):
- Label entropy:     Shannon entropy over {NCR, ED, ET} sub-region volumes.
                     High entropy → all three regions present and roughly equal
                     → harder multi-region characterisation → more iterations.
- Total volume:      Large tumours require more visual reasoning.
- Sub-region count:  More distinct regions → more analytical steps.
- Tiny region flag:  Any region < 0.5 cc is hard to characterise visually.

Complexity → recommended iterations:
    score < 0.25  →  3   (simple single-region, small tumour)
    0.25–0.45     →  4
    0.45–0.65     →  5
    ≥ 0.65        →  6   (complex multi-region, large tumour)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rvlm.core.types import RLMIteration

# Weights for the complexity score components
_W_ENTROPY = 0.35
_W_VOLUME = 0.30
_W_REGIONS = 0.25
_W_TINY = 0.10

# Maximum entropy for 3 equal-probability labels: log2(3) ≈ 1.585
_MAX_ENTROPY = math.log2(3)

# Volume at which the volume contribution saturates (cc)
_VOLUME_SATURATION = 50.0

# Minimum volume (cc) to count a sub-region as "present"
_PRESENT_THRESHOLD = 0.01

# Minimum volume (cc) below which a sub-region is considered "tiny"
_TINY_THRESHOLD = 0.5

# Stall: stdout shorter than this is considered empty
_STALL_STDOUT_THRESHOLD = 20


def _label_entropy(mask_stats: dict[str, Any]) -> float:
    """Shannon entropy (bits) of the NCR / ED / ET volume distribution."""
    volumes = [
        mask_stats.get("NCR_volume_cc", 0.0),
        mask_stats.get("ED_volume_cc", 0.0),
        mask_stats.get("ET_volume_cc", 0.0),
    ]
    total = sum(volumes)
    if total <= 0:
        return 0.0
    probs = [v / total for v in volumes if v > 0]
    return -sum(p * math.log2(p) for p in probs)


def _complexity_score(mask_stats: dict[str, Any]) -> float:
    """Return a composite complexity score in [0.0, 1.0]."""
    if not mask_stats:
        return 0.5  # neutral prior when no stats available

    entropy = _label_entropy(mask_stats)
    total_vol = mask_stats.get("total_tumor_volume_cc", 0.0)

    volumes = [
        mask_stats.get("NCR_volume_cc", 0.0),
        mask_stats.get("ED_volume_cc", 0.0),
        mask_stats.get("ET_volume_cc", 0.0),
    ]
    present_regions = sum(1 for v in volumes if v > _PRESENT_THRESHOLD)
    any_tiny = any(0 < v < _TINY_THRESHOLD for v in volumes)

    score = (
        (entropy / _MAX_ENTROPY) * _W_ENTROPY
        + min(total_vol / _VOLUME_SATURATION, 1.0) * _W_VOLUME
        + (present_regions / 3) * _W_REGIONS
        + (1.0 if any_tiny else 0.0) * _W_TINY
    )
    return min(max(score, 0.0), 1.0)


def _score_to_iterations(score: float) -> int:
    """Map a complexity score to a recommended iteration budget."""
    if score < 0.25:
        return 3
    elif score < 0.45:
        return 4
    elif score < 0.65:
        return 5
    else:
        return 6


class RecursionRouter:
    """
    Adaptive recursion depth controller for RLM / RVLM.

    Instantiate via `RecursionRouter.from_mask_stats(mask_stats)` for
    medical-imaging tasks, or directly with a pre-computed complexity score.

    Args:
        complexity_score: Float in [0.0, 1.0].  Higher = more iterations.
        verbose:          Print one-line decision log after each iteration.
    """

    def __init__(self, complexity_score: float = 0.5, verbose: bool = False):
        self.complexity_score = min(max(complexity_score, 0.0), 1.0)
        self._recommended: int = _score_to_iterations(self.complexity_score)
        self._stall_count: int = 0
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_mask_stats(
        cls,
        mask_stats: dict[str, Any] | None,
        verbose: bool = False,
    ) -> "RecursionRouter":
        """
        Build a router from segmentation mask statistics.

        Args:
            mask_stats: Output of `compute_mask_stats()` from brats_example,
                        or any dict with keys like `NCR_volume_cc`,
                        `ED_volume_cc`, `ET_volume_cc`, `total_tumor_volume_cc`.
            verbose:    Print router decisions during the completion loop.

        Returns:
            A configured `RecursionRouter` instance.
        """
        score = _complexity_score(mask_stats or {})
        return cls(complexity_score=score, verbose=verbose)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommended_max_iterations(self) -> int:
        """Return the task-complexity-derived iteration budget."""
        return self._recommended

    def should_continue(
        self,
        iteration_num: int,
        rlm_iteration: "RLMIteration",
        repl_locals: dict[str, Any],
    ) -> bool:
        """
        Decide whether the loop should proceed to the next iteration.

        Called *after* `format_iteration` has been applied and the model's
        response for iteration `iteration_num` (0-indexed) is available.

        Note: completion detection (FINAL_VAR / FINAL) is handled by the RVLM loop
        itself — this method only handles *stall* detection.  Terminating here on
        "report in locals" would bypass FINAL_VAR and cause `_default_answer` to
        replace the actual report with a generic unstructured response.

        Returns:
            True  → proceed to the next iteration.
            False → terminate now (caller will use locals or call `_default_answer`).
        """
        # Stall detection: was this iteration unproductive?
        productive = self._is_productive(rlm_iteration)
        if productive:
            self._stall_count = 0
        else:
            self._stall_count += 1

        # 3. Terminate if stalling beyond the recommended budget.
        if self._stall_count >= 2 and iteration_num >= self._recommended - 1:
            decision = False
            reason = f"stall_count={self._stall_count} at iter {iteration_num + 1}/{self._recommended}"
            self._log(iteration_num, decision, reason)
            return decision

        decision = True
        reason = f"stalls={self._stall_count}"
        self._log(iteration_num, decision, reason)
        return decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_productive(self, rlm_iteration: "RLMIteration") -> bool:
        """Return True if this iteration made meaningful progress."""
        for cb in rlm_iteration.code_blocks:
            result = cb.result
            # Sub-LM calls happened → definitely productive
            if result.rvlm_calls:
                return True
            # Meaningful stdout → something was computed / printed
            if len(result.stdout.strip()) >= _STALL_STDOUT_THRESHOLD:
                return True
            # New locals beyond trivial helper variables
            non_trivial = {
                k: v for k, v in result.locals.items()
                if not k.startswith("_") and not callable(v)
            }
            if non_trivial:
                return True
        return False

    def _log(self, iteration_num: int, decision: bool, reason: str) -> None:
        if not self.verbose:
            return
        action = "CONTINUE" if decision else "TERMINATE"
        print(
            f"[Router] iter={iteration_num + 1}"
            f"  complexity={self.complexity_score:.2f}"
            f"  recommended={self._recommended}"
            f"  stalls={self._stall_count}"
            f"  → {action}  ({reason})"
        )
