"""Sequential testing: peeking simulation, O'Brien-Fleming bounds, mSPRT.

The pedagogical centerpiece of the repo. Naive interim looks (peek at the
p-value before the planned sample size is reached) inflate the false-positive
rate from the nominal alpha to several multiples of it. Two corrections that
restore calibration:

1. **O'Brien-Fleming spending function** (group-sequential). Pre-commit to K
   equally-spaced interim looks; use stricter critical z-values at early looks
   that taper to the unadjusted z at the final look. The literature constant
   c=2.024 for K=4 at alpha=0.05 two-sided gives critical values 4.05, 2.86,
   2.34, 2.02 at looks 1 through 4. Source: O'Brien and Fleming (1979).

2. **mSPRT (mixture Sequential Probability Ratio Test)**. Always-valid test
   based on a Bayesian likelihood ratio under a Gaussian mixing prior on the
   effect size. Reject when the likelihood ratio exceeds 1/alpha. Source:
   Johari, Pekelis, Walsh (2017), 'Peeking at A/B Tests'.

The peeking simulator is vectorized: for K peeks at sample sizes n_1, ..., n_K,
each peek's increment is drawn with a single np.random.binomial call across all
n_sims simulations. Memory is O(n_sims * K) rather than O(n_sims * n_K)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBF_CONSTANT_K4_ALPHA05 = 2.024


@dataclass
class PeekingResult:
    method: str
    n_sims: int
    n_per_arm: int
    baseline: float
    alpha: float
    peek_fractions: list[float]
    empirical_fpr: float
    rejection_share_at_peek: np.ndarray
    critical_zs: np.ndarray | None = None


def obrien_fleming_bounds(n_peeks: int = 4, alpha: float = 0.05, constant: float | None = None) -> np.ndarray:
    """Critical |z|-values at each of n_peeks equally-spaced interim looks.
    z_k = constant * sqrt(K / k). For non-K=4/alpha=0.05 settings, supply the
    constant from a published table or solve for it via Monte Carlo."""
    if constant is None:
        if n_peeks == 4 and abs(alpha - 0.05) < 1e-9:
            constant = OBF_CONSTANT_K4_ALPHA05
        else:
            raise ValueError(
                f"no literature constant available for n_peeks={n_peeks} alpha={alpha}; "
                "supply 'constant' explicitly"
            )
    looks = np.arange(1, n_peeks + 1)
    return constant * np.sqrt(n_peeks / looks)


def simulate_peeking(
    n_per_arm: int,
    baseline: float,
    method: str = "naive",
    n_sims: int = 10_000,
    peek_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    alpha: float = 0.05,
    mspr_tau: float = 0.025,
    seed: int = 42,
) -> PeekingResult:
    """Run n_sims A/A experiments with interim looks at the specified fractions.

    method:
      'naive' — reject if |z| > z_{alpha/2} at ANY peek.
      'obf'   — reject if |z| > OBF critical at peek k; once rejected, stop.
      'mspr'  — reject if mSPRT likelihood ratio > 1/alpha at ANY peek.
    """
    if method not in {"naive", "obf", "mspr"}:
        raise ValueError(f"unknown method {method!r}")

    rng = np.random.default_rng(seed)
    peek_n = [int(round(n_per_arm * f)) for f in peek_fractions]
    increments = [peek_n[0]] + [peek_n[i] - peek_n[i - 1] for i in range(1, len(peek_n))]
    K = len(peek_n)

    cum_c = np.zeros(n_sims, dtype=np.int64)
    cum_t = np.zeros(n_sims, dtype=np.int64)
    rejected_so_far = np.zeros(n_sims, dtype=bool)
    rejection_share_at_peek = np.zeros(K)

    critical_zs: np.ndarray | None = None
    if method == "obf":
        critical_zs = obrien_fleming_bounds(K, alpha)
    elif method == "naive":
        from scipy.stats import norm
        critical_zs = np.full(K, float(norm.ppf(1 - alpha / 2)))

    for k, (n_k, inc) in enumerate(zip(peek_n, increments)):
        cum_c += rng.binomial(inc, baseline, size=n_sims).astype(np.int64)
        cum_t += rng.binomial(inc, baseline, size=n_sims).astype(np.int64)

        p_c = cum_c / n_k
        p_t = cum_t / n_k
        p_pool = (cum_c + cum_t) / (2 * n_k)
        se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_k)
        se = np.where(se == 0, 1e-12, se)
        z = (p_t - p_c) / se

        if method in {"naive", "obf"}:
            reject_this_peek = np.abs(z) > critical_zs[k]
        else:  # mspr
            theta_hat = p_t - p_c
            v_n = p_pool * (1 - p_pool) * 2 / n_k
            v_n = np.where(v_n == 0, 1e-12, v_n)
            tau2 = mspr_tau ** 2
            lr = np.sqrt(v_n / (v_n + tau2)) * np.exp(
                (tau2 * theta_hat ** 2) / (2 * v_n * (v_n + tau2))
            )
            reject_this_peek = lr > (1.0 / alpha)

        newly_rejected = reject_this_peek & ~rejected_so_far
        rejected_so_far = rejected_so_far | newly_rejected
        rejection_share_at_peek[k] = rejected_so_far.mean()

    return PeekingResult(
        method=method,
        n_sims=int(n_sims),
        n_per_arm=int(n_per_arm),
        baseline=float(baseline),
        alpha=float(alpha),
        peek_fractions=list(peek_fractions),
        empirical_fpr=float(rejected_so_far.mean()),
        rejection_share_at_peek=rejection_share_at_peek,
        critical_zs=critical_zs,
    )
