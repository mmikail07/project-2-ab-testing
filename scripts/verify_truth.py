"""Ground-truth recovery gate.

Re-runs the synthetic experiment under 5 seeds, runs the primary two-proportion
z-test, and asserts that across runs the recovered point estimate falls inside
the 95% CI of the true lift in at least 95% of runs. Exits non-zero on failure.

Run after any change to src/simulate.py or src/stats.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulate import generate_experiment  # noqa: E402
from src.stats import two_proportion_ztest  # noqa: E402

SEEDS = [1, 7, 13, 42, 101]
N_USERS = 200_000
LIFT_PP = 2.5


def main() -> int:
    contained = 0
    rows = []
    for seed in SEEDS:
        df, truth = generate_experiment(n_users=N_USERS, lift_pp=LIFT_PP, seed=seed)
        c = df[df["arm"] == "control"]
        t = df[df["arm"] == "treatment"]
        result = two_proportion_ztest(
            successes_c=int(c["completion"].sum()),
            n_c=len(c),
            successes_t=int(t["completion"].sum()),
            n_t=len(t),
        )
        true_lift = truth.true_lift_pp / 100.0
        in_ci = result.ci_low <= true_lift <= result.ci_high
        contained += int(in_ci)
        rows.append((seed, result.point_estimate, result.ci_low, result.ci_high, in_ci))

    print(f"{'seed':>6} {'point':>10} {'ci_low':>10} {'ci_high':>10}  contains_true")
    for seed, pe, lo, hi, ok in rows:
        flag = "yes" if ok else "NO"
        print(f"{seed:>6} {pe:>10.5f} {lo:>10.5f} {hi:>10.5f}  {flag}")

    coverage = contained / len(SEEDS)
    print(f"\ntrue lift = {LIFT_PP}pp ({LIFT_PP/100:.5f})")
    print(f"coverage = {contained}/{len(SEEDS)} ({coverage:.0%})")

    if coverage < 0.95:
        print("FAIL: coverage below 95% — analysis is not reliably recovering truth", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
