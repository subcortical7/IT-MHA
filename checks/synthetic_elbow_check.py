"""
synthetic_elbow_check.py
========================
Validates the elbow detector used for the nested-dropout VAE variance
spectra (detect_elbow in vae_core.py) on synthetic curves whose elbow is
known in advance, and shows why the original formula failed.

Two checks:

  1. Failure mode of the original formula. The first version of the
     detector took argmax(norm - line), i.e. the point furthest ABOVE the
     chord joining the curve's endpoints. That is the right sign for a
     concave curve, but a variance spectrum is convex-decreasing (steep
     drop, long flat tail): every interior point lies BELOW the chord, so
     norm - line <= 0 everywhere and the argmax lands on an endpoint
     (k = 1 or k = 64). With noise added, a point next to an endpoint can
     edge just above the chord, so the result is an endpoint or its
     immediate neighbour -- either way, set by noise rather than by the
     curve's shape. This script confirms both cases.

  2. Recovery by the corrected detector. detect_elbow takes
     argmax(line - norm), the point furthest BELOW the chord. On
     piecewise-linear convex spectra with a known breakpoint (plus small
     noise), it should return that breakpoint.

Run (from the repository folder):
    python checks/synthetic_elbow_check.py
"""

import os, sys
# Repo root, so scripts work from any working directory and find vae_core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')

import numpy as np
from vae_core import detect_elbow  # the corrected detector used in the paper

N_DIMS = 64
N_NOISE_REPEATS = 20
NOISE_SD = 0.01          # multiplicative noise on each variance value
TRUE_ELBOWS = [4, 6, 8, 10, 12, 16, 20, 24, 32]


def detect_elbow_original(variances):
    """The original (incorrect) sign convention, kept only for comparison."""
    sorted_var = np.sort(variances)[::-1]
    n = len(sorted_var)
    norm = (sorted_var - sorted_var[-1]) / (sorted_var[0] - sorted_var[-1] + 1e-10)
    line = np.linspace(1.0, 0.0, n)
    diff = norm - line
    return int(np.argmax(diff)) + 1


def piecewise_spectrum(k_true, rng, knee_level=0.1):
    """Convex-decreasing spectrum: steep linear drop from 1.0 to knee_level
    over the first k_true dimensions, then a shallow linear decline to 0
    over the rest. The breakpoint (dimension k_true) is the known elbow."""
    head = np.linspace(1.0, knee_level, k_true)
    tail = np.linspace(knee_level, 0.0, N_DIMS - k_true + 1)[1:]
    v = np.concatenate([head, tail])
    if rng is not None:
        v = v * (1 + NOISE_SD * rng.standard_normal(N_DIMS))
    return np.clip(v, 0.0, None)


def exponential_spectrum(tau, rng):
    """Smooth convex-decreasing spectrum, used only for the failure-mode check."""
    v = np.exp(-np.arange(N_DIMS) / tau)
    return v * (1 + NOISE_SD * rng.standard_normal(N_DIMS))


def main():
    rng = np.random.default_rng(0)

    # ── Check 1: the original formula collapses to an endpoint ──────────────
    print("Check 1: original formula on convex-decreasing spectra")
    near_edge = lambda k: k <= 2 or k >= N_DIMS - 1

    # (a) noiseless convex curves: always exactly an endpoint
    clean = [detect_elbow_original(np.exp(-np.arange(N_DIMS) / tau))
             for tau in [1, 2, 4, 8, 16, 32]]
    clean += [detect_elbow_original(piecewise_spectrum(k, None))
              for k in TRUE_ELBOWS]
    exact_edge = sum(k in (1, N_DIMS) for k in clean)
    print(f"  noiseless: returned k=1 or k={N_DIMS} on {exact_edge}/{len(clean)} curves")

    # (b) noisy convex curves: an endpoint or its immediate neighbour
    noisy = []
    for tau in [2, 4, 8, 16]:
        noisy += [detect_elbow_original(exponential_spectrum(tau, rng))
                  for _ in range(N_NOISE_REPEATS)]
    for k_true in TRUE_ELBOWS:
        noisy += [detect_elbow_original(piecewise_spectrum(k_true, rng))
                  for _ in range(N_NOISE_REPEATS)]
    edge = sum(near_edge(k) for k in noisy)
    print(f"  noisy:     returned k in {{1, 2, {N_DIMS-1}, {N_DIMS}}} on {edge}/{len(noisy)} curves")
    check1 = exact_edge == len(clean) and edge == len(noisy)

    # ── Check 2: the corrected detector recovers the known elbow ────────────
    print("\nCheck 2: corrected detector on spectra with a known elbow")
    print(f"  {'true k':>6}  {'mean found':>10}  {'exact':>7}  {'within 1':>8}")
    all_within_1 = True
    for k_true in TRUE_ELBOWS:
        found = np.array([detect_elbow(piecewise_spectrum(k_true, rng))[0]
                          for _ in range(N_NOISE_REPEATS)])
        exact = np.mean(found == k_true)
        within1 = np.mean(np.abs(found - k_true) <= 1)
        all_within_1 &= within1 == 1.0
        print(f"  {k_true:>6}  {found.mean():>10.2f}  {exact:>7.0%}  {within1:>8.0%}")
    check2 = all_within_1

    print("\nRESULT:",
          "PASS" if (check1 and check2) else "FAIL",
          "-- original formula lands on (or next to) an endpoint; corrected"
          " detector recovers every known elbow to within one dimension."
          if (check1 and check2) else "-- see the numbers above.")


if __name__ == "__main__":
    main()
