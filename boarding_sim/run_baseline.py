"""
run_baseline.py
===============
Milestone 4 entry point: Experiment 1 – Baseline Comparison.

Runs all four boarding strategies under the baseline scenario (no Filipino
behaviour features) for 200 replications using Common Random Numbers, then:
  - Prints a summary statistics table to stdout.
  - Saves raw results to ``results/exp1_baseline.csv``.
  - Saves a boxplot to     ``results/fig1_boxplot_baseline.png``.
  - Performs the PRD §13 qualitative validation check.

Usage
-----
    python run_baseline.py

Runtime
-------
~30-60 seconds on a typical laptop (800 simulation runs).
"""

from __future__ import annotations

import pathlib
import sys
import time

# Ensure the project root is on sys.path when run from any working directory.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd

from boarding import load_config
from boarding.experiments import run_experiment
from boarding.strategies import STRATEGIES
from boarding.charts import plot_boxplot_boarding_time

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """Print a statistics summary table to stdout."""
    summary = (
        df.groupby("strategy")["total_boarding_time_s"]
        .agg(["mean", "std", "median", "min", "max"])
        .round(1)
    )
    # Re-order strategies for readability
    order = ["random", "back_to_front", "outside_in", "steffen"]
    summary = summary.reindex([s for s in order if s in summary.index])
    summary.index.name = "Strategy"
    summary.columns    = ["Mean (s)", "Std (s)", "Median (s)", "Min (s)", "Max (s)"]
    print(summary.to_string())
    print()


def _validate_ranking(df: pd.DataFrame) -> None:
    """
    PRD §13 qualitative validation check.

    Expected ranking (based on published literature):
      Steffen ≈ Outside-In < Random ≤ Back-to-Front

    Prints a warning if results contradict this expectation.
    Do NOT tune parameters to force the result.
    """
    means = df.groupby("strategy")["total_boarding_time_s"].mean()

    bt_random   = means.get("random",        float("inf"))
    bt_btf      = means.get("back_to_front", float("inf"))
    bt_oi       = means.get("outside_in",    float("inf"))
    bt_steffen  = means.get("steffen",       float("inf"))

    problems = []
    if bt_btf < bt_random * 0.90:
        problems.append(
            "back_to_front is significantly FASTER than random "
            "(expected: back_to_front ≥ random)"
        )
    if bt_oi > bt_random * 1.10:
        problems.append(
            "outside_in is significantly SLOWER than random "
            "(expected: outside_in < random)"
        )
    if bt_steffen > bt_random * 1.10:
        problems.append(
            "steffen is significantly SLOWER than random "
            "(expected: steffen < random)"
        )

    if problems:
        print("  ⚠  VALIDATION WARNING – results may contradict published findings:")
        for p in problems:
            print(f"     • {p}")
        print("     Investigate before proceeding. Do NOT tune parameters.")
        print("     (PRD §13: 'if results contradict these, print a warning')")
    else:
        print("  ✓  Qualitative ranking agrees with published findings.")
        label_means = {
            "random":        bt_random,
            "outside_in":    bt_oi,
            "steffen":       bt_steffen,
            "back_to_front": bt_btf,
        }
        ranked = sorted(label_means.items(), key=lambda kv: kv[1])
        print("     Ranking (fastest → slowest): "
              + " < ".join(f"{k} ({v:.0f} s)" for k, v in ranked))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(scenario="baseline")
    n_reps    = cfg["experiment"]["replications"]
    base_seed = cfg["experiment"]["base_seed"]

    print("=" * 65)
    print("Airplane Boarding Simulation – Experiment 1: Baseline")
    print("=" * 65)
    print(f"\n  Aircraft     : Airbus A320  "
          f"({cfg['cabin']['rows']} rows × "
          f"{len(cfg['cabin']['seat_letters'])} seats = "
          f"{cfg['cabin']['rows'] * len(cfg['cabin']['seat_letters'])} passengers)")
    print(f"  Scenario     : baseline (all behaviour features OFF)")
    print(f"  Replications : {n_reps}")
    print(f"  Base seed    : {base_seed}")
    print(f"  Strategies   : {', '.join(STRATEGIES.keys())}")
    print()

    # ---- Run experiment --------------------------------------------------
    t0 = time.perf_counter()
    df = run_experiment(
        strategies=STRATEGIES,
        scenario="baseline",
        cfg=cfg,
        experiment_name="exp1_baseline",
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Completed {len(df)} runs in {elapsed:.1f} s "
          f"({elapsed / len(df) * 1000:.0f} ms / run)\n")

    # ---- Save CSV --------------------------------------------------------
    csv_path = RESULTS_DIR / "exp1_baseline.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved  : {csv_path}")

    # ---- Print summary table --------------------------------------------
    print()
    print("  Summary statistics – total boarding time (seconds)")
    print("  " + "-" * 55)
    _print_summary(df)

    # ---- Validation check -----------------------------------------------
    _validate_ranking(df)

    # ---- Chart -----------------------------------------------------------
    png_path = str(RESULTS_DIR / "fig1_boxplot_baseline.png")
    plot_boxplot_boarding_time(
        df,
        output_path=png_path,
        scenario_label="Baseline",
        n_reps=n_reps,
    )

    print()
    print("=" * 65)
    print("  Experiment 1 complete.  Run 'python run_experiments.py' for")
    print("  the full Philippine scenario + sweep experiments (M5/M6).")
    print("=" * 65)


if __name__ == "__main__":
    main()
