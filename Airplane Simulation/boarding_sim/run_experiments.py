"""
run_experiments.py
==================
Experiments 2–6: Philippine scenario and sweep experiments.

Usage
-----
    python run_experiments.py [--exp EXP]

    --exp : comma-separated list of experiment numbers to run, e.g. '2,3,4'
            Default: run all (2,3,4,5,6)

Outputs (all in results/)
-------------------------
  exp2_philippine.csv           + fig2_boxplot_comparison.png  (Exp 2)
  exp3_compliance_sweep.csv     + fig3_compliance_line.png     (Exp 3)
  exp4_group_sweep.csv          + fig4_group_line.png          (Exp 4)
  exp5_sensitivity.csv          + fig5_sensitivity_heatmap.png (Exp 5)
  exp6_ablation.csv             + fig6_ablation_bars.png       (Exp 6)
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd

from boarding import load_config, apply_scenario
from boarding.experiments import run_experiment
from boarding.strategies import STRATEGIES
from boarding.charts import (
    plot_boxplot_baseline_vs_philippine,
    plot_interference_bars,
    plot_compliance_line,
    plot_group_fraction_line,
    plot_sensitivity_heatmap,
    plot_ablation_bars,
)

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Experiment 2 – Philippine scenario
# ---------------------------------------------------------------------------

def run_exp2(base_cfg: dict) -> pd.DataFrame:
    """200 reps × 4 strategies, Philippine scenario."""
    ph_cfg = load_config(scenario="philippine")

    print("\n── Experiment 2: Philippine Scenario ──")
    df_ph = run_experiment(
        strategies=STRATEGIES,
        scenario="philippine",
        cfg=ph_cfg,
        experiment_name="exp2_philippine",
    )

    csv = RESULTS_DIR / "exp2_philippine.csv"
    df_ph.to_csv(csv, index=False)
    print(f"  CSV: {csv}")

    # Side-by-side boxplot with baseline (load or re-run baseline)
    bl_csv = RESULTS_DIR / "exp1_baseline.csv"
    if bl_csv.exists():
        df_bl = pd.read_csv(bl_csv)
        df_combined = pd.concat([df_bl, df_ph], ignore_index=True)
        plot_boxplot_baseline_vs_philippine(
            df_combined,
            output_path=str(RESULTS_DIR / "fig2_boxplot_comparison.png"),
            n_reps=base_cfg["experiment"]["replications"],
        )
    else:
        print("  ⚠  Baseline CSV not found; run run_baseline.py first.")

    # Summary
    print("\n  Philippine scenario – mean boarding times (s):")
    ph_means = df_ph.groupby("strategy")["total_boarding_time_s"].mean().round(1)
    print(ph_means.to_string())

    return df_ph


# ---------------------------------------------------------------------------
# Experiment 3 – Compliance sweep
# ---------------------------------------------------------------------------

def run_exp3(base_cfg: dict) -> pd.DataFrame:
    """Philippine scenario with compliance_rate swept over [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]."""
    compliance_levels = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]
    print("\n── Experiment 3: Compliance Rate Sweep ──")

    dfs = []
    for rate in compliance_levels:
        # Deep-copy base config and override compliance_rate
        cfg = load_config(scenario="philippine")
        cfg["behavior"]["non_compliance"]["enabled"] = True
        cfg["behavior"]["non_compliance"]["compliance_rate"] = rate
        cfg["experiment"]["replications"] = base_cfg["experiment"]["replications"]

        df = run_experiment(
            strategies=STRATEGIES,
            scenario="philippine",
            cfg=cfg,
            experiment_name=f"exp3_compliance_{rate:.2f}",
            sweep_params={"compliance_rate": rate},
        )
        dfs.append(df)
        print(f"  compliance_rate={rate:.2f}  done.")

    df_all = pd.concat(dfs, ignore_index=True)
    csv = RESULTS_DIR / "exp3_compliance_sweep.csv"
    df_all.to_csv(csv, index=False)
    print(f"  CSV: {csv}")

    try:
        plot_compliance_line(df_all, str(RESULTS_DIR / "fig3_compliance_line.png"))
    except NotImplementedError:
        print("  Chart fig3 deferred to M6.")

    return df_all


# ---------------------------------------------------------------------------
# Experiment 4 – Group fraction sweep
# ---------------------------------------------------------------------------

def run_exp4(base_cfg: dict) -> pd.DataFrame:
    """Philippine scenario with group_fraction swept over [0.0, 0.2, 0.4, 0.6]."""
    group_fractions = [0.0, 0.2, 0.4, 0.6]
    print("\n── Experiment 4: Group Fraction Sweep ──")

    dfs = []
    for frac in group_fractions:
        cfg = load_config(scenario="philippine")
        cfg["behavior"]["group_travel"]["enabled"] = (frac > 0)
        cfg["behavior"]["group_travel"]["group_fraction"] = frac
        cfg["experiment"]["replications"] = base_cfg["experiment"]["replications"]

        df = run_experiment(
            strategies=STRATEGIES,
            scenario="philippine",
            cfg=cfg,
            experiment_name=f"exp4_group_{frac:.1f}",
            sweep_params={"group_fraction": frac},
        )
        dfs.append(df)
        print(f"  group_fraction={frac:.1f}  done.")

    df_all = pd.concat(dfs, ignore_index=True)
    csv = RESULTS_DIR / "exp4_group_sweep.csv"
    df_all.to_csv(csv, index=False)
    print(f"  CSV: {csv}")

    try:
        plot_group_fraction_line(df_all, str(RESULTS_DIR / "fig4_group_line.png"))
    except NotImplementedError:
        print("  Chart fig4 deferred to M6.")

    return df_all


# ---------------------------------------------------------------------------
# Experiment 5 – Sensitivity analysis
# ---------------------------------------------------------------------------

def run_exp5(base_cfg: dict) -> pd.DataFrame:
    """Vary mean stow time (×0.5, ×1, ×1.5, ×2) and walk speed (0.6, 0.8, 1.0 m/s)."""
    stow_multipliers = [0.5, 1.0, 1.5, 2.0]
    walk_speeds      = [0.6, 0.8, 1.0]
    print("\n── Experiment 5: Sensitivity Analysis ──")

    dfs = []
    base_stow_mean = base_cfg["passenger"]["stow_time_s"]["per_bag"]["mean"]
    base_walk_mean = base_cfg["passenger"]["walk_speed_mps"]["mean"]

    for stow_mult in stow_multipliers:
        for walk_spd in walk_speeds:
            cfg = load_config(scenario="baseline")
            cfg["passenger"]["stow_time_s"]["per_bag"]["mean"] = base_stow_mean * stow_mult
            cfg["passenger"]["walk_speed_mps"]["mean"] = walk_spd
            cfg["experiment"]["replications"] = base_cfg["experiment"]["replications"]

            df = run_experiment(
                strategies=STRATEGIES,
                scenario="baseline",
                cfg=cfg,
                experiment_name=f"exp5_stow{stow_mult}_walk{walk_spd}",
                sweep_params={
                    "stow_multiplier": stow_mult,
                    "walk_speed_mean": walk_spd,
                },
            )
            dfs.append(df)
        print(f"  stow_mult={stow_mult}  done.")

    df_all = pd.concat(dfs, ignore_index=True)
    csv = RESULTS_DIR / "exp5_sensitivity.csv"
    df_all.to_csv(csv, index=False)
    print(f"  CSV: {csv}")

    try:
        plot_sensitivity_heatmap(df_all, str(RESULTS_DIR / "fig5_sensitivity_heatmap.png"))
    except NotImplementedError:
        print("  Chart fig5 deferred to M6.")

    return df_all


# ---------------------------------------------------------------------------
# Experiment 6 – Ablation
# ---------------------------------------------------------------------------

def run_exp6(base_cfg: dict) -> pd.DataFrame:
    """Turn each Filipino behaviour on one at a time; measure slowdown vs baseline."""
    print("\n── Experiment 6: Ablation ──")

    features = [
        "group_travel",
        "non_compliance",
        "late_passengers",
        "heavy_handcarry",
        "bayanihan",
        "seat_search_delay",
    ]

    dfs = []

    # Baseline reference (all off) – load if exists else re-run
    bl_csv = RESULTS_DIR / "exp1_baseline.csv"
    if bl_csv.exists():
        df_bl = pd.read_csv(bl_csv)
        df_bl["ablation_feature"] = "baseline"
        dfs.append(df_bl)
    else:
        cfg = load_config(scenario="baseline")
        df_bl = run_experiment(
            strategies=STRATEGIES, scenario="baseline", cfg=cfg,
            experiment_name="exp6_baseline",
            sweep_params={"ablation_feature": "baseline"},
        )
        dfs.append(df_bl)

    # One feature at a time
    for feature in features:
        cfg = load_config(scenario="baseline")
        # Enable only this one feature
        cfg["behavior"][feature]["enabled"] = True
        cfg["experiment"]["replications"] = base_cfg["experiment"]["replications"]

        df = run_experiment(
            strategies=STRATEGIES,
            scenario="baseline",
            cfg=cfg,
            experiment_name=f"exp6_{feature}",
            sweep_params={"ablation_feature": feature},
        )
        dfs.append(df)
        print(f"  Feature '{feature}'  done.")

    df_all = pd.concat(dfs, ignore_index=True)
    csv = RESULTS_DIR / "exp6_ablation.csv"
    df_all.to_csv(csv, index=False)
    print(f"  CSV: {csv}")

    try:
        plot_ablation_bars(df_all, str(RESULTS_DIR / "fig6_ablation_bars.png"))
    except NotImplementedError:
        print("  Chart fig6 deferred to M6.")

    return df_all


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run boarding-strategy sweep experiments (Experiments 2-6)."
    )
    parser.add_argument(
        "--exp",
        default="2,3,4,5,6",
        help="Comma-separated experiment numbers to run (default: 2,3,4,5,6).",
    )
    args = parser.parse_args()

    exp_to_run = {int(x.strip()) for x in args.exp.split(",") if x.strip().isdigit()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(scenario="baseline")

    print("=" * 65)
    print("Airplane Boarding Simulation – Experiments 2-6")
    print("=" * 65)
    print(f"  Running experiments: {sorted(exp_to_run)}")
    print(f"  Replications per condition: {base_cfg['experiment']['replications']}")
    print()

    t_start = time.perf_counter()

    results = {}
    if 2 in exp_to_run:
        results[2] = run_exp2(base_cfg)
    if 3 in exp_to_run:
        results[3] = run_exp3(base_cfg)
    if 4 in exp_to_run:
        results[4] = run_exp4(base_cfg)
    if 5 in exp_to_run:
        results[5] = run_exp5(base_cfg)
    if 6 in exp_to_run:
        results[6] = run_exp6(base_cfg)

    elapsed = time.perf_counter() - t_start
    print(f"\n\n  All requested experiments complete in {elapsed:.1f} s.")
    print("  Results in:", RESULTS_DIR)
    print("=" * 65)


if __name__ == "__main__":
    main()
