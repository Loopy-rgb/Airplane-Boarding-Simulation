"""
boarding/stats.py
=================
Statistical analysis of simulation results.

Two test families are provided because CRN (common random numbers) makes
paired tests valid and more powerful:

  - **Paired t-test** (Welch) between every strategy pair, pairing on
    replication *i* so each pair uses the same passenger population.
    Bonferroni-corrected.  This is the primary test (PRD §11).

  - **Mann-Whitney U** (non-parametric fallback) for robustness checking,
    since boarding time is not guaranteed normal.

  - **One-way ANOVA** (repeated-measures approximation) as a global
    omnibus test before pairwise comparisons.

  - **Rank table**: per condition, rank each strategy 1 (fastest) to 4;
    report how rankings change across conditions.

Public interface
----------------
    summary_table(df, group_col)                         -> pd.DataFrame
    paired_ttest_pairwise(df, group_col, value_col)      -> pd.DataFrame
    mannwhitney_pairwise(df, group_col, value_col)       -> pd.DataFrame
    one_way_anova(df, group_col, value_col)              -> dict
    rank_strategies(df, group_col, value_col)            -> pd.DataFrame
    cohens_d(a, b)                                       -> float
    bootstrap_ci(a, b, n_boot, alpha, seed)              -> (float, float)
    compare_scenarios(df_base, df_ph, value_col)         -> pd.DataFrame
    print_report(df_base, df_ph)                         -> None
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_rel, f_oneway


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def summary_table(
    df: pd.DataFrame,
    group_col: str = "strategy",
    value_col: str = "total_boarding_time_s",
) -> pd.DataFrame:
    """
    Return a descriptive statistics table grouped by *group_col*.

    Columns: mean, std, median, q25, q75, min, max, n.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment results.
    group_col : str
        Column to group by (default ``"strategy"``).
    value_col : str
        Metric of interest (default ``"total_boarding_time_s"``).

    Returns
    -------
    pd.DataFrame
        One row per group, statistics as columns.
    """
    grp = df.groupby(group_col)[value_col]
    tbl = grp.agg(
        mean="mean",
        std="std",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        min="min",
        max="max",
        n="count",
    ).round(1)
    tbl.columns.name = None
    return tbl


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d effect size between samples *a* and *b*.

    Uses the pooled standard deviation (biased denominator) for consistency
    with two-sample t-test effect sizes.

    Interpretation (Cohen 1988):
      |d| < 0.2 : negligible
      |d| < 0.5 : small
      |d| < 0.8 : medium
      |d| ≥ 0.8 : large

    Parameters
    ----------
    a, b : array_like
        Observation samples.

    Returns
    -------
    float
        Cohen's d (positive when mean(a) > mean(b)).
    """
    a, b   = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_var = (
        (n1 - 1) * float(np.var(a, ddof=1))
        + (n2 - 1) * float(np.var(b, ddof=1))
    ) / (n1 + n2 - 2)
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / pooled_sd


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for the mean difference
# ---------------------------------------------------------------------------

def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Bootstrap (1 - alpha) confidence interval for mean(a) - mean(b).

    Uses 2 000 resamples (percentile method).

    Parameters
    ----------
    a, b : array_like
        Observation samples.
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (default 0.05 → 95% CI).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (lo, hi) : (float, float)
        Lower and upper bounds of the CI.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng  = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa      = rng.choice(a, size=len(a), replace=True)
        sb      = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.mean(sa) - np.mean(sb)
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Paired t-test (primary PRD §11 test — valid due to CRN)
# ---------------------------------------------------------------------------

def paired_ttest_pairwise(
    df: pd.DataFrame,
    group_col: str = "strategy",
    value_col: str = "total_boarding_time_s",
) -> pd.DataFrame:
    """
    All pairwise **paired** Welch t-tests between groups in *group_col*.

    Pairing is on ``rep_id``: for each replication both strategies saw the
    same passenger population (CRN), so the paired test is valid and more
    powerful than an independent test.

    Bonferroni-corrects p-values.

    Parameters
    ----------
    df : pd.DataFrame
    group_col, value_col : str

    Returns
    -------
    pd.DataFrame
        Columns: group_a, group_b, t_stat, p_raw, p_bonferroni,
                 cohens_d, ci_lo_95, ci_hi_95, significant_005.
    """
    groups  = sorted(df[group_col].unique())
    pairs   = list(itertools.combinations(groups, 2))
    n_pairs = len(pairs)
    rows    = []

    for g_a, g_b in pairs:
        sub_a = df[df[group_col] == g_a].sort_values("rep_id")
        sub_b = df[df[group_col] == g_b].sort_values("rep_id")
        a = sub_a[value_col].values
        b = sub_b[value_col].values

        # Align lengths (should always match under CRN)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        t_stat, p_raw = ttest_rel(a, b)
        p_bonf        = min(1.0, p_raw * n_pairs)
        d             = cohens_d(a, b)
        lo, hi        = bootstrap_ci(a, b)

        rows.append({
            "group_a":         g_a,
            "group_b":         g_b,
            "t_stat":          round(t_stat, 3),
            "p_raw":           round(abs(p_raw), 6),
            "p_bonferroni":    round(p_bonf, 6),
            "cohens_d":        round(d, 3),
            "ci_lo_95":        round(lo, 1),
            "ci_hi_95":        round(hi, 1),
            "significant_005": p_bonf < 0.05,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One-way ANOVA (omnibus test)
# ---------------------------------------------------------------------------

def one_way_anova(
    df: pd.DataFrame,
    group_col: str = "strategy",
    value_col: str = "total_boarding_time_s",
) -> dict:
    """
    One-way ANOVA across all groups in *group_col*.

    Returns a dict with keys ``F``, ``p``, ``significant_005``.
    A significant result justifies pairwise comparisons.

    Parameters
    ----------
    df : pd.DataFrame
    group_col, value_col : str

    Returns
    -------
    dict
        {"F": float, "p": float, "significant_005": bool,
         "groups_tested": list[str]}
    """
    groups = sorted(df[group_col].unique())
    arrays = [df[df[group_col] == g][value_col].values for g in groups]
    F, p   = f_oneway(*arrays)
    return {
        "F":               round(float(F), 3),
        "p":               round(float(p), 8),
        "significant_005": float(p) < 0.05,
        "groups_tested":   groups,
    }


# ---------------------------------------------------------------------------
# Strategy ranking table
# ---------------------------------------------------------------------------

def rank_strategies(
    df: pd.DataFrame,
    group_col: str = "strategy",
    value_col: str = "total_boarding_time_s",
    condition_col: str | None = None,
) -> pd.DataFrame:
    """
    Rank strategies from 1 (fastest) to N (slowest) by mean *value_col*.

    If *condition_col* is provided (e.g., ``"compliance_rate"``), ranks
    are computed separately for each condition value, producing a table
    that shows how rankings change across the sweep.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str      Column identifying the strategy.
    value_col : str      Metric to rank by.
    condition_col : str  Optional sweep column (compliance_rate, etc.).

    Returns
    -------
    pd.DataFrame
        If no condition_col: one row per strategy, columns: strategy, mean, rank.
        If condition_col given: strategies as rows, condition values as columns,
        cells are rank integers (1 = fastest).
    """
    strategies = sorted(df[group_col].unique())

    if condition_col is None:
        means = df.groupby(group_col)[value_col].mean()
        ranked = means.rank(method="min").astype(int)
        result = pd.DataFrame({
            "strategy": strategies,
            "mean":     [round(float(means[s]), 1) for s in strategies],
            "rank":     [int(ranked[s]) for s in strategies],
        }).sort_values("rank").reset_index(drop=True)
        return result

    # Per-condition ranking
    conditions = sorted(df[condition_col].unique())
    rank_data  = {}
    for cond in conditions:
        sub   = df[df[condition_col] == cond]
        means = sub.groupby(group_col)[value_col].mean()
        ranked = means.rank(method="min").astype(int)
        rank_data[cond] = {s: int(ranked.get(s, 0)) for s in strategies}

    result = pd.DataFrame(rank_data, index=strategies)
    result.index.name = group_col
    return result


# ---------------------------------------------------------------------------
# Pairwise Mann-Whitney (non-parametric fallback)
# ---------------------------------------------------------------------------

def mannwhitney_pairwise(
    df: pd.DataFrame,
    group_col: str = "strategy",
    value_col: str = "total_boarding_time_s",
) -> pd.DataFrame:
    """
    All pairwise Mann-Whitney U tests between groups in *group_col*.

    Bonferroni-corrects p-values for the number of comparisons.

    Parameters
    ----------
    df : pd.DataFrame
    group_col, value_col : str

    Returns
    -------
    pd.DataFrame
        Columns: group_a, group_b, U_stat, p_raw, p_bonferroni,
                 cohens_d, ci_lo_95, ci_hi_95, significant_005.
    """
    groups  = sorted(df[group_col].unique())
    pairs   = list(itertools.combinations(groups, 2))
    n_pairs = len(pairs)
    rows    = []
    for g_a, g_b in pairs:
        a = df[df[group_col] == g_a][value_col].values
        b = df[df[group_col] == g_b][value_col].values
        U, p_raw = mannwhitneyu(a, b, alternative="two-sided")
        p_bonf   = min(1.0, p_raw * n_pairs)
        d        = cohens_d(a, b)
        lo, hi   = bootstrap_ci(a, b)
        rows.append({
            "group_a":          g_a,
            "group_b":          g_b,
            "U_stat":           round(U),
            "p_raw":            round(p_raw, 6),
            "p_bonferroni":     round(p_bonf, 6),
            "cohens_d":         round(d, 3),
            "ci_lo_95":         round(lo, 1),
            "ci_hi_95":         round(hi, 1),
            "significant_005":  p_bonf < 0.05,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario comparison (baseline vs Philippine)
# ---------------------------------------------------------------------------

def compare_scenarios(
    df_base: pd.DataFrame,
    df_ph: pd.DataFrame,
    value_col: str = "total_boarding_time_s",
    group_col: str = "strategy",
) -> pd.DataFrame:
    """
    Compare baseline vs Philippine scenario for each strategy.

    Performs Mann-Whitney U and computes Cohen's d and 95% bootstrap CI
    for (Philippine mean – baseline mean).

    Parameters
    ----------
    df_base : pd.DataFrame   Experiment 1 results.
    df_ph   : pd.DataFrame   Experiment 2 results.
    value_col, group_col : str

    Returns
    -------
    pd.DataFrame
        One row per strategy, columns:
        strategy, mean_base, mean_ph, delta_s, delta_pct,
        U_stat, p_bonferroni, cohens_d, ci_lo_95, ci_hi_95.
    """
    strategies = sorted(df_base[group_col].unique())
    n_strats   = len(strategies)
    rows       = []
    for strat in strategies:
        a   = df_base[df_base[group_col] == strat][value_col].values
        b   = df_ph  [df_ph  [group_col] == strat][value_col].values
        U, p_raw = mannwhitneyu(a, b, alternative="two-sided")
        p_bonf   = min(1.0, p_raw * n_strats)
        d        = cohens_d(b, a)            # positive = Philippine worse
        lo, hi   = bootstrap_ci(b, a)        # CI for mean_ph - mean_base
        delta_s   = float(np.mean(b) - np.mean(a))
        delta_pct = delta_s / float(np.mean(a)) * 100.0
        rows.append({
            "strategy":     strat,
            "mean_base":    round(float(np.mean(a)), 1),
            "mean_ph":      round(float(np.mean(b)), 1),
            "delta_s":      round(delta_s, 1),
            "delta_pct":    round(delta_pct, 1),
            "U_stat":       round(U),
            "p_bonferroni": round(p_bonf, 6),
            "cohens_d":     round(d, 3),
            "ci_lo_95":     round(lo, 1),
            "ci_hi_95":     round(hi, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full text report
# ---------------------------------------------------------------------------

def print_report(
    df_base: pd.DataFrame,
    df_ph: pd.DataFrame,
    value_col: str = "total_boarding_time_s",
) -> None:
    """
    Print a formatted statistical report to stdout (PRD §11).

    Sections
    --------
    1. Baseline – descriptive statistics
    2. Philippine – descriptive statistics
    3. Strategy ranking (baseline, fastest → slowest)
    4. One-way ANOVA – baseline (omnibus test)
    5. Paired t-test pairwise – baseline (primary, CRN-valid)
    6. Mann-Whitney pairwise – baseline (non-parametric check)
    7. Scenario comparison (Philippine − Baseline) per strategy
    """
    sep  = "=" * 70
    sep2 = "-" * 70

    print(f"\n{sep}")
    print("STATISTICAL REPORT - AIRPLANE BOARDING SIMULATION")
    print(f"Metric: {value_col}")
    print(f"{sep}\n")

    # 1. Descriptive statistics - baseline
    print("1. Baseline scenario - descriptive statistics")
    print(sep2)
    print(summary_table(df_base, value_col=value_col).to_string())
    print()

    # 2. Descriptive statistics - Philippine
    print("2. Philippine scenario - descriptive statistics")
    print(sep2)
    print(summary_table(df_ph, value_col=value_col).to_string())
    print()

    # 3. Strategy ranking (baseline)
    print("3. Strategy ranking - baseline (1 = fastest)")
    print(sep2)
    print(rank_strategies(df_base, value_col=value_col).to_string(index=False))
    print()

    # 4. One-way ANOVA
    print("4. One-way ANOVA - baseline scenario")
    print(sep2)
    anova = one_way_anova(df_base, value_col=value_col)
    sig = anova["significant_005"]
    print(f"   F = {anova['F']},  p = {anova['p']:.2e},  significant (a=0.05): {sig}")
    if sig:
        print("   -> Proceed to pairwise comparisons (global difference confirmed).")
    print()

    # 5. Paired t-test (primary - CRN makes pairing valid)
    print("5. Paired t-test (CRN-paired, Bonferroni-corrected) - baseline")
    print("   Pairing on rep_id is valid: CRN gives same passengers to all strategies.")
    print(sep2)
    pt = paired_ttest_pairwise(df_base, value_col=value_col)
    print(pt[["group_a", "group_b", "t_stat", "p_bonferroni",
               "cohens_d", "ci_lo_95", "ci_hi_95", "significant_005"
              ]].to_string(index=False))
    print()

    # 6. Mann-Whitney (non-parametric robustness check)
    print("6. Mann-Whitney U (non-parametric, Bonferroni-corrected) - baseline")
    print(sep2)
    mw = mannwhitney_pairwise(df_base, value_col=value_col)
    print(mw[["group_a", "group_b", "U_stat", "p_bonferroni",
               "cohens_d", "significant_005"]].to_string(index=False))
    print()

    # 7. Scenario comparison (Philippine minus Baseline)
    print("7. Scenario comparison (Philippine - Baseline) per strategy")
    print(sep2)
    sc = compare_scenarios(df_base, df_ph, value_col=value_col)
    print(sc.to_string(index=False))
    print()
    print(f"{sep}\n")
