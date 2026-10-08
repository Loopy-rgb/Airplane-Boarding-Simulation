"""
boarding/charts.py
==================
Matplotlib chart generators.  All functions save a PNG to *output_path*
and return nothing.  Axes labels, titles, and colours follow the PRD §11
"readable colours" requirement.

Charts implemented (M4 and M6)
-------------------------------
M4:
  plot_boxplot_boarding_time  – Chart 1: boxplot by strategy

M6 (stubs here, implemented in M6):
  plot_interference_bars      – Chart 2: aisle + seat events by strategy
  plot_compliance_line        – Chart 3: boarding time vs compliance rate
  plot_group_fraction_line    – Chart 4: boarding time vs group fraction
  plot_sensitivity_heatmap    – Chart 5: strategy ranking under sensitivity
  plot_ablation_bars          – Chart 6: slowdown (%) by behaviour feature
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Display names for strategy labels on plots.
STRATEGY_LABELS: dict[str, str] = {
    "random":        "Random",
    "back_to_front": "Back-to-Front\n(Zones)",
    "outside_in":    "Outside-In\n(WilMA)",
    "steffen":       "Steffen",
}

#: Canonical strategy order for x-axis consistency across all charts.
STRATEGY_ORDER: list[str] = ["random", "back_to_front", "outside_in", "steffen"]

#: Colour palette – one per strategy (used consistently across all charts).
STRATEGY_COLORS: dict[str, str] = {
    "random":        "#4C9BE8",   # sky blue
    "back_to_front": "#F5A623",   # amber
    "outside_in":    "#27AE60",   # emerald
    "steffen":       "#9B59B6",   # purple
}

#: Scenario colours for side-by-side plots (M6).
SCENARIO_COLORS: dict[str, str] = {
    "baseline":   "#4C9BE8",
    "philippine": "#E74C3C",
}


def _ensure_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not already exist."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def _strategy_order_present(df: pd.DataFrame) -> list[str]:
    """Return STRATEGY_ORDER filtered to strategies that appear in *df*."""
    present = df["strategy"].unique()
    return [s for s in STRATEGY_ORDER if s in present]


# ---------------------------------------------------------------------------
# Chart 1 – Boxplot of total boarding time by strategy
# ---------------------------------------------------------------------------

def plot_boxplot_boarding_time(
    df: pd.DataFrame,
    output_path: str,
    scenario_label: str = "Baseline",
    n_reps: int | None = None,
) -> None:
    """
    Boxplot of ``total_boarding_time_s`` grouped by strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment results (one row per run).  Must contain columns
        ``strategy`` and ``total_boarding_time_s``.
    output_path : str
        Full path where the PNG is saved (parent dir created if needed).
    scenario_label : str
        Human-readable scenario name used in the title.
    n_reps : int, optional
        Number of replications; included in the title if provided.
    """
    _ensure_dir(output_path)

    strat_order = _strategy_order_present(df)
    data   = [df[df["strategy"] == s]["total_boarding_time_s"].values
               for s in strat_order]
    labels = [STRATEGY_LABELS.get(s, s) for s in strat_order]
    colors = [STRATEGY_COLORS[s] for s in strat_order]

    fig, ax = plt.subplots(figsize=(9, 6))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    rep_str = f", {n_reps} replications" if n_reps else ""
    ax.set_title(
        f"Total Boarding Time by Strategy\n"
        f"({scenario_label} scenario{rep_str})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Boarding Strategy", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Boarding Time (seconds)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Annotate median values above each box
    for i, d in enumerate(data):
        med = float(np.median(d))
        ax.text(i + 1, med + ax.get_ylim()[1] * 0.01,
                f"{med:.0f} s", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_boxplot_baseline_vs_philippine(
    df: pd.DataFrame,
    output_path: str,
    n_reps: int | None = None,
) -> None:
    """
    Side-by-side boxplot comparing baseline and Philippine scenarios
    for each strategy (Chart 1 final version, produced in M5/M6).

    Parameters
    ----------
    df : pd.DataFrame
        Combined results from Experiments 1 and 2.  Must contain
        ``strategy``, ``scenario``, and ``total_boarding_time_s``.
    output_path : str
        Full path where the PNG is saved.
    n_reps : int, optional
        Replications per scenario per strategy (for the title).
    """
    _ensure_dir(output_path)

    scenarios     = ["baseline", "philippine"]
    strat_order   = _strategy_order_present(df)
    n_strats      = len(strat_order)
    n_scens       = len(scenarios)
    group_width   = 0.8
    bar_width     = group_width / n_scens
    offsets       = np.linspace(-group_width / 2 + bar_width / 2,
                                 group_width / 2 - bar_width / 2, n_scens)

    fig, ax = plt.subplots(figsize=(11, 6))

    for s_idx, scenario in enumerate(scenarios):
        sub = df[df["scenario"] == scenario]
        for st_idx, strat in enumerate(strat_order):
            vals = sub[sub["strategy"] == strat]["total_boarding_time_s"].values
            pos  = st_idx + 1 + offsets[s_idx]
            bp   = ax.boxplot(
                vals,
                positions=[pos],
                widths=bar_width * 0.85,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            )
            color = SCENARIO_COLORS[scenario]
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)

    ax.set_xticks(range(1, n_strats + 1))
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strat_order], fontsize=10)

    legend_patches = [
        mpatches.Patch(facecolor=SCENARIO_COLORS[sc], alpha=0.7, label=sc.capitalize())
        for sc in scenarios
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10)

    rep_str = f", {n_reps} replications each" if n_reps else ""
    ax.set_title(
        f"Total Boarding Time: Baseline vs Philippine Scenario{rep_str}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Boarding Strategy", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Boarding Time (seconds)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Chart 2 – Bar chart of interference events (M6 stub)
# ---------------------------------------------------------------------------

def plot_interference_bars(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Grouped bar chart of mean aisle and seat interference events per strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment results with columns ``strategy``,
        ``aisle_interference_events``, ``seat_interference_events``.
    output_path : str
        Full path where PNG is saved.
    """
    _ensure_dir(output_path)

    strat_order = _strategy_order_present(df)
    n_strats    = len(strat_order)

    aisle_means = [
        df[df["strategy"] == s]["aisle_interference_events"].mean()
        for s in strat_order
    ]
    seat_means = [
        df[df["strategy"] == s]["seat_interference_events"].mean()
        for s in strat_order
    ]

    x        = np.arange(n_strats)
    bar_w    = 0.35
    labels   = [STRATEGY_LABELS.get(s, s) for s in strat_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_a = ax.bar(x - bar_w / 2, aisle_means, bar_w,
                    label="Aisle interference", color="#3498DB", alpha=0.8)
    bars_s = ax.bar(x + bar_w / 2, seat_means, bar_w,
                    label="Seat interference",  color="#E74C3C", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean events per run", fontsize=11)
    ax.set_xlabel("Boarding Strategy", fontsize=11, labelpad=8)
    ax.set_title("Mean Interference Events by Strategy",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Value labels on top of bars
    for bar in list(bars_a) + list(bars_s):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                f"{h:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Chart 3 – Compliance sweep line chart (M6 stub)
# ---------------------------------------------------------------------------

def plot_compliance_line(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Line chart of mean boarding time vs compliance_rate with 95% CI shading.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment 3 results; must contain ``compliance_rate``,
        ``strategy``, ``total_boarding_time_s``.
    output_path : str
    """
    _ensure_dir(output_path)
    _line_sweep_chart(
        df=df,
        x_col="compliance_rate",
        y_col="total_boarding_time_s",
        x_label="Compliance Rate",
        y_label="Mean Total Boarding Time (seconds)",
        title=("Effect of Passenger Compliance Rate on Boarding Time\n"
               "(Philippine scenario, 200 replications per level)"),
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Chart 4 – Group fraction sweep line chart (M6 stub)
# ---------------------------------------------------------------------------

def plot_group_fraction_line(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Line chart of mean boarding time vs group_fraction with 95% CI shading.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment 4 results; must contain ``group_fraction``,
        ``strategy``, ``total_boarding_time_s``.
    output_path : str
    """
    _ensure_dir(output_path)
    _line_sweep_chart(
        df=df,
        x_col="group_fraction",
        y_col="total_boarding_time_s",
        x_label="Group Travel Fraction",
        y_label="Mean Total Boarding Time (seconds)",
        title=("Effect of Group Travel Fraction on Boarding Time\n"
               "(Philippine scenario, 200 replications per level)"),
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Chart 5 – Sensitivity heatmap / table (M6 stub)
# ---------------------------------------------------------------------------

def plot_sensitivity_heatmap(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    4-panel facet grid: mean boarding time vs stow_multiplier,
    one line per walk_speed_mean, one panel per strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment 5 results with ``stow_multiplier``, ``walk_speed_mean``,
        ``strategy``, ``total_boarding_time_s``.
    output_path : str
    """
    _ensure_dir(output_path)

    strat_order  = _strategy_order_present(df)
    walk_speeds  = sorted(df["walk_speed_mean"].unique())
    stow_mults   = sorted(df["stow_multiplier"].unique())
    speed_colors = ["#E74C3C", "#3498DB", "#27AE60"]   # slow → fast
    speed_labels = {spd: f"{spd} m/s" for spd in walk_speeds}

    n_strats = len(strat_order)
    fig, axes = plt.subplots(1, n_strats, figsize=(4 * n_strats, 4.5),
                             sharey=True)
    if n_strats == 1:
        axes = [axes]

    for ax, strat in zip(axes, strat_order):
        sub = df[df["strategy"] == strat]
        for spd, color in zip(walk_speeds, speed_colors):
            ssub  = sub[sub["walk_speed_mean"] == spd]
            means = ssub.groupby("stow_multiplier")["total_boarding_time_s"].mean()
            stes  = ssub.groupby("stow_multiplier")["total_boarding_time_s"].sem()
            xs    = means.index.values
            ys    = means.values
            errs  = 1.96 * stes.values
            ax.plot(xs, ys, marker="o", color=color,
                    label=speed_labels[spd], linewidth=2)
            ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=0.15)

        ax.set_title(STRATEGY_LABELS.get(strat, strat), fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("Stow Time Multiplier", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean Total Boarding Time (s)", fontsize=10)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(walk_speeds),
               fontsize=9, title="Walk Speed", title_fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Sensitivity Analysis: Stow Time vs Walk Speed",
                 fontsize=12, fontweight="bold", y=1.08)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Chart 6 – Ablation bar chart (M6 stub)
# ---------------------------------------------------------------------------

def plot_ablation_bars(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Bar chart of mean boarding time by ablation feature and strategy.

    Each feature group shows 4 bars (one per strategy).  The "baseline"
    group anchors the comparison.  A horizontal dashed line marks the
    overall baseline for the best strategy (Steffen).

    Parameters
    ----------
    df : pd.DataFrame
        Experiment 6 results with ``ablation_feature``, ``strategy``,
        ``total_boarding_time_s``.
    output_path : str
    """
    _ensure_dir(output_path)

    strat_order   = _strategy_order_present(df)
    features      = df["ablation_feature"].unique()
    # Put baseline first
    feat_order    = ["baseline"] + [f for f in features if f != "baseline"]
    feat_order    = [f for f in feat_order if f in features]

    n_feats  = len(feat_order)
    n_strats = len(strat_order)
    bar_w    = 0.18
    x        = np.arange(n_feats)
    offsets  = np.linspace(-(n_strats - 1) * bar_w / 2,
                            (n_strats - 1) * bar_w / 2, n_strats)

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_feats), 5))

    for strat, offset in zip(strat_order, offsets):
        means = [
            df[(df["ablation_feature"] == feat) &
               (df["strategy"] == strat)]["total_boarding_time_s"].mean()
            for feat in feat_order
        ]
        ax.bar(x + offset, means, bar_w,
               label=STRATEGY_LABELS.get(strat, strat),
               color=STRATEGY_COLORS[strat], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.replace("_", "\n") for f in feat_order],
        fontsize=9,
    )
    ax.set_ylabel("Mean Total Boarding Time (s)", fontsize=11)
    ax.set_xlabel("Ablation Feature", fontsize=11, labelpad=8)
    ax.set_title("Ablation Study: Effect of Each Filipino Behaviour Feature",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Private helper: line sweep chart with CI bands (shared by Charts 3 and 4)
# ---------------------------------------------------------------------------

def _line_sweep_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str,
) -> None:
    """Draw a line chart of mean *y_col* vs *x_col* with 95% SEM CI bands."""
    strat_order = _strategy_order_present(df)
    x_values    = sorted(df[x_col].unique())

    fig, ax = plt.subplots(figsize=(9, 5))

    for strat in strat_order:
        sub   = df[df["strategy"] == strat]
        means = sub.groupby(x_col)[y_col].mean()
        stes  = sub.groupby(x_col)[y_col].sem()
        xs    = np.array([means.index.get_loc(x) for x in x_values], dtype=float)
        xs    = np.array(x_values, dtype=float)
        ys    = means.reindex(x_values).values
        errs  = 1.96 * stes.reindex(x_values).values
        color = STRATEGY_COLORS[strat]
        ax.plot(xs, ys, marker="o", color=color, linewidth=2,
                label=STRATEGY_LABELS.get(strat, strat))
        ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=0.15)

    ax.set_xlabel(x_label, fontsize=11, labelpad=8)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

