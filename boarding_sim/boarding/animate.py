"""
Visualise a single boarding run as a Matplotlib animation.

The animation shows a top-down (overhead) view of the cabin.
Passengers are coloured by state and move from left (door) to right (rear).

Coordinate system
-----------------
  x : row number (0 = row 1, near door; 29 = row 30, rear).
  y : cross-aisle position
        0 → A (window left)
        1 → B (middle left)
        2 → C (aisle left)
        3 → aisle / walking lane
        4 → D (aisle right)
        5 → E (middle right)
        6 → F (window right)

Passengers in the QUEUE wait to the left of the cabin (x < 0).

Public interface
----------------
    record_one_run(strategy_name, cfg, seed) -> dict
        Run a single simulation with history recording.

    animate_boarding(history, cfg, strategy_name, output_path, fps,
                     max_ticks) -> FuncAnimation
        Build and optionally save the animation.  Returns the animation
        object for display in Jupyter.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless; caller can switch to TkAgg/Qt5Agg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Seat letter → y-coordinate in the plot
_SEAT_Y: dict[str, float] = {
    "A": 0.0, "B": 1.0, "C": 2.0,   # left side
    "D": 4.0, "E": 5.0, "F": 6.0,   # right side
}
_AISLE_Y = 3.0                       # y-position while walking the aisle

# Queue area: passengers waiting to enter are shown to the left
_QUEUE_START_X = -2.5                # leftmost x for queued passengers
_QUEUE_STRIDE  =  0.35               # horizontal spacing per queued passenger

# State colours
_STATE_COLORS: dict[str, str] = {
    "QUEUED":   "#AAAAAA",     # light grey – waiting outside the plane
    "WALKING":  "#3498DB",     # blue       – moving through the aisle
    "BLOCKED":  "#E67E22",     # amber      – stuck behind another passenger
    "STOWING":  "#E74C3C",     # red        – stowing bags at row
    "SEATING":  "#F1C40F",     # yellow     – shuffling into seat
    "SEATED":   "#2ECC71",     # green      – seated
}
_MARKER_SIZE = 55       # scatter marker size (points²)


# ---------------------------------------------------------------------------
# Record one run
# ---------------------------------------------------------------------------

def record_one_run(
    strategy_name: str,
    cfg: dict,
    seed: int = 12345,
) -> dict:
    """
    Run one full simulation with history recording and return the result dict.

    The returned dict contains ``"history"`` (list of per-tick snapshots)
    in addition to the usual metrics keys.

    Parameters
    ----------
    strategy_name : str
        One of ``"random"``, ``"back_to_front"``, ``"outside_in"``,
        ``"steffen"``.
    cfg : dict
        Full simulation config (scenario already loaded).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Simulation result with ``"history"`` key populated.
    """
    import numpy as np
    from boarding import Cabin
    from boarding.simulation import generate_passengers, run_simulation
    from boarding.strategies import STRATEGIES
    from boarding.behavior import apply_behavior

    rng_pop   = np.random.default_rng(seed)
    passengers = generate_passengers(cfg, rng_pop)

    strategy_fn = STRATEGIES[strategy_name]
    rng_strat   = np.random.default_rng(seed + 1)
    order       = strategy_fn(passengers, rng_strat)

    rng_beh = np.random.default_rng(seed + 2)
    order   = apply_behavior(order, passengers, rng_beh, cfg)

    cabin      = Cabin(cfg["cabin"])
    rng_engine = np.random.default_rng(seed + 3)

    result = run_simulation(
        order=order,
        cabin=cabin,
        cfg=cfg,
        rng=rng_engine,
        record_history=True,
    )
    return result


# ---------------------------------------------------------------------------
# Build the animation
# ---------------------------------------------------------------------------

def animate_boarding(
    history: list[dict],
    cfg: dict,
    strategy_name: str = "",
    output_path: Optional[str] = None,
    fps: int = 10,
    max_frames: Optional[int] = None,
) -> FuncAnimation:
    """
    Build a Matplotlib FuncAnimation from a recorded history.

    The animation shows all passengers each tick as coloured scatter dots.
    Each dot's position encodes:
      - x = row (0-indexed) when in the aisle or seated
      - y = 3 (aisle lane) when WALKING/BLOCKED/STOWING
      - y = seat position (0-6) when SEATING or SEATED
    Queued passengers appear to the left of the cabin as a horizontal line.

    Parameters
    ----------
    history : list[dict]
        Output of ``run_simulation(..., record_history=True)["history"]``.
    cfg : dict
        Simulation config (used for cabin row/seat info).
    strategy_name : str
        Used in the plot title.
    output_path : str, optional
        If provided, save the animation as a GIF (requires Pillow).
    fps : int
        Frames per second for display and saving (default 10).
    max_frames : int, optional
        Subsample the history to at most this many frames for speed.

    Returns
    -------
    FuncAnimation
        The Matplotlib animation object (can be displayed in Jupyter with
        ``HTML(anim.to_jshtml())``).
    """
    n_rows        = cfg["cabin"]["rows"]           # 30
    seat_letters  = cfg["cabin"]["seat_letters"]   # ["A","B","C","D","E","F"]
    n_pax         = len(history[0]["states"])

    # Subsample frames for GIF
    frames = history
    if max_frames and len(frames) > max_frames:
        step   = max(1, len(frames) // max_frames)
        frames = frames[::step]

    # ---- Figure setup -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(_QUEUE_START_X - 0.5, n_rows + 0.5)
    ax.set_ylim(-0.8, 6.8)

    # Draw seat grid lines
    for row in range(n_rows):
        ax.axvline(row, color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.axhline(_AISLE_Y, color="#BBBBBB", linewidth=1.0, zorder=0, linestyle="--")

    # Seat labels on y-axis
    ax.set_yticks(list(_SEAT_Y.values()))
    ax.set_yticklabels(list(_SEAT_Y.keys()), fontsize=9)

    # Row labels (sparse, every 5 rows)
    ax.set_xticks([r for r in range(0, n_rows, 5)])
    ax.set_xticklabels([str(r + 1) for r in range(0, n_rows, 5)], fontsize=9)
    ax.set_xlabel("Row", fontsize=10)

    # Door indicator
    ax.axvline(-0.5, color="#555555", linewidth=2, zorder=0, label="Door")

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=col, label=state)
        for state, col in _STATE_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=7, ncol=2, framealpha=0.8)

    title = ax.set_title("", fontsize=11, fontweight="bold")

    # ---- Scatter artists (one per state) ------------------------------------
    scatter_artists: dict[str, plt.PathCollection] = {}
    for state, color in _STATE_COLORS.items():
        sc = ax.scatter([], [], s=_MARKER_SIZE, c=color,
                        alpha=0.85, zorder=5, edgecolors="none",
                        label=state)
        scatter_artists[state] = sc

    # ---- Counter text -------------------------------------------------------
    counter_text = ax.text(
        0.01, 0.97, "",
        transform=ax.transAxes, fontsize=9,
        va="top", ha="left", family="monospace",
    )

    # ---- Update function ----------------------------------------------------
    def _update(frame_idx: int):
        snap        = frames[frame_idx]
        tick        = snap["tick"]
        state_data  = snap["states"]

        # Bucket passengers by state
        buckets: dict[str, list[tuple[float, float]]] = {s: [] for s in _STATE_COLORS}

        # Assign queue positions for QUEUED passengers
        q_count = 0
        for pax in state_data:
            state = pax["state"]
            if state == "QUEUED":
                x = _QUEUE_START_X - q_count * _QUEUE_STRIDE
                y = _AISLE_Y
                q_count += 1
            elif state in ("WALKING", "BLOCKED", "STOWING"):
                ac = pax["aisle_cell"]
                x  = float(ac) if ac is not None else -1.0
                y  = _AISLE_Y
            elif state == "SEATING":
                ac = pax["aisle_cell"]
                x  = float(pax["row"] - 1)
                # midpoint between aisle and seat
                sy = _SEAT_Y.get(pax["letter"], _AISLE_Y)
                y  = (_AISLE_Y + sy) / 2.0
                if ac is not None:
                    x = float(ac)
            else:   # SEATED
                x = float(pax["row"] - 1)
                y = _SEAT_Y.get(pax["letter"], _AISLE_Y)

            buckets[state].append((x, y))

        for state, sc in scatter_artists.items():
            pts = np.array(buckets[state]) if buckets[state] else np.empty((0, 2))
            if len(pts):
                sc.set_offsets(pts)
            else:
                sc.set_offsets(np.empty((0, 2)))

        # Count by state
        counts = {s: len(v) for s, v in buckets.items()}
        n_seated  = counts["SEATED"]
        n_walking = counts["WALKING"] + counts["BLOCKED"] + counts["STOWING"]
        n_queue   = counts["QUEUED"]

        pct   = n_seated / n_pax * 100
        strat = f" – {strategy_name.replace('_', ' ').title()}" if strategy_name else ""
        title.set_text(
            f"Boarding Simulation{strat}  |  Tick {tick:4d} s  "
            f"|  Seated {n_seated}/{n_pax} ({pct:.0f}%)"
        )
        counter_text.set_text(
            f"In aisle: {n_walking:3d}   Queue: {n_queue:3d}"
        )
        return list(scatter_artists.values()) + [title, counter_text]

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=max(20, 1000 // fps),
        blit=False,
    )

    # ---- Save ---------------------------------------------------------------
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        suffix = pathlib.Path(output_path).suffix.lower()
        try:
            if suffix == ".gif":
                writer = PillowWriter(fps=fps)
                anim.save(output_path, writer=writer, dpi=100)
            else:
                anim.save(output_path, fps=fps, dpi=100)
            print(f"  Animation saved: {output_path}")
        except Exception as exc:
            print(f"  ⚠  Could not save animation: {exc}")

    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Generate a GIF animation for one or all four boarding strategies.

    Usage
    -----
        python boarding/animate.py --strategy steffen --scenario baseline --seed 12345
        python boarding/animate.py --strategy all --scenario philippine
        python -m boarding.animate --strategy outside_in
    """
    import argparse, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    from boarding import load_config

    parser = argparse.ArgumentParser(
        description="Generate a boarding simulation animation (GIF).",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "back_to_front", "outside_in", "steffen", "all"],
        default="steffen",
        help="Boarding strategy to animate (default: steffen). Use 'all' for all 4.",
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline", "philippine"],
        default="baseline",
        help="Scenario (default: baseline).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed (default: 12345).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second in the output GIF (default: 15).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum number of frames to include (subsampled, default: 200).",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Output directory for GIF files (default: results/).",
    )
    args = parser.parse_args()

    cfg      = load_config(scenario=args.scenario)
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = (
        ["random", "back_to_front", "outside_in", "steffen"]
        if args.strategy == "all"
        else [args.strategy]
    )

    for strat in strategies:
        print(f"Recording {strat} ({args.scenario}, seed={args.seed}) ...", flush=True)
        result  = record_one_run(strat, cfg, seed=args.seed)
        history = result.get("history", [])
        total   = result["total_boarding_time_s"]
        print(f"  {len(history)} ticks, total boarding time = {total} s")

        out_path = str(out_dir / f"anim_{strat}_{args.scenario}.gif")
        animate_boarding(
            history=history,
            cfg=cfg,
            strategy_name=strat,
            output_path=out_path,
            fps=args.fps,
            max_frames=args.max_frames,
        )

    print("Done.")



if __name__ == "__main__":
    main()
