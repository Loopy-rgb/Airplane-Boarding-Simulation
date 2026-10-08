"""
Flask dashboard for the Airplane Boarding Strategy Simulation.

Usage
-----
    cd boarding_sim
    python web_app.py

Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import pathlib
import sys

# Make sure the boarding package is importable from this file's location.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from boarding import load_config
from boarding.behavior import apply_behavior
from boarding.cabin import Cabin
from boarding.simulation import generate_passengers, run_simulation
from boarding.strategies import STRATEGIES

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates")
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Seat-letter → aisle-y index used in the compact history format.
_SEAT_Y  = {"A": 0, "B": 1, "C": 2, "D": 4, "E": 5, "F": 6}
_AISLE_Y = 3
_S_CODE  = {
    "QUEUED":  0,
    "WALKING": 1,
    "BLOCKED": 2,
    "STOWING": 3,
    "SEATING": 4,
    "SEATED":  5,
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    """Serve the main dashboard page."""
    # List available chart files so the template can render tabs.
    charts = [
        {"name": p.stem, "label": _chart_label(p.stem)}
        for p in sorted(RESULTS_DIR.glob("fig*.png"))
    ]
    return render_template("index.html", charts=charts)


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Run one simulation and return a compact history for animation.

    Request JSON
    ------------
    {
        "strategy":  "steffen" | "random" | "back_to_front" | "outside_in",
        "scenario":  "baseline" | "philippine",
        "seed":      int   (default 12345),
        "overrides": {
            "compliance_rate": float,   # 0.0 – 1.0
            "group_fraction":  float,   # 0.0 – 0.6
        }
    }

    Response JSON
    -------------
    {
        "ok": true,
        "history": [ { "t": tick, "s": [[id, code, x, y], …] }, … ],
        "metrics": { total_time, aisle_events, seat_events, mean_wait, max_wait },
        "n_passengers": 180
    }
    """
    data      = request.get_json(force=True) or {}
    strategy  = data.get("strategy", "steffen")
    scenario  = data.get("scenario", "baseline")
    seed      = int(data.get("seed", 12345))
    overrides = data.get("overrides", {})

    # Validate strategy
    if strategy not in STRATEGIES:
        return jsonify({"ok": False, "error": f"Unknown strategy: {strategy}"}), 400

    # Load + patch config
    cfg = load_config(scenario=scenario)

    # Individual behavior feature flags (e.g. {"group_travel": true, "bayanihan": false})
    behavior_flags = overrides.get("behaviors", {})
    for feature_name, enabled_flag in behavior_flags.items():
        if feature_name in cfg.get("behavior", {}):
            cfg["behavior"][feature_name]["enabled"] = bool(enabled_flag)

    if "compliance_rate" in overrides:
        cfg["behavior"]["non_compliance"]["compliance_rate"] = float(
            overrides["compliance_rate"]
        )
    if "group_fraction" in overrides:
        cfg["behavior"]["group_travel"]["group_fraction"] = float(
            overrides["group_fraction"]
        )

    # Independent RNG streams (one per concern)
    rng_pop    = np.random.default_rng(seed)
    rng_strat  = np.random.default_rng(seed + 1)
    rng_beh    = np.random.default_rng(seed + 2)
    rng_engine = np.random.default_rng(seed + 3)

    passengers = generate_passengers(cfg, rng_pop)

    # Strategy ordering (pass num_zones from config for back_to_front)
    kwargs: dict = {}
    if strategy == "back_to_front":
        kwargs["num_zones"] = cfg.get("strategies", {}).get("back_to_front_zones", 5)
    order = STRATEGIES[strategy](passengers, rng_strat, **kwargs)

    # Behavior modifiers
    order = apply_behavior(order, passengers, rng_beh, cfg)

    # Build per-passenger behavior tag bitmask (after behaviors are assigned).
    # Bit 0 = group member, Bit 1 = late, Bit 2 = non-compliant.
    pax_map = {p.id: p for p in passengers}
    def _tag(p) -> int:
        t = 0
        if p.group_id is not None:
            t |= 1
        if p.is_late:
            t |= 2
        if not p.is_compliant:
            t |= 4
        return t
    passenger_tags = {p.id: _tag(p) for p in passengers}

    # Simulate with history recording
    cabin  = Cabin(cfg["cabin"])
    result = run_simulation(order, cabin, cfg, rng_engine, record_history=True)

    # Subsample history to ≤ 300 frames for the animation
    history  = result.get("history", [])
    n_target = 300
    step     = max(1, len(history) // n_target)
    sampled  = history[::step]
    # Always include the final frame
    if history and history[-1] is not sampled[-1]:
        sampled = sampled + [history[-1]]

    # Convert to compact format: each passenger as [id, state_code, x, y]
    # x = row index 0-29 (or negative for queue)
    # y = lane index 0-6  (A=0 … F=6, aisle=3)
    compact_history = []
    for frame in sampled:
        states_list = []
        q_pos = 0
        for p in frame["states"]:
            code = _S_CODE[p["state"]]
            if p["state"] == "QUEUED":
                x, y = -(q_pos + 1), _AISLE_Y
                q_pos += 1
            elif p["state"] in ("WALKING", "BLOCKED", "STOWING"):
                x = p["aisle_cell"] if p["aisle_cell"] is not None else 0
                y = _AISLE_Y
            elif p["state"] == "SEATING":
                x = p["aisle_cell"] if p["aisle_cell"] is not None else p["row"] - 1
                y = _AISLE_Y
            else:  # SEATED
                x = p["row"] - 1
                y = _SEAT_Y.get(p["letter"], _AISLE_Y)
            tag = passenger_tags.get(p["id"], 0)
            states_list.append([p["id"], code, x, y, tag])
        compact_history.append({"t": frame["tick"], "s": states_list})

    # Summarise which behavior tags are actually active (for legend display).
    active_tags = {
        "group":          any(p.group_id is not None for p in passengers),
        "late":           any(p.is_late for p in passengers),
        "non_compliant":  any(not p.is_compliant for p in passengers),
    }

    return jsonify({
        "ok":             True,
        "history":        compact_history,
        "passenger_tags": passenger_tags,
        "active_tags":    active_tags,
        "metrics": {
            "total_time":   result["total_boarding_time_s"],
            "aisle_events": result["aisle_interference_events"],
            "seat_events":  result["seat_interference_events"],
            "mean_wait":    round(result["mean_wait_time_s"], 1),
            "max_wait":     round(result["max_wait_time_s"], 1),
        },
        "n_passengers":  180,
        "n_frames":      len(compact_history),
        "total_ticks":   result["total_boarding_time_s"],
    })


@app.route("/api/chart/<name>")
def api_chart(name: str):
    """Serve a pre-generated chart PNG."""
    safe = name.replace("/", "").replace("..", "").strip()
    path = RESULTS_DIR / f"{safe}.png"
    if path.exists():
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "chart not found"}), 404


@app.route("/api/gif/<name>")
def api_gif(name: str):
    """Serve a pre-generated boarding animation GIF."""
    safe = name.replace("/", "").replace("..", "").strip()
    path = RESULTS_DIR / f"{safe}.gif"
    if path.exists():
        return send_file(path, mimetype="image/gif")
    return jsonify({"error": "gif not found"}), 404


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chart_label(stem: str) -> str:
    """Convert file stem like 'fig3_compliance_line' → 'Fig 3: Compliance Line'."""
    stem = stem.replace("_", " ")
    # Capitalise first letter of each word
    return stem.title().replace("Fig ", "Fig ")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("  ✈  Airplane Boarding Simulation Dashboard")
    print("  ─────────────────────────────────────────")
    print("  Open your browser at:  http://localhost:5050")
    print()
    app.run(debug=False, port=5050, use_reloader=False)
