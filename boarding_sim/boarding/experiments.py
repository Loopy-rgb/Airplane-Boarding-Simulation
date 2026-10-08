"""
Runs replicated experiments using Common Random Numbers (CRN).

Common Random Numbers design
------------------------------
For each replication *rep* (0-based), a single SeedSequence is spawned from
``base_seed + rep``.  Its children supply independent RNG streams:

  child[0]  → population generator  (bags, walk speeds, stow times)
  child[1]  → strategy ordering for "random"
  child[2]  → strategy ordering for "back_to_front"
  child[3]  → strategy ordering for "outside_in"
  child[4]  → strategy ordering for "steffen"
  child[5]  → behavior modifier     (group assignment, compliance, etc.)
  child[6]  → simulation engine     (shuffle times during seating)

All four strategies therefore board the *same* passenger population
(same attributes, same assignment of bags / walk speeds / stow times),
making paired comparisons valid.

Public interface
----------------
    run_experiment(strategies, scenario, cfg, experiment_name,
                   sweep_params=None) -> pd.DataFrame
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

# Fixed ordering so child seeds are stable regardless of dict insertion order.
_STRAT_CHILD_IDX: dict[str, int] = {
    "random":        1,
    "back_to_front": 2,
    "outside_in":    3,
    "steffen":       4,
}
_BEHAVIOR_CHILD_IDX  = 5
_ENGINE_CHILD_IDX    = 6
_N_CHILDREN          = 7    # total children per replication


def run_experiment(
    strategies: dict[str, Callable],
    scenario: str,
    cfg: dict,
    experiment_name: str = "experiment",
    sweep_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run *replications* replications of every strategy under *scenario*.

    Parameters
    ----------
    strategies : dict[str, Callable]
        Mapping of strategy name → ordering function.
    scenario : str
        Scenario name (``"baseline"`` or ``"philippine"``); used only as a
        label in the output – the caller is responsible for passing a *cfg*
        with the correct scenario already merged.
    cfg : dict
        Full simulation config (scenario already merged if applicable).
    experiment_name : str
        Label written into every result row.
    sweep_params : dict, optional
        Extra parameter key-value pairs to include verbatim in every row
        (e.g. ``{"compliance_rate": 0.7}``).

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with one row per (replication × strategy) run.
        Columns: experiment, strategy, scenario, seed, rep_id,
                 total_boarding_time_s, aisle_interference_events,
                 seat_interference_events, mean_wait_time_s,
                 max_wait_time_s, [+ any sweep_params keys].
    """
    from boarding.cabin import Cabin
    from boarding.simulation import generate_passengers, run_simulation
    from boarding.behavior import apply_behavior

    exp_cfg   = cfg["experiment"]
    base_seed = exp_cfg["base_seed"]
    n_reps    = exp_cfg["replications"]

    extra_cols = sweep_params or {}

    # Try to import tqdm for progress bars; fall back to a plain range.
    try:
        from tqdm import tqdm
        rep_iter = tqdm(range(n_reps), desc=f"{experiment_name}", unit="rep")
    except ImportError:
        rep_iter = range(n_reps)

    all_rows: list[dict] = []

    for rep in rep_iter:
        seed = base_seed + rep

        # ------------------------------------------------------------------
        # Spawn independent RNG streams (one SeedSequence per replication)
        # ------------------------------------------------------------------
        parent_seq = np.random.SeedSequence(seed)
        children   = parent_seq.spawn(_N_CHILDREN)

        pop_rng     = np.random.default_rng(children[0])
        beh_rng     = np.random.default_rng(children[_BEHAVIOR_CHILD_IDX])
        engine_rng  = np.random.default_rng(children[_ENGINE_CHILD_IDX])

        # ------------------------------------------------------------------
        # Generate the passenger population (shared across all strategies)
        # ------------------------------------------------------------------
        passengers = generate_passengers(cfg, pop_rng)

        # Apply attribute-modifying behaviour features once per replication
        # so all strategies board passengers with the same physical attributes.
        # Queue-order modifications are applied per strategy below.
        # (For baseline scenario all features are off → no-op.)
        # A temporary order is used; the per-strategy calls will reorder.
        _apply_attribute_behaviors(passengers, beh_rng, cfg)

        # ------------------------------------------------------------------
        # Run each strategy on the same population
        # ------------------------------------------------------------------
        for name, strategy_fn in strategies.items():
            child_idx   = _STRAT_CHILD_IDX.get(name, 4)  # default to steffen slot
            strat_rng   = np.random.default_rng(children[child_idx])

            # 1. Strategy ordering
            #    Pass num_zones from config to back_to_front so the configurable
            #    zone count in config.yaml is actually honoured.
            strat_kwargs: dict = {}
            if name == "back_to_front":
                strat_kwargs["num_zones"] = cfg.get("strategies", {}).get(
                    "back_to_front_zones", 5
                )
            order = strategy_fn(passengers, strat_rng, **strat_kwargs)

            # 2. Apply queue-order behaviour (non-compliance, late, groups, …)
            order = apply_behavior(order, passengers, strat_rng, cfg)

            # 3. Run simulation (fresh cabin; engine RNG shared across strats
            #    to keep shuffle draws comparable – valid because the draws
            #    occur at different ticks for each strategy).
            cabin  = Cabin(cfg["cabin"])
            result = run_simulation(
                order=order, cabin=cabin, cfg=cfg, rng=engine_rng
            )

            # 4. Collect row
            row: dict = {
                "experiment":                experiment_name,
                "strategy":                  name,
                "scenario":                  scenario,
                "seed":                      seed,
                "rep_id":                    rep,
                "total_boarding_time_s":     result["total_boarding_time_s"],
                "aisle_interference_events": result["aisle_interference_events"],
                "seat_interference_events":  result["seat_interference_events"],
                "mean_wait_time_s":          result["mean_wait_time_s"],
                "max_wait_time_s":           result["max_wait_time_s"],
            }
            row.update(extra_cols)
            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Private helper: attribute-only behaviour pass
# ---------------------------------------------------------------------------

def _apply_attribute_behaviors(
    passengers: list,
    rng: np.random.Generator,
    cfg: dict,
) -> None:
    """
    Apply behaviour features that modify *passenger attributes* (bags,
    stow time, seat-search delay, group_id) in-place before any strategy runs.

    This is called once per replication so all strategies see the same
    physical passenger attributes (CRN requirement).

    Features applied here:
      - heavy_handcarry   : reassign num_bags / stow_time_s
      - seat_search_delay : assign seat_search_delay_s
      - group_travel      : assign group_id (queue ordering happens per-strategy)
    """
    from boarding.behavior import (
        _apply_heavy_handcarry,
        _apply_seat_search_delay,
        _assign_groups,
    )

    beh = cfg.get("behavior", {})

    if beh.get("heavy_handcarry", {}).get("enabled", False):
        _apply_heavy_handcarry(
            passengers, rng,
            beh["heavy_handcarry"],
            cfg["passenger"]["stow_time_s"],
        )

    if beh.get("seat_search_delay", {}).get("enabled", False):
        _apply_seat_search_delay(passengers, rng, beh["seat_search_delay"])

    if beh.get("group_travel", {}).get("enabled", False):
        _assign_groups(passengers, rng, beh["group_travel"])

