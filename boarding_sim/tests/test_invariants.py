"""
tests/test_invariants.py
========================
Tests that verify the four simulation invariants (PRD §6) hold across
50 random full runs.

Invariants
----------
1. No two passengers share an aisle cell.
2. A passenger never moves backward.
3. Each seat holds at most one passenger.
4. A passenger is seated only in their assigned seat.

Also tests:
- Same seed twice gives identical results.
- Aisle interference is counted once per episode (not per tick).

All tests are live as of Milestone 2 (engine implemented).
"""

import pytest
import numpy as np


def _full_run(seed: int, strategy_name: str = "random"):
    """Run a full simulation with the given seed and strategy."""
    from boarding import load_config, Cabin
    from boarding.simulation import generate_passengers, run_simulation
    from boarding.strategies import STRATEGIES

    cfg = load_config(scenario="baseline")
    cabin = Cabin(cfg["cabin"])

    rng = np.random.default_rng(seed=seed)
    passengers = generate_passengers(cfg, rng)

    strategy_fn = STRATEGIES[strategy_name]
    order = strategy_fn(passengers, rng)

    return run_simulation(order=order, cabin=cabin, cfg=cfg, rng=rng, test_mode=True)


def test_invariants_hold_across_50_runs():
    """
    Run 50 random-seed simulations in test_mode=True (which enables internal
    invariant assertions) and confirm no AssertionError is raised.
    """
    for i in range(50):
        _full_run(seed=1000 + i, strategy_name="random")


def test_same_seed_identical_results():
    """Running the same seed twice must produce bit-identical total boarding time."""
    result_a = _full_run(seed=42, strategy_name="random")
    result_b = _full_run(seed=42, strategy_name="random")
    assert result_a["total_boarding_time_s"] == result_b["total_boarding_time_s"], (
        "Simulation is not reproducible: same seed gave different total boarding times."
    )


def test_aisle_interference_counted_once_per_episode():
    """
    A passenger blocked for multiple *consecutive* ticks should generate
    exactly one aisle interference event for that contiguous episode.

    Scenario: a slow passenger (row 3) stows for 60 s, blocking a fast
    passenger (row 29) directly behind them.  The fast passenger catches up
    and is stuck in one continuous blocked episode until the stow finishes.

    The slow passenger uses walk_speed=0.8 (same tick rate as fast) to
    avoid the stop-and-go pattern; what creates the long block is the 60-s
    stow at row 3.  The fast passenger reaches row 2 (one cell behind row 3)
    and waits there until the slow one seats and frees the aisle.

    Expected: exactly 1 aisle interference event for the fast passenger.
    """
    from boarding import load_config, Cabin
    from boarding.passenger import Passenger
    from boarding.simulation import run_simulation
    import numpy as np

    cfg  = load_config(scenario="baseline")
    cabin = Cabin(cfg["cabin"])

    # Both walk at identical speed so the fast one advances at the same rate.
    # The slow one stows for 60 ticks at row 3, creating a long single block.
    slow = Passenger(id=0, row=3,  letter="C", num_bags=2,
                     stow_time_s=60.0, walk_speed_mps=0.8)
    fast = Passenger(id=1, row=29, letter="C", num_bags=0,
                     stow_time_s=0.0,  walk_speed_mps=0.8)

    rng = np.random.default_rng(seed=0)
    result = run_simulation(order=[slow, fast], cabin=cabin, cfg=cfg, rng=rng)

    # The fast passenger is blocked in exactly one continuous episode
    # (it gets stuck behind the slow stower and waits until it clears).
    fast_pax = next(p for p in result["passengers"] if p.id == 1)
    assert fast_pax.aisle_interference_events == 1, (
        f"Expected 1 aisle interference episode for the fast passenger, "
        f"got {fast_pax.aisle_interference_events}."
    )
