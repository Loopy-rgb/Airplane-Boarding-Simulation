"""
tests/test_single_passenger.py
===============================
Verification tests: hand-computable single-passenger scenarios (PRD §13).

Engine implemented in Milestone 2; all tests are live.
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _single_run(row: int, letter: str, num_bags: int, stow_time: float,
                walk_speed: float = 0.8):
    """Run a one-passenger simulation and return (result_dict, cfg)."""
    from boarding import load_config, Cabin
    from boarding.passenger import Passenger
    from boarding.simulation import run_simulation

    cfg   = load_config(scenario="baseline")
    cabin = Cabin(cfg["cabin"])
    pax   = Passenger(
        id=0, row=row, letter=letter,
        num_bags=num_bags, walk_speed_mps=walk_speed,
        stow_time_s=stow_time, seat_search_delay_s=0.0,
    )
    rng = np.random.default_rng(seed=0)
    return run_simulation(order=[pax], cabin=cabin, cfg=cfg, rng=rng), cfg


# ---------------------------------------------------------------------------
# Test 1 – Row 1, seat C (PRD §13: exact hand-computed time)
# ---------------------------------------------------------------------------

def test_single_passenger_row1_seat_c():
    """
    Row 1 = cell 0 = entry cell, so walk time = 0.
    Seat C is an aisle seat (no blockers).

    Hand-computed:
      0 (walk) + 9 (stow) + 2 (base seat) = 11 ticks.
    """
    stow_time = 9      # fix to mean for a deterministic result
    result, cfg = _single_run(row=1, letter="C", num_bags=1, stow_time=stow_time)

    base_seat_time = cfg["simulation"]["base_seat_time_s"]
    expected = stow_time + base_seat_time   # 11

    assert result["total_boarding_time_s"] == pytest.approx(expected, abs=1), (
        f"Row-1/C: expected ~{expected} s, got {result['total_boarding_time_s']} s"
    )


# ---------------------------------------------------------------------------
# Test 2 – Row 30, seat C (PRD §13: walk-time verification)
# ---------------------------------------------------------------------------

def test_single_passenger_row30_walk_time():
    """
    Row 30 = cell 29.  The passenger must traverse 29 cells from the door.

    PRD §13: 'walking time is approximately 29 * pitch / speed'.

    Hand-computation for walk_speed=0.8 m/s, row_pitch=0.79 m:
      Each tick progress += 0.8.  Since 0.8 > 0.79, the passenger advances
      one cell every tick.  Entering at cell 0 on tick 0, they reach cell 29
      at the end of tick 28.  Walk time = 28 ticks.

    Theoretical: 29 × 0.79 / 0.8 = 28.6 ticks → ≈ 28–29 ticks.

    With num_bags=0 (stow_time=0) and seat C (no blockers):
      total = walk_time + 0 (stow) + 2 (seat) = 30 ticks.
    """
    walk_speed = 0.8
    result, cfg = _single_run(row=30, letter="C", num_bags=0,
                               stow_time=0.0, walk_speed=walk_speed)

    row_pitch      = cfg["cabin"]["row_pitch_m"]
    base_seat_time = cfg["simulation"]["base_seat_time_s"]

    # Separate walk time from total (stow=0, no blockers)
    walk_time   = result["total_boarding_time_s"] - base_seat_time
    theoretical = 29 * row_pitch / walk_speed     # 28.6 ticks

    # Allow ±2 ticks of discretisation error (1-second tick resolution)
    assert abs(walk_time - theoretical) <= 2, (
        f"Walk time {walk_time} s differs from theoretical {theoretical:.2f} s "
        f"by more than 2 ticks."
    )


# ---------------------------------------------------------------------------
# Test 3 – Row 1, seat A after C (seat-interference metric)
# ---------------------------------------------------------------------------

def test_single_passenger_seat_A_blocked_by_seated_C():
    """
    When C is already seated and A boards second in the same row, A must
    pass C (occupied) → exactly 1 seat_interference_event for A.
    """
    from boarding import load_config, Cabin
    from boarding.passenger import Passenger
    from boarding.simulation import run_simulation

    cfg   = load_config(scenario="baseline")
    cabin = Cabin(cfg["cabin"])

    pax_c = Passenger(id=0, row=1, letter="C", num_bags=0, stow_time_s=0.0,
                      walk_speed_mps=0.8)
    pax_a = Passenger(id=1, row=1, letter="A", num_bags=0, stow_time_s=0.0,
                      walk_speed_mps=0.8)

    rng = np.random.default_rng(seed=0)
    result = run_simulation(order=[pax_c, pax_a], cabin=cabin, cfg=cfg, rng=rng)

    pax_a_out = next(p for p in result["passengers"] if p.id == 1)
    assert pax_a_out.seat_interference_events == 1, (
        f"Expected 1 seat interference (C blocks A), "
        f"got {pax_a_out.seat_interference_events}"
    )
    assert result["seat_interference_events"] == 1
