"""
Verification tests: two passengers in the same row.

Seat A is the window, B the middle, C the aisle. To reach A a passenger
must pass B and C, so a seat interference event is counted for each of
those seats that is already occupied.

Case A - C boards before A:
  Passenger 0 takes seat C and is seated first.
  Passenger 1 takes seat A. B is empty, C is occupied, so C must stand up.
  Expected: 1 seat interference event.

Case B - A boards before C:
  Passenger 0 takes seat A while B and C are empty.
  Passenger 1 takes seat C, which nothing blocks.
  Expected: 0 seat interference events.
"""

import pytest


def _make_two_pax_run(row: int, first_letter: str, second_letter: str):
    """Helper: run a two-passenger simulation and return the result dict."""
    from boarding import load_config, Cabin
    from boarding.passenger import Passenger
    from boarding.simulation import run_simulation
    import numpy as np

    cfg = load_config(scenario="baseline")
    cabin = Cabin(cfg["cabin"])

    pax0 = Passenger(
        id=0, row=row, letter=first_letter,
        num_bags=0, stow_time_s=0.0, walk_speed_mps=0.8,
    )
    pax1 = Passenger(
        id=1, row=row, letter=second_letter,
        num_bags=0, stow_time_s=0.0, walk_speed_mps=0.8,
    )

    rng = np.random.default_rng(seed=0)
    return run_simulation(order=[pax0, pax1], cabin=cabin, cfg=cfg, rng=rng)


def test_order_C_then_A_has_one_seat_interference():
    """
    C boards first (seated), then A tries to board.
    Seat C is occupied and lies between A and the aisle → 1 seat interference.
    """
    result = _make_two_pax_run(row=15, first_letter="C", second_letter="A")
    assert result["seat_interference_events"] == 1, (
        "Expected 1 seat interference when A boards after C is seated."
    )


def test_order_A_then_C_has_zero_seat_interference():
    """
    A boards first (window), then C boards (aisle).
    C is an aisle seat → no seats between C and the aisle → 0 interference.
    """
    result = _make_two_pax_run(row=15, first_letter="A", second_letter="C")
    assert result["seat_interference_events"] == 0, (
        "Expected 0 seat interference when window boards before aisle."
    )
