"""
tests/test_two_same_row.py
==========================
Verification tests: two passengers in the same row.

Cases
-----
Case A – C then A (aisle seated before window):
  Passenger 0: seat C (no blockers)  → boards first.
  Passenger 1: seat A (blockers: C then B; B is empty) → only C is seated.
  Expected: **zero** seat interference events for passenger 1 because
  seat B is empty; C is the blocker but the path A←B←aisle has B free.

  Wait – the PRD rule: check whether any seat BETWEEN the passenger's seat
  and the aisle is already occupied.  Seat A must pass B then C.  In this
  case C is already seated.  So passenger 1 (seat A) DOES have 1 blocker (C).

  Correction: The seat-interference count depends on which seats are occupied
  between A and the aisle.  If only C is seated (B is empty), the path is
  A → B(empty) → C(occupied) → aisle.  The blocking set for A is [C, B].
  C is occupied → 1 seat interference event, extra shuffle for 1 blocker.

  ACTUALLY re-read PRD §6: "Check whether any seat between the passenger's
  seat and the aisle is already occupied."  For seat A, the seats between A
  and the aisle are B and C.  B is empty, C is occupied → 1 blocker.

  So:
    Order C-then-A: passenger to A finds C occupied → 1 seat interference.
    Corrected Case A expected: 1 seat interference event.

Case B – A then C (window seated before aisle):
  Passenger 0: seat A boards first → no one in B or C → 0 seat interference.
  Passenger 1: seat C boards second → C is aisle, no one blocks C → 0 interference.
  Expected: **zero** seat interference events total.

NOTE (M1): Marked xfail until engine is implemented in M2/M3.
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
