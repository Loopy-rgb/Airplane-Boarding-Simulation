"""
tests/test_strategies.py
========================
Unit tests for boarding strategy functions.

Each strategy must return a valid permutation of all 180 passengers:
  - Exactly 180 elements.
  - No duplicates (by passenger id).
  - All passenger ids are present.

NOTE (M1): Strategy stubs return [] for unimplemented strategies.
Tests for unimplemented strategies are marked xfail.
random_order is implemented in M1 and its test should pass now.
"""

import pytest
import numpy as np

from boarding import load_config
from boarding.passenger import Passenger
from boarding.strategies import random_order, back_to_front, outside_in, steffen


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_passengers() -> list[Passenger]:
    """Create the full 180-passenger roster."""
    cfg = load_config(scenario="baseline")
    rows = cfg["cabin"]["rows"]
    letters = cfg["cabin"]["seat_letters"]
    pax = []
    pid = 0
    for row in range(1, rows + 1):
        for letter in letters:
            pax.append(Passenger(id=pid, row=row, letter=letter))
            pid += 1
    return pax


def assert_valid_permutation(result: list, passengers: list, strategy_name: str):
    """Assert result is a permutation of all passengers with no duplicates."""
    assert len(result) == len(passengers), (
        f"{strategy_name}: expected {len(passengers)} passengers, "
        f"got {len(result)}"
    )
    ids = [p.id for p in result]
    assert len(ids) == len(set(ids)), (
        f"{strategy_name}: duplicate passenger ids in result"
    )
    expected_ids = {p.id for p in passengers}
    assert set(ids) == expected_ids, (
        f"{strategy_name}: passenger ids do not match expected set"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_random_order_is_valid_permutation():
    """random_order must return all 180 passengers with no duplicates."""
    passengers = make_passengers()
    rng = np.random.default_rng(seed=99)
    result = random_order(passengers, rng)
    assert_valid_permutation(result, passengers, "random_order")


def test_random_order_is_actually_random():
    """Two different seeds should produce different orderings (almost surely)."""
    passengers = make_passengers()
    rng1 = np.random.default_rng(seed=1)
    rng2 = np.random.default_rng(seed=2)
    order1 = [p.id for p in random_order(passengers, rng1)]
    order2 = [p.id for p in random_order(passengers, rng2)]
    assert order1 != order2, "random_order produced identical results for different seeds"


def test_same_seed_gives_same_random_order():
    """Same seed must give identical ordering (reproducibility)."""
    passengers = make_passengers()
    rng_a = np.random.default_rng(seed=42)
    rng_b = np.random.default_rng(seed=42)
    order_a = [p.id for p in random_order(passengers, rng_a)]
    order_b = [p.id for p in random_order(passengers, rng_b)]
    assert order_a == order_b, "random_order is not reproducible with the same seed"


def test_back_to_front_is_valid_permutation():
    passengers = make_passengers()
    rng = np.random.default_rng(seed=99)
    result = back_to_front(passengers, rng)
    assert_valid_permutation(result, passengers, "back_to_front")


def test_back_to_front_rear_before_front():
    """Last passenger in the list should have a lower row number than the first."""
    passengers = make_passengers()
    rng = np.random.default_rng(seed=99)
    result = back_to_front(passengers, rng)
    assert len(result) == 180
    # Zone 1 (rows 25-30) should appear before zone 5 (rows 1-6).
    zone1_idx = [i for i, p in enumerate(result) if p.row >= 25]
    zone5_idx = [i for i, p in enumerate(result) if p.row <= 6]
    assert max(zone1_idx) < min(zone5_idx), (
        "back_to_front: zone 1 (rear) should appear before zone 5 (front)"
    )


def test_outside_in_is_valid_permutation():
    passengers = make_passengers()
    rng = np.random.default_rng(seed=99)
    result = outside_in(passengers, rng)
    assert_valid_permutation(result, passengers, "outside_in")


def test_outside_in_window_before_aisle():
    """All window-seat passengers should appear before any aisle-seat passenger."""
    passengers = make_passengers()
    rng = np.random.default_rng(seed=99)
    result = outside_in(passengers, rng)
    window_letters = {"A", "F"}
    aisle_letters  = {"C", "D"}
    last_window = max(i for i, p in enumerate(result) if p.letter in window_letters)
    first_aisle = min(i for i, p in enumerate(result) if p.letter in aisle_letters)
    assert last_window < first_aisle, (
        "outside_in: all window seats must board before any aisle seat"
    )


def test_steffen_is_valid_permutation():
    passengers = make_passengers()
    rng = np.random.default_rng(seed=99)
    result = steffen(passengers, rng)
    assert_valid_permutation(result, passengers, "steffen")
