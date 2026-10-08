"""
Boarding-order strategies.  Each strategy is a function with the signature:

    strategy(passengers, rng, **kwargs) -> list[Passenger]

returning the call order: index 0 boards first.

Strategies
----------
1. random_order      – uniform shuffle
2. back_to_front     – 5 zones (configurable), rear first; random within zone
3. outside_in        – WilMA: windows → middles → aisles; random within wave
4. steffen           – Steffen (2008); six waves interleaved for minimal conflict

All strategies are unit-tested in tests/test_strategies.py to confirm they
return a valid permutation of all passengers with no duplicates.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from boarding.cabin import SEAT_TYPE, SeatType

if TYPE_CHECKING:
    from boarding.passenger import Passenger


# ---------------------------------------------------------------------------
# Strategy 1 – Random
# ---------------------------------------------------------------------------

def random_order(
    passengers: list["Passenger"],
    rng: np.random.Generator,
) -> list["Passenger"]:
    """
    Strategy 1 – Random: every passenger called in a uniformly random order.

    Parameters
    ----------
    passengers : list[Passenger]
        All passengers to board.
    rng : numpy.random.Generator
        Seeded RNG.

    Returns
    -------
    list[Passenger]
        A shuffled copy of *passengers*.
    """
    order = list(passengers)
    rng.shuffle(order)
    return order


# ---------------------------------------------------------------------------
# Strategy 2 – Back-to-Front (zones)
# ---------------------------------------------------------------------------

def back_to_front(
    passengers: list["Passenger"],
    rng: np.random.Generator,
    num_zones: int = 5,
) -> list["Passenger"]:
    """
    Strategy 2 – Back-to-Front (zones).

    Divides the plane into *num_zones* equal-sized zones; zone 1 is the
    rearmost block and is called first.  Passengers within each zone board
    in a random order.

    Zone layout (30 rows, 5 zones of 6 rows each):
      Zone 1 (called first) : rows 25-30  (rear)
      Zone 2                : rows 19-24
      Zone 3                : rows 13-18
      Zone 4                : rows  7-12
      Zone 5 (called last)  : rows  1-6   (front)

    Parameters
    ----------
    passengers : list[Passenger]
        All passengers to board.
    rng : numpy.random.Generator
        Seeded RNG (used for within-zone shuffle).
    num_zones : int
        Number of boarding zones (the specification default 5; configurable 3-6).

    Returns
    -------
    list[Passenger]
        Passengers ordered rear-zone first, random within each zone.
    """
    total_rows    = max(p.row for p in passengers)
    rows_per_zone = math.ceil(total_rows / num_zones)

    result: list[Passenger] = []
    for zone_num in range(1, num_zones + 1):   # zone 1 = rear, called first
        zone_rear  = total_rows - (zone_num - 1) * rows_per_zone
        zone_front = zone_rear - rows_per_zone + 1
        zone_pax   = [p for p in passengers if zone_front <= p.row <= zone_rear]
        zone_pax   = list(zone_pax)
        rng.shuffle(zone_pax)
        result.extend(zone_pax)

    return result


# ---------------------------------------------------------------------------
# Strategy 3 – Outside-In (WilMA)
# ---------------------------------------------------------------------------

def outside_in(
    passengers: list["Passenger"],
    rng: np.random.Generator,
) -> list["Passenger"]:
    """
    Strategy 3 – Outside-In (WilMA: Window – Middle – Aisle).

    Boarding waves (random within each wave):
      Wave 1: all window seats (A and F)
      Wave 2: all middle seats (B and E)
      Wave 3: all aisle seats  (C and D)

    Parameters
    ----------
    passengers : list[Passenger]
        All passengers to board.
    rng : numpy.random.Generator
        Seeded RNG (within-wave shuffle).

    Returns
    -------
    list[Passenger]
        Passengers ordered windows → middles → aisles.
    """
    windows = [p for p in passengers if SEAT_TYPE[p.letter] == SeatType.WINDOW]
    middles = [p for p in passengers if SEAT_TYPE[p.letter] == SeatType.MIDDLE]
    aisles  = [p for p in passengers if SEAT_TYPE[p.letter] == SeatType.AISLE]

    rng.shuffle(windows)
    rng.shuffle(middles)
    rng.shuffle(aisles)

    return windows + middles + aisles


# ---------------------------------------------------------------------------
# Strategy 4 – Steffen (2008)
# ---------------------------------------------------------------------------

def steffen(
    passengers: list["Passenger"],
    rng: np.random.Generator,
) -> list["Passenger"]:
    """
    Strategy 4 – Steffen Method, following Steffen (2008).

    The key idea: within each wave, passengers are spaced at least 2 rows
    apart so that no one has to wait for their neighbour to finish stowing.

    Six waves, all ordered back-to-front within the wave:
      1. Window seats on even rows  (A and F; rows 30, 28, 26, …, 2)
      2. Window seats on odd rows   (rows 29, 27, 25, …, 1)
      3. Middle seats on even rows  (B and E)
      4. Middle seats on odd rows
      5. Aisle seats on even rows   (C and D)
      6. Aisle seats on odd rows

    Within each wave, left-side (A/B/C) and right-side (D/E/F) passengers
    at the same row are placed together (left first, right second) so both
    sides of the same row board simultaneously.

    NOTE: This follows Steffen (2008).  The exact tie-breaking rule for
    same-row passengers should be verified against the original paper before
    use in an academic submission.

    Parameters
    ----------
    passengers : list[Passenger]
        All passengers to board.
    rng : numpy.random.Generator
        Included for interface consistency; Steffen order is deterministic
        (no randomness used).

    Returns
    -------
    list[Passenger]
        Passengers in Steffen boarding order.
    """
    # Seat-letter pairs: (left_letter, right_letter) for each seat class
    wave_definitions = [
        ("A", "F", True),    # wave 1: window, even rows
        ("A", "F", False),   # wave 2: window, odd rows
        ("B", "E", True),    # wave 3: middle, even rows
        ("B", "E", False),   # wave 4: middle, odd rows
        ("C", "D", True),    # wave 5: aisle, even rows
        ("C", "D", False),   # wave 6: aisle, odd rows
    ]

    result: list[Passenger] = []

    for left_letter, right_letter, even_only in wave_definitions:
        # Collect passengers for this wave
        wave_pax = [
            p for p in passengers
            if p.letter in (left_letter, right_letter)
            and (p.row % 2 == 0) == even_only
        ]

        # Sort back-to-front (highest row number first)
        rows_in_wave = sorted({p.row for p in wave_pax}, reverse=True)

        for row in rows_in_wave:
            # Within each row, place left then right (both board together)
            left  = [p for p in wave_pax if p.row == row and p.letter == left_letter]
            right = [p for p in wave_pax if p.row == row and p.letter == right_letter]
            result.extend(left + right)

    return result


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, callable] = {
    "random":        random_order,
    "back_to_front": back_to_front,
    "outside_in":    outside_in,
    "steffen":       steffen,
}
