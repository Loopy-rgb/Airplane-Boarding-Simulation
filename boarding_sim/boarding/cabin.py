"""
Defines the physical layout of the Airbus A320 cabin used in the
simulation.

Key concepts
------------
- The cabin has ``rows`` rows (1-indexed, row 1 = front, row 30 = rear).
- The aisle is modelled as a 1-D array of cells indexed 0 … rows-1.
  Cell 0 is the front (door side), cell rows-1 is the rear.
- Row *r* occupies aisle cell *r - 1*.
- Seats are labelled A B C | D E F.
  Left block: A (window) – B (middle) – C (aisle-side).
  Right block: D (aisle-side) – E (middle) – F (window).
- Blocking rule for seating:
    Left block : C blocks B blocks A  (must pass C, then B, to reach A)
    Right block: D blocks E blocks F  (must pass D, then E, to reach F)
"""

from __future__ import annotations

import enum
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Seat type enumeration
# ---------------------------------------------------------------------------

class SeatType(enum.Enum):
    """Categorises every seat as window, middle, or aisle."""
    WINDOW = "window"
    MIDDLE = "middle"
    AISLE  = "aisle"


# ---------------------------------------------------------------------------
# Seat-type and blocking maps (fixed for an A320-style 3+3 layout)
# ---------------------------------------------------------------------------

#: Maps each seat letter to its SeatType.
SEAT_TYPE: Dict[str, SeatType] = {
    "A": SeatType.WINDOW,
    "B": SeatType.MIDDLE,
    "C": SeatType.AISLE,
    "D": SeatType.AISLE,
    "E": SeatType.MIDDLE,
    "F": SeatType.WINDOW,
}

#: For each seat letter, lists the seats that must be *empty* for the
#: passenger to reach their seat without a shuffle.
#: Example: to reach seat A the passenger must pass seat C then seat B.
BLOCKING_SEATS: Dict[str, list[str]] = {
    "A": ["C", "B"],  # C and B block A
    "B": ["C"],       # C blocks B
    "C": [],          # aisle side – no one to pass
    "D": [],          # aisle side – no one to pass
    "E": ["D"],       # D blocks E
    "F": ["D", "E"],  # D and E block F
}


# ---------------------------------------------------------------------------
# Cabin class
# ---------------------------------------------------------------------------

class Cabin:
    """
    Represents the static cabin layout and tracks which seats are occupied.

    Parameters
    ----------
    cfg : dict
        The ``cabin`` sub-dict from ``config.yaml``.

    Attributes
    ----------
    rows : int
        Number of rows (e.g. 30 for an A320).
    seat_letters : list[str]
        Ordered list of seat letters, e.g. ``["A","B","C","D","E","F"]``.
    row_pitch_m : float
        Distance between rows in metres.
    num_seats : int
        Total seat count (rows × seats-per-row).
    occupied : dict[int, dict[str, Optional[int]]]
        ``occupied[row][letter]`` holds the passenger ``id`` of the seated
        passenger, or ``None`` if the seat is empty.
        Rows are 1-indexed (1 … rows).
    aisle_cell : list[Optional[int]]
        ``aisle_cell[cell]`` holds the passenger ``id`` currently standing in
        that aisle cell, or ``None`` if the cell is empty.
        Cells are 0-indexed (0 = front/door, rows-1 = rear).
    """

    def __init__(self, cfg: dict) -> None:
        self.rows: int = cfg["rows"]
        self.seat_letters: list[str] = cfg["seat_letters"]
        self.row_pitch_m: float = cfg["row_pitch_m"]

        # Derived
        self.num_seats: int = self.rows * len(self.seat_letters)

        # Seat occupancy map: occupied[row_1indexed][letter] = passenger_id or None
        self.occupied: Dict[int, Dict[str, Optional[int]]] = {
            row: {letter: None for letter in self.seat_letters}
            for row in range(1, self.rows + 1)
        }

        # Aisle occupancy array: aisle_cell[cell_0indexed] = passenger_id or None
        self.aisle_cell: list[Optional[int]] = [None] * self.rows

    # ------------------------------------------------------------------
    # Aisle helpers
    # ------------------------------------------------------------------

    def row_to_cell(self, row: int) -> int:
        """
        Convert a 1-indexed row number to a 0-indexed aisle cell index.

        Row 1 → cell 0, Row 30 → cell 29.
        """
        return row - 1

    def cell_to_row(self, cell: int) -> int:
        """
        Convert a 0-indexed aisle cell to a 1-indexed row number.

        Cell 0 → row 1, cell 29 → row 30.
        """
        return cell + 1

    def is_aisle_cell_free(self, cell: int) -> bool:
        """Return True if the aisle cell at *cell* is unoccupied."""
        return self.aisle_cell[cell] is None

    def enter_aisle(self, passenger_id: int, cell: int) -> None:
        """Place *passenger_id* in aisle *cell*; raises if already occupied."""
        if self.aisle_cell[cell] is not None:
            raise ValueError(
                f"Aisle cell {cell} already occupied by "
                f"passenger {self.aisle_cell[cell]}; "
                f"cannot place passenger {passenger_id}."
            )
        self.aisle_cell[cell] = passenger_id

    def leave_aisle(self, cell: int) -> None:
        """Mark aisle *cell* as empty."""
        self.aisle_cell[cell] = None

    def move_aisle(self, passenger_id: int, from_cell: int, to_cell: int) -> None:
        """
        Move *passenger_id* from *from_cell* to *to_cell*.

        Validates both that the passenger is in *from_cell* and that
        *to_cell* is free.
        """
        if self.aisle_cell[from_cell] != passenger_id:
            raise ValueError(
                f"Passenger {passenger_id} is not in aisle cell {from_cell}."
            )
        if self.aisle_cell[to_cell] is not None:
            raise ValueError(
                f"Aisle cell {to_cell} is already occupied; "
                f"cannot move passenger {passenger_id} there."
            )
        self.aisle_cell[from_cell] = None
        self.aisle_cell[to_cell] = passenger_id

    # ------------------------------------------------------------------
    # Seat helpers
    # ------------------------------------------------------------------

    def seat_type(self, letter: str) -> SeatType:
        """Return the SeatType for *letter*."""
        return SEAT_TYPE[letter]

    def blocking_seats_for(self, letter: str) -> list[str]:
        """
        Return the ordered list of seat letters that block access to *letter*.

        For seat A this is ``["C", "B"]``; for seat C it is ``[]``.
        """
        return BLOCKING_SEATS[letter]

    def is_seat_free(self, row: int, letter: str) -> bool:
        """Return True if seat (*row*, *letter*) is unoccupied."""
        return self.occupied[row][letter] is None

    def seat_passenger(self, passenger_id: int, row: int, letter: str) -> None:
        """Mark seat (*row*, *letter*) as occupied by *passenger_id*."""
        if self.occupied[row][letter] is not None:
            raise ValueError(
                f"Seat {row}{letter} already occupied by "
                f"passenger {self.occupied[row][letter]}."
            )
        self.occupied[row][letter] = passenger_id

    def count_blockers(self, row: int, letter: str) -> int:
        """
        Return the number of already-seated passengers that block access to
        seat (*row*, *letter*).

        For example, if passenger wants seat A and seats B and C are both
        occupied, this returns 2.
        """
        blockers = self.blocking_seats_for(letter)
        return sum(1 for bl in blockers if self.occupied[row][bl] is not None)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def all_seated_count(self) -> int:
        """Return the total number of seated passengers."""
        return sum(
            1
            for row_dict in self.occupied.values()
            for pid in row_dict.values()
            if pid is not None
        )

    def __repr__(self) -> str:  # pragma: no cover
        seated = self.all_seated_count()
        return (
            f"Cabin(rows={self.rows}, seats={self.num_seats}, "
            f"seated={seated})"
        )
