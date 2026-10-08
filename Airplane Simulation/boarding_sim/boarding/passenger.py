"""
boarding/passenger.py
=====================
Defines the ``Passenger`` dataclass and the ``PassengerState`` enum.

State machine
-------------
    QUEUED → WALKING → BLOCKED → STOWING → SEATING → SEATED

Transitions
~~~~~~~~~~~
QUEUED
    Passenger is in the pre-door queue, waiting for their turn to board.
WALKING
    Passenger is in the aisle and advancing toward their assigned row.
BLOCKED
    Passenger is in the aisle but the next cell is occupied; they wait.
STOWING
    Passenger has reached their row and is loading hand-carry into the
    overhead bin.  They occupy the aisle cell and block everyone behind.
SEATING
    Passenger is sliding into their seat.  If blockers are already seated,
    they add shuffle time first; then they sit down.
SEATED
    Passenger is seated; they no longer occupy the aisle cell.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------

class PassengerState(enum.Enum):
    """Ordered states a passenger passes through during boarding."""
    QUEUED   = "QUEUED"
    WALKING  = "WALKING"
    BLOCKED  = "BLOCKED"
    STOWING  = "STOWING"
    SEATING  = "SEATING"
    SEATED   = "SEATED"


# ---------------------------------------------------------------------------
# Passenger dataclass
# ---------------------------------------------------------------------------

@dataclass
class Passenger:
    """
    Represents a single boarding passenger (agent).

    All time fields are in simulation ticks (seconds when dt = 1 s).

    Parameters / fields set at creation
    ------------------------------------
    id : int
        Unique passenger identifier.
    row : int
        Assigned seat row (1-indexed; 1 = front, 30 = rear).
    letter : str
        Assigned seat letter (one of A, B, C, D, E, F).
    group_id : int or None
        Group the passenger belongs to, or ``None`` if travelling alone.
    num_bags : int
        Number of hand-carry items (0, 1, or 2+).
    walk_speed_mps : float
        Walking speed in metres per second.
    stow_time_s : float
        Time in seconds to load hand-carry into the overhead bin.
    seat_search_delay_s : float
        Extra seconds spent searching for the row before stowing (0 for most).
    is_late : bool
        If True the passenger was moved to the end of the boarding queue.
    is_compliant : bool
        If True the passenger boards in the strategy-assigned order;
        non-compliant passengers jump to a random early position.

    Fields updated at runtime
    -------------------------
    entry_time : int or None
        Tick at which the passenger entered aisle cell 0.
    state : PassengerState
        Current boarding state.
    aisle_cell : int or None
        0-indexed aisle cell where the passenger is currently standing,
        or ``None`` if not yet in the aisle or already seated.
    aisle_progress_m : float
        Accumulated metres walked since the last cell advance.  Used to
        determine when the passenger moves to the next cell.
    stow_ticks_remaining : int
        Countdown (in ticks) until stowing is complete.
    seating_ticks_remaining : int
        Countdown (in ticks) until seating / shuffling is complete.
    wait_ticks : int
        Total ticks spent in the BLOCKED state.
    aisle_interference_events : int
        Number of separate blocking episodes in the aisle (incremented
        once per continuous blocked episode, not once per blocked tick).
    seat_interference_events : int
        Number of blocking passengers that caused seat-row shuffles.
    seated_time : int or None
        Tick at which the passenger reached the SEATED state.
    _was_blocked_last_tick : bool
        Internal flag used to count discrete aisle-interference episodes
        rather than counting every blocked tick.
    """

    # ---- Assigned seat -------------------------------------------------------
    id: int
    row: int
    letter: str

    # ---- Group and behaviour attributes (set before the run starts) ----------
    group_id: Optional[int] = None
    num_bags: int = 1
    walk_speed_mps: float = 0.8
    stow_time_s: float = 9.0
    seat_search_delay_s: float = 0.0
    is_late: bool = False
    is_compliant: bool = True

    # ---- Runtime state (updated each tick by the engine) ---------------------
    entry_time: Optional[int] = None
    state: PassengerState = field(default=PassengerState.QUEUED)
    aisle_cell: Optional[int] = None

    # Fractional progress toward the next aisle cell (metres walked so far).
    aisle_progress_m: float = 0.0

    # Countdown timers
    stow_ticks_remaining: int = 0
    seating_ticks_remaining: int = 0

    # ---- Metric counters -----------------------------------------------------
    wait_ticks: int = 0
    aisle_interference_events: int = 0
    seat_interference_events: int = 0
    seated_time: Optional[int] = None

    # Internal: tracks whether the passenger was blocked on the *previous* tick
    # so we count one interference event per episode, not per tick.
    _was_blocked_last_tick: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_seated(self) -> bool:
        """True when the passenger has fully reached the SEATED state."""
        return self.state == PassengerState.SEATED

    @property
    def in_aisle(self) -> bool:
        """True when the passenger is somewhere in the aisle (not seated, not queued)."""
        return self.aisle_cell is not None

    @property
    def total_stow_ticks(self) -> int:
        """
        Total ticks allocated for stowing, rounded to nearest int.

        This is ``stow_time_s + seat_search_delay_s`` (since dt = 1 s).
        Stored here for reference/debugging; the engine sets
        ``stow_ticks_remaining`` directly.
        """
        return round(self.stow_time_s + self.seat_search_delay_s)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Passenger(id={self.id}, seat={self.row}{self.letter}, "
            f"state={self.state.value}, cell={self.aisle_cell})"
        )
