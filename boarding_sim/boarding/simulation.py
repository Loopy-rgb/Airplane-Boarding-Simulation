"""
Core time-step simulation engine.

Rules enforced
-----------------------
- dt = 1 s ticks.
- Entry: one passenger every entry_interval s, if cell 0 is free.
- Walking: accumulated-distance model; progress += walk_speed * dt each tick;
  advance when progress >= row_pitch.  At most one cell advance per tick.
- Blocking: BLOCKED state when next cell is occupied; wait_ticks counted;
  one aisle_interference_event per contiguous blocked episode.
- Stowing: passenger occupies aisle cell for stow_time + seat_search_delay
  ticks; everyone behind is blocked.
- Seating: count already-seated blockers; draw shuffle_time per blocker;
  seat_interference_events counted per blocking passenger.
- Termination: max(seated_time); safety cap raises SimulationError.
- Update order: front-most in aisle (lowest cell) first, so a clearing
  passenger frees its cell before the one behind it tries to advance.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from boarding.cabin import Cabin
from boarding.passenger import Passenger, PassengerState

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SimulationError(Exception):
    """Raised when the simulation hits the safety cap – possible deadlock."""


# ---------------------------------------------------------------------------
# Lognormal helper
# ---------------------------------------------------------------------------

def _lognormal_params(desired_mean: float, desired_sd: float) -> tuple[float, float]:
    """
    Convert a desired mean and standard deviation of a lognormal distribution
    to the (mu, sigma) parameters of the underlying normal.

    numpy's ``rng.lognormal(mean=mu, sigma=sigma)`` draws from a lognormal
    whose underlying normal has mean *mu* and std *sigma*, so:
        E[X]   = exp(mu + sigma^2/2)
        Std[X] = sqrt((exp(sigma^2)-1) * exp(2*mu + sigma^2))

    Inverting:
        sigma^2 = log(1 + (desired_sd / desired_mean)^2)
        mu      = log(desired_mean) - sigma^2/2
    """
    sigma_sq = math.log(1.0 + (desired_sd / desired_mean) ** 2)
    sigma = math.sqrt(sigma_sq)
    mu = math.log(desired_mean) - sigma_sq / 2.0
    return mu, sigma


# ---------------------------------------------------------------------------
# Passenger population generator
# ---------------------------------------------------------------------------

def generate_passengers(cfg: dict, rng: np.random.Generator) -> list[Passenger]:
    """
    Create and return the full list of passengers (one per seat) with
    random attributes drawn from the distributions in *cfg*.

    Attribute distributions (all ASSUMED unless noted):
    - walk_speed_mps : Normal(mean, sd) clipped at *min*
    - num_bags       : Categorical with probabilities from config
    - stow_time_s    : Sum of Lognormal draws (one per bag); 0 if 0 bags

    Parameters
    ----------
    cfg : dict
        Full simulation config (scenario already merged).
    rng : numpy.random.Generator
        Seeded RNG; all randomness consumed here for reproducibility.

    Returns
    -------
    list[Passenger]
        180 passengers (30 rows × 6 seats), id=0 … 179, sorted by
        row then seat letter.
    """
    cabin_cfg = cfg["cabin"]
    pax_cfg   = cfg["passenger"]

    rows    = cabin_cfg["rows"]
    letters = cabin_cfg["seat_letters"]

    # ---- Walk speed -------------------------------------------------------
    ws_cfg  = pax_cfg["walk_speed_mps"]
    ws_mean = ws_cfg["mean"]
    ws_sd   = ws_cfg["sd"]
    ws_min  = ws_cfg["min"]

    # ---- Bag-count distribution -------------------------------------------
    # Config stores keys as ints; yaml may parse them as ints or strings.
    bags_probs = pax_cfg["bags_probs"]
    bag_values  = [int(k) for k in bags_probs.keys()]
    bag_weights = [float(v) for v in bags_probs.values()]
    # Normalise weights (guard against floating-point drift in yaml)
    total_w = sum(bag_weights)
    bag_weights = [w / total_w for w in bag_weights]

    # ---- Stow-time distribution (lognormal, per-bag) ----------------------
    stow_cfg    = pax_cfg["stow_time_s"]
    per_bag_cfg = stow_cfg["per_bag"]
    stow_mu, stow_sigma = _lognormal_params(
        desired_mean=float(per_bag_cfg["mean"]),
        desired_sd=float(per_bag_cfg["sd"]),
    )

    # ---- Build passengers -------------------------------------------------
    passengers: list[Passenger] = []
    pid = 0
    for row in range(1, rows + 1):
        for letter in letters:
            # Walk speed: Normal, clipped at minimum
            speed = float(rng.normal(loc=ws_mean, scale=ws_sd))
            speed = max(speed, ws_min)

            # Number of bags
            num_bags = int(rng.choice(bag_values, p=bag_weights))

            # Stow time: sum of per-bag lognormal draws (0 if no bags)
            if num_bags == 0:
                stow_time = 0.0
            else:
                stow_time = float(
                    sum(rng.lognormal(mean=stow_mu, sigma=stow_sigma)
                        for _ in range(num_bags))
                )

            passengers.append(Passenger(
                id=pid,
                row=row,
                letter=letter,
                num_bags=num_bags,
                walk_speed_mps=speed,
                stow_time_s=stow_time,
                # group_id, is_late, is_compliant, seat_search_delay_s left
                # at their dataclass defaults; behavior.py may update them.
            ))
            pid += 1

    return passengers


# ---------------------------------------------------------------------------
# State-transition helpers
# ---------------------------------------------------------------------------

def _reset_passenger(pax: Passenger) -> None:
    """Reset all runtime fields to initial state (QUEUED, no aisle cell)."""
    pax.state                    = PassengerState.QUEUED
    pax.aisle_cell               = None
    pax.aisle_progress_m         = 0.0
    pax.stow_ticks_remaining     = 0
    pax.seating_ticks_remaining  = 0
    pax.wait_ticks               = 0
    pax.aisle_interference_events  = 0
    pax.seat_interference_events   = 0
    pax.seated_time              = None
    pax.entry_time               = None
    pax._was_blocked_last_tick   = False


def _start_stowing(
    pax: Passenger,
    base_seat_time: int,
    shuffle_low: float,
    shuffle_high: float,
    cabin: Cabin,
    rng: np.random.Generator,
) -> None:
    """
    Transition *pax* to STOWING at their target aisle cell.

    If stow_time + seat_search_delay rounds to 0 the STOWING phase is
    skipped and we go straight to SEATING.
    """
    stow_ticks = round(pax.stow_time_s + pax.seat_search_delay_s)
    if stow_ticks <= 0:
        # No bags / no delay → skip directly to seating
        _start_seating(pax, base_seat_time, shuffle_low, shuffle_high, cabin, rng)
    else:
        pax.state = PassengerState.STOWING
        pax.stow_ticks_remaining = stow_ticks


def _start_seating(
    pax: Passenger,
    base_seat_time: int,
    shuffle_low: float,
    shuffle_high: float,
    cabin: Cabin,
    rng: np.random.Generator,
) -> None:
    """
    Transition *pax* to SEATING.

    Count how many seats between *pax*'s seat and the aisle are already
    occupied.  For each blocker, draw a shuffle time from Uniform(low, high)
    and add one seat_interference_event.
    """
    n_blockers = cabin.count_blockers(pax.row, pax.letter)
    total_shuffle = 0.0
    for _ in range(n_blockers):
        total_shuffle += rng.uniform(shuffle_low, shuffle_high)
        pax.seat_interference_events += 1  # one event per blocking passenger

    pax.seating_ticks_remaining = base_seat_time + round(total_shuffle)
    pax.state = PassengerState.SEATING


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_simulation(
    order: list[Passenger],
    cabin: Cabin,
    cfg: dict,
    rng: np.random.Generator,
    test_mode: bool = False,
    record_history: bool = False,
) -> dict:
    """
    Run the boarding simulation for *order* and return per-run metrics.

    The *cabin* must be a freshly initialised Cabin (all cells empty).
    Passenger runtime fields are reset at the start of this function so
    the same Passenger objects can be reused across strategy runs (CRN).

    Parameters
    ----------
    order : list[Passenger]
        Boarding call order (index 0 boards first).
    cabin : Cabin
        Fresh, empty cabin layout.
    cfg : dict
        Full simulation config (scenario merged).
    rng : numpy.random.Generator
        RNG for stochastic engine events (shuffle times).
    test_mode : bool
        If True, assert invariants every tick (slow but catches bugs).
    record_history : bool
        If True, record per-tick snapshots for animation replay.
        The result dict will include a ``"history"`` key containing a
        list of dicts, one per tick, each with::

            { "tick": int,
              "states": [
                  { "id": int, "state": str, "aisle_cell": int|None,
                    "row": int, "letter": str, "group_id": int|None },
                  … one entry per passenger …
              ] }

    Returns
    -------
    dict
        Metrics dict from ``metrics.collect_metrics``; also contains
        ``passengers`` key with the Passenger list for further inspection.
        If *record_history* is True, also contains ``"history"``.

    Raises
    ------
    SimulationError
        If the simulation exceeds ``max_ticks`` (deadlock guard).
    """
    sim_cfg      = cfg["simulation"]
    dt           = sim_cfg["dt_seconds"]          # 1 s per tick
    entry_ivl    = sim_cfg["entry_interval_s"]    # 3 s between entries
    base_seat    = sim_cfg["base_seat_time_s"]    # 2 s to sit down (no blocker)
    shuf_low     = float(sim_cfg["shuffle_time_s"]["low"])
    shuf_high    = float(sim_cfg["shuffle_time_s"]["high"])
    max_ticks    = sim_cfg["max_ticks"]
    row_pitch    = cfg["cabin"]["row_pitch_m"]

    n_pax = len(order)

    # ------------------------------------------------------------------
    # Reset all passengers to QUEUED state
    # ------------------------------------------------------------------
    for pax in order:
        _reset_passenger(pax)

    # ------------------------------------------------------------------
    # Simulation state
    # ------------------------------------------------------------------
    queue_idx       = 0    # index of next passenger in *order* to enter
    next_entry_tick = 0    # earliest tick the next entry is permitted
    in_aisle: list[Passenger] = []   # passengers currently in the aisle
    history: list[dict] = []         # per-tick snapshots (when record_history=True)

    def _snapshot(tick: int) -> None:
        """Append a lightweight state snapshot to *history*."""
        history.append({
            "tick": tick,
            "states": [
                {
                    "id":         pax.id,
                    "state":      pax.state.name,
                    "aisle_cell": pax.aisle_cell,
                    "row":        pax.row,
                    "letter":     pax.letter,
                    "group_id":   getattr(pax, "group_id", None),
                }
                for pax in order
            ],
        })

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------
    for tick in range(max_ticks + 1):

        # --- Termination check (beginning of tick) ----------------------
        if sum(1 for p in order if p.state == PassengerState.SEATED) == n_pax:
            from boarding.metrics import collect_metrics
            return collect_metrics(order)

        # --- Entry -------------------------------------------------------
        # Try to admit the next queued passenger if:
        #   (a) there are passengers still waiting, AND
        #   (b) it is past/at the scheduled entry tick, AND
        #   (c) aisle cell 0 is free.
        # The entry timer advances only when a passenger actually enters
        # (if cell 0 is occupied, we retry each subsequent tick).
        if queue_idx < n_pax and tick >= next_entry_tick:
            if cabin.is_aisle_cell_free(0):
                pax = order[queue_idx]
                queue_idx += 1
                pax.state           = PassengerState.WALKING
                pax.aisle_cell      = 0
                pax.entry_time      = tick
                pax.aisle_progress_m = 0.0
                cabin.enter_aisle(pax.id, 0)
                in_aisle.append(pax)
                next_entry_tick = tick + entry_ivl

        # --- Update passengers: front-most aisle cell first --------------
        # Sort once per tick by current aisle_cell so that a passenger
        # clearing their cell is processed before the one behind them.
        in_aisle_sorted = sorted(in_aisle, key=lambda p: p.aisle_cell)

        for pax in in_aisle_sorted:
            target_cell = cabin.row_to_cell(pax.row)

            # ............. STOWING ...........................................
            if pax.state == PassengerState.STOWING:
                pax.stow_ticks_remaining -= 1
                if pax.stow_ticks_remaining <= 0:
                    _start_seating(pax, base_seat, shuf_low, shuf_high, cabin, rng)

            # ............. SEATING ...........................................
            elif pax.state == PassengerState.SEATING:
                pax.seating_ticks_remaining -= 1
                if pax.seating_ticks_remaining <= 0:
                    # Passenger is fully seated: free the aisle cell.
                    cabin.leave_aisle(pax.aisle_cell)
                    cabin.seat_passenger(pax.id, pax.row, pax.letter)
                    pax.aisle_cell  = None
                    pax.state       = PassengerState.SEATED
                    pax.seated_time = tick

            # ............. WALKING / BLOCKED .................................
            elif pax.state in (PassengerState.WALKING, PassengerState.BLOCKED):

                if pax.aisle_cell == target_cell:
                    # Passenger has arrived at their row → start stowing.
                    _start_stowing(pax, base_seat, shuf_low, shuf_high, cabin, rng)

                else:
                    next_cell = pax.aisle_cell + 1

                    # Accumulate walking progress only when not blocked
                    # (a blocked passenger stands still at the cell boundary).
                    if pax.state != PassengerState.BLOCKED:
                        # Cap at row_pitch: prevents accumulation beyond one
                        # cell's worth, keeping advance to at most 1 per tick.
                        pax.aisle_progress_m = min(
                            pax.aisle_progress_m + pax.walk_speed_mps * dt,
                            row_pitch,
                        )

                    if pax.aisle_progress_m >= row_pitch:
                        # Ready to advance – check if next cell is free.
                        if cabin.is_aisle_cell_free(next_cell):
                            # ---- Move forward one cell ----
                            cabin.move_aisle(pax.id, pax.aisle_cell, next_cell)
                            pax.aisle_cell       = next_cell
                            pax.aisle_progress_m = 0.0   # reset; one move per tick
                            pax.state            = PassengerState.WALKING
                            pax._was_blocked_last_tick = False

                            # Check if the advance landed on the target row.
                            if next_cell == target_cell:
                                _start_stowing(
                                    pax, base_seat, shuf_low, shuf_high, cabin, rng
                                )

                        else:
                            # ---- Blocked by the passenger ahead ----
                            pax.state       = PassengerState.BLOCKED
                            pax.wait_ticks += 1
                            # Count one interference event per continuous episode
                            # (not per tick).
                            if not pax._was_blocked_last_tick:
                                pax.aisle_interference_events += 1
                            pax._was_blocked_last_tick = True

                    else:
                        # Still accumulating progress, not yet ready to advance.
                        pax.state = PassengerState.WALKING
                        pax._was_blocked_last_tick = False

        # --- Remove newly seated passengers from the active list ----------
        in_aisle = [p for p in in_aisle if p.state != PassengerState.SEATED]

        # --- Termination check (end of tick) – catches the last seating ---
        if sum(1 for p in order if p.state == PassengerState.SEATED) == n_pax:
            if record_history:
                _snapshot(tick)
            from boarding.metrics import collect_metrics
            result = collect_metrics(order)
            if record_history:
                result["history"] = history
            return result

        # --- Optional invariant assertions (test mode) -------------------
        if test_mode:
            _assert_invariants(cabin, in_aisle, order, tick)

        # --- History snapshot --------------------------------------------
        if record_history:
            _snapshot(tick)

    # Safety cap reached – something went wrong.
    seated = sum(1 for p in order if p.state == PassengerState.SEATED)
    raise SimulationError(
        f"Simulation exceeded {max_ticks} ticks without all passengers seated. "
        f"Seated: {seated}/{n_pax}. Possible deadlock."
    )


# ---------------------------------------------------------------------------
# Early termination helper for metrics (called from both termination points)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _assert_invariants(
    cabin: Cabin,
    in_aisle: list[Passenger],
    all_passengers: list[Passenger],
    tick: int,
) -> None:
    """
    Assert all four simulation invariants.

    Called every tick when test_mode=True.  Raises AssertionError with a
    descriptive message on the first violation found.

    Invariants
    ----------
    1. No two passengers share an aisle cell.
    2. Passengers never move backward (structurally guaranteed by the engine –
       aisle_cell only ever increases; asserted via the cabin's move_aisle).
    3. Each seat holds at most one passenger (checked via cabin state).
    4. A passenger is seated only in their assigned seat.
    """
    # ---- Invariant 1: unique aisle cells --------------------------------
    cells = [p.aisle_cell for p in in_aisle if p.aisle_cell is not None]
    assert len(cells) == len(set(cells)), (
        f"Tick {tick} | Invariant 1 violated: "
        f"duplicate aisle cells detected: {sorted(cells)}"
    )

    # ---- Invariant 2: no backward movement ------------------------------
    # The engine only ever calls cabin.move_aisle(id, from_cell, to_cell)
    # with to_cell = from_cell + 1.  The cabin.move_aisle method validates
    # that the passenger is in from_cell.  So backward movement is
    # structurally impossible; no separate runtime check is needed.

    # ---- Invariant 3: each seat holds at most one passenger -------------
    # The cabin.seat_passenger method raises ValueError on double-seating,
    # so this is also structurally enforced.  Cross-check via occupied map:
    for row, row_dict in cabin.occupied.items():
        occupants = [pid for pid in row_dict.values() if pid is not None]
        assert len(occupants) == len(set(occupants)), (
            f"Tick {tick} | Invariant 3 violated: "
            f"row {row} has duplicate occupant ids: {occupants}"
        )

    # ---- Invariant 4: passenger seated only in assigned seat ------------
    for pax in all_passengers:
        if pax.state == PassengerState.SEATED:
            actual_pid = cabin.occupied[pax.row].get(pax.letter)
            assert actual_pid == pax.id, (
                f"Tick {tick} | Invariant 4 violated: "
                f"passenger {pax.id} (assigned {pax.row}{pax.letter}) "
                f"found in cabin as pid={actual_pid}."
            )
