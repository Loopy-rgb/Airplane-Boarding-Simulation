"""
Filipino passenger behaviour modifiers.

Design principle
---------------------------------
The engine must be free of behaviour-specific ``if`` clutter.
This module modifies *passenger attributes* and *queue order* **before**
the engine runs, so the engine itself stays generic.

The two features that need engine hooks (bayanihan, seat swapping) mark
passenger objects with flag attributes that the engine already reads:
  - ``stow_time_s`` is reduced for bayanihan recipients
  - Seat-swap behaviour adds shuffle time via ``seating_ticks_remaining``
    (handled via a pre-run attribute adjustment rather than a real-time hook)

Public interface
----------------
    apply_behavior(order, passengers, rng, cfg) -> list[Passenger]

Each feature is controlled by a ``behavior.<feature>.enabled`` flag in
``config.yaml``.  All parameter values are ASSUMED (no published
Filipino-aviation data).

Feature application order
--------------------------
1. heavy_handcarry   – re-draw num_bags and stow_time from Philippine dist
2. seat_search_delay – assign seat_search_delay_s to a fraction of pax
3. group_travel      – assign group_id; keep group members together
4. non_compliance    – move non-compliant passengers to first 30% of queue
5. late_passengers   – move late passengers to end of queue
6. bayanihan         – reduce stow_time_s for passengers who get help
7. seat_swapping     – add extra shuffle penalty for swappers (optional)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from boarding.passenger import Passenger


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_behavior(
    order: list["Passenger"],
    passengers: list["Passenger"],
    rng: np.random.Generator,
    cfg: dict,
) -> list["Passenger"]:
    """
    Apply all enabled Filipino behaviour features to the boarding queue.

    Parameters
    ----------
    order : list[Passenger]
        The ideal boarding order produced by a strategy function.
    passengers : list[Passenger]
        All 180 Passenger objects.  Attribute-modifying features update
        ``num_bags``, ``stow_time_s``, ``seat_search_delay_s``, etc.
        in-place on these objects.
    rng : numpy.random.Generator
        Seeded RNG; all randomness must come from here.
    cfg : dict
        Full simulation config.

    Returns
    -------
    list[Passenger]
        The (possibly reordered) boarding queue.
    """
    beh = cfg.get("behavior", {})

    # Attribute-modifying features (affect passenger physical properties)
    if beh.get("heavy_handcarry", {}).get("enabled", False):
        _apply_heavy_handcarry(passengers, rng, beh["heavy_handcarry"],
                               cfg["passenger"]["stow_time_s"])

    if beh.get("seat_search_delay", {}).get("enabled", False):
        _apply_seat_search_delay(passengers, rng, beh["seat_search_delay"])

    # Group assignment (modifies group_id, used later for queue ordering)
    if beh.get("group_travel", {}).get("enabled", False):
        _assign_groups(passengers, rng, beh["group_travel"])

    # Queue-order features
    if beh.get("group_travel", {}).get("enabled", False):
        order = _apply_group_travel(order, rng)

    if beh.get("non_compliance", {}).get("enabled", False):
        order = _apply_non_compliance(order, rng, beh["non_compliance"])

    if beh.get("late_passengers", {}).get("enabled", False):
        order = _apply_late_passengers(order, rng, beh["late_passengers"])

    # Stow-time hook: bayanihan reduces stow_time_s before the engine runs
    if beh.get("bayanihan", {}).get("enabled", False):
        _apply_bayanihan(order, rng, beh["bayanihan"])

    # Seating hook: seat swapping adds extra seat_search_delay_s
    if beh.get("seat_swapping", {}).get("enabled", False):
        _apply_seat_swapping(order, rng, beh["seat_swapping"])

    return order


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _apply_heavy_handcarry(
    passengers: list["Passenger"],
    rng: np.random.Generator,
    feature_cfg: dict,
    stow_cfg: dict,
) -> None:
    """
    Shift the bag-count distribution to the Philippine profile and
    recompute stow_time_s for affected passengers.

    Philippine bag distribution (ASSUMED):
      0 bags: 5%, 1 bag: 45%, 2 bags: 50%

    Parameters modified in-place: ``num_bags``, ``stow_time_s``.
    """
    import math

    ph_probs = feature_cfg.get("bags_probs", {0: 0.05, 1: 0.45, 2: 0.50})
    bag_vals    = [int(k) for k in ph_probs.keys()]
    bag_weights = [float(v) for v in ph_probs.values()]
    total_w     = sum(bag_weights)
    bag_weights = [w / total_w for w in bag_weights]

    # Stow-time lognormal parameters (same distribution as baseline)
    per_bag_cfg = stow_cfg["per_bag"]
    desired_mean = float(per_bag_cfg["mean"])
    desired_sd   = float(per_bag_cfg["sd"])
    sigma_sq = math.log(1.0 + (desired_sd / desired_mean) ** 2)
    stow_sigma = math.sqrt(sigma_sq)
    stow_mu    = math.log(desired_mean) - sigma_sq / 2.0

    for pax in passengers:
        pax.num_bags = int(rng.choice(bag_vals, p=bag_weights))
        if pax.num_bags == 0:
            pax.stow_time_s = 0.0
        else:
            pax.stow_time_s = float(
                sum(rng.lognormal(mean=stow_mu, sigma=stow_sigma)
                    for _ in range(pax.num_bags))
            )


def _apply_seat_search_delay(
    passengers: list["Passenger"],
    rng: np.random.Generator,
    feature_cfg: dict,
) -> None:
    """
    Add Uniform(5, 15) s of extra row-finding delay to a fraction of pax.

    Parameter modified in-place: ``seat_search_delay_s``.
    All parameters are ASSUMED.
    """
    fraction = feature_cfg.get("seat_search_fraction", 0.10)
    low      = float(feature_cfg.get("delay_s", {}).get("low", 5))
    high     = float(feature_cfg.get("delay_s", {}).get("high", 15))

    mask = rng.random(len(passengers)) < fraction
    for pax, apply_delay in zip(passengers, mask):
        if apply_delay:
            pax.seat_search_delay_s = float(rng.uniform(low, high))
        else:
            pax.seat_search_delay_s = 0.0


def _assign_groups(
    passengers: list["Passenger"],
    rng: np.random.Generator,
    feature_cfg: dict,
) -> None:
    """
    Assign passengers to travel groups (group_id attribute).

    Group sizes are drawn from weights {2: 50%, 3: 25%, 4: 25%} (ASSUMED).
    Groups are assigned to adjacent seats in the same row where possible.
    Passengers not in a group keep group_id = None.

    Parameter modified in-place: ``group_id``, ``is_compliant``.
    """
    group_fraction = feature_cfg.get("group_fraction", 0.40)
    size_weights_cfg = feature_cfg.get("group_size_weights", {2: 0.50, 3: 0.25, 4: 0.25})
    size_vals    = [int(k) for k in size_weights_cfg.keys()]
    size_weights = [float(v) for v in size_weights_cfg.values()]
    total_w      = sum(size_weights)
    size_weights = [w / total_w for w in size_weights]

    n_total          = len(passengers)
    n_group_target   = round(n_total * group_fraction)

    # Reset group_id first (in case this function is called multiple times)
    for pax in passengers:
        pax.group_id = None

    # Build a map: row → list of passengers in that row
    from collections import defaultdict
    row_map: dict[int, list] = defaultdict(list)
    for pax in passengers:
        row_map[pax.row].append(pax)

    # Shuffle rows for randomness
    all_rows = list(row_map.keys())
    rng.shuffle(all_rows)

    group_id      = 0
    n_grouped_so_far = 0

    for row in all_rows:
        if n_grouped_so_far >= n_group_target:
            break

        row_pax = [p for p in row_map[row] if p.group_id is None]
        if len(row_pax) < 2:
            continue

        # Draw group size; cap at available ungrouped passengers in this row
        size = int(rng.choice(size_vals, p=size_weights))
        size = min(size, len(row_pax))
        if size < 2:
            continue

        # Pick *size* passengers randomly from this row
        chosen = list(rng.choice(row_pax, size=size, replace=False))
        for pax in chosen:
            pax.group_id = group_id

        group_id         += 1
        n_grouped_so_far += size


def _apply_group_travel(
    order: list["Passenger"],
    rng: np.random.Generator,
) -> list["Passenger"]:
    """
    Clump group members together in the queue at the position of the
    earliest-called member.  Internal group order is randomised.

    This must be called AFTER _assign_groups.
    """
    from collections import defaultdict

    # Identify groups and their earliest position in order
    group_positions: dict[int, int] = {}
    group_members:   dict[int, list] = defaultdict(list)

    for pos, pax in enumerate(order):
        if pax.group_id is not None:
            group_members[pax.group_id].append(pax)
            if pax.group_id not in group_positions:
                group_positions[pax.group_id] = pos

    # Build new order: non-group passengers keep their slots; groups are
    # inserted at their earliest member's position.
    groups_inserted: set[int] = set()
    new_order: list = []

    for pax in order:
        if pax.group_id is None:
            new_order.append(pax)
        elif pax.group_id not in groups_inserted:
            # Insert the whole group (shuffled internally) at this position
            members = list(group_members[pax.group_id])
            rng.shuffle(members)
            new_order.extend(members)
            groups_inserted.add(pax.group_id)
        # else: this member was already inserted with the group → skip

    return new_order


def _apply_non_compliance(
    order: list["Passenger"],
    rng: np.random.Generator,
    feature_cfg: dict,
) -> list["Passenger"]:
    """
    Move non-compliant passengers (and their groups) to a random position
    in the first 30% of the queue.

    compliance_rate = 0.70 (ASSUMED): each solo passenger or group is
    compliant with probability 0.70; non-compliant ones jump forward.

    Parameters modified in-place: ``is_compliant``.
    """
    compliance_rate = feature_cfg.get("compliance_rate", 0.70)
    cutoff = 0.30  # non-compliant jump to first 30% of queue (ASSUMED)

    from collections import defaultdict

    # Identify unique boarding units (solo passenger or group)
    unit_pax: dict = {}   # unit_key → list of passengers
    for pax in order:
        key = pax.group_id if pax.group_id is not None else pax.id
        if key not in unit_pax:
            unit_pax[key] = []
        unit_pax[key].append(pax)

    # Assign compliance to each unit
    non_compliant_keys = []
    for key, members in unit_pax.items():
        is_compliant = rng.random() < compliance_rate
        for m in members:
            m.is_compliant = is_compliant
        if not is_compliant:
            non_compliant_keys.append(key)

    if not non_compliant_keys:
        return order

    # Remove non-compliant passengers from current positions
    nc_set = {id(p) for key in non_compliant_keys for p in unit_pax[key]}
    compliant_order = [p for p in order if id(p) not in nc_set]
    nc_flat         = [p for key in non_compliant_keys for p in unit_pax[key]]

    # Shuffle non-compliant units among themselves
    rng.shuffle(nc_flat)

    # Insert non-compliant passengers at random positions in first 30%
    n_total    = len(compliant_order) + len(nc_flat)
    max_insert = max(1, round(n_total * cutoff))
    insert_pos = sorted(rng.integers(0, max_insert, size=len(nc_flat)))

    new_order: list = []
    ci   = 0  # index into compliant_order
    ni   = 0  # index into nc_flat
    for idx in range(n_total):
        if ni < len(nc_flat) and idx == insert_pos[ni]:
            new_order.append(nc_flat[ni])
            ni += 1
        elif ci < len(compliant_order):
            new_order.append(compliant_order[ci])
            ci += 1

    # Append any remaining
    while ci < len(compliant_order):
        new_order.append(compliant_order[ci]); ci += 1
    while ni < len(nc_flat):
        new_order.append(nc_flat[ni]); ni += 1

    return new_order


def _apply_late_passengers(
    order: list["Passenger"],
    rng: np.random.Generator,
    feature_cfg: dict,
) -> list["Passenger"]:
    """
    Move a small fraction of passengers to the end of the queue.

    late_fraction = 0.03 (ASSUMED).
    Parameter modified in-place: ``is_late``.
    """
    late_fraction = feature_cfg.get("late_fraction", 0.03)
    n_late        = max(1, round(len(order) * late_fraction))

    # Reset flags
    for pax in order:
        pax.is_late = False

    # Choose late passengers randomly (avoid duplicating group-logic;
    # we treat each passenger independently here – groups may be split,
    # which is intentional: a single straggler causes the group to split).
    n_late = min(n_late, len(order))
    late_indices = rng.choice(len(order), size=n_late, replace=False)
    late_set = set(late_indices)

    normal = [p for i, p in enumerate(order) if i not in late_set]
    late   = [p for i, p in enumerate(order) if i in late_set]
    for p in late:
        p.is_late = True

    return normal + late


def _apply_bayanihan(
    order: list["Passenger"],
    rng: np.random.Generator,
    feature_cfg: dict,
) -> None:
    """
    "Bayanihan" (communal helping): a group mate or the next passenger
    directly behind helps a passenger stow their bags, reducing stow_time_s.

    Applied BEFORE the engine runs (reduces stow_time_s in-place).

    p_help        = 0.15  (ASSUMED)
    help_reduction = 0.35  (ASSUMED – 35% cut to stow time)
    """
    p_help         = feature_cfg.get("p_help", 0.15)
    help_reduction = feature_cfg.get("help_reduction", 0.35)

    for i, pax in enumerate(order):
        if pax.stow_time_s <= 0:
            continue   # no bags → nothing to help with

        # Determine if a helper is available
        helper_available = False

        # Check if next passenger in queue is a group mate in the same row
        if i + 1 < len(order):
            next_pax = order[i + 1]
            if (pax.group_id is not None
                    and next_pax.group_id == pax.group_id
                    and next_pax.row == pax.row):
                helper_available = True

        # Even without a group mate, the next passenger can help with p_help
        if not helper_available and i + 1 < len(order):
            helper_available = rng.random() < p_help

        if helper_available:
            pax.stow_time_s = pax.stow_time_s * (1.0 - help_reduction)


def _apply_seat_swapping(
    order: list["Passenger"],
    rng: np.random.Generator,
    feature_cfg: dict,
) -> None:
    """
    After arriving at the row, a fraction of group members swap seats.
    This is modelled as an extra ``seat_search_delay_s`` penalty applied
    before the engine runs (avoids real-time engine hooks).

    swap_fraction = 0.05 (ASSUMED).
    Extra delay is drawn from Uniform(10, 30) s (ASSUMED; seat swap takes
    longer than normal shuffle).
    """
    swap_fraction = feature_cfg.get("swap_fraction", 0.05)
    extra_low     = 10.0   # ASSUMED
    extra_high    = 30.0   # ASSUMED

    for pax in order:
        if pax.group_id is None:
            continue
        if rng.random() < swap_fraction:
            pax.seat_search_delay_s += float(rng.uniform(extra_low, extra_high))
