"""
boarding/metrics.py
===================
Collects per-run metrics from a completed simulation run.

Public interface
----------------
    collect_metrics(passengers) -> dict

The dict is the unit of data for CSV export; the experiment runner adds
columns for strategy, scenario, seed, replication id, and sweep parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boarding.passenger import Passenger


def collect_metrics(passengers: list["Passenger"]) -> dict:
    """
    Aggregate passenger-level data from a completed run into a metrics dict.

    All passengers must be in the SEATED state before calling this function.
    If any passenger is not seated, a ValueError is raised (indicates a bug).

    Parameters
    ----------
    passengers : list[Passenger]
        All Passenger objects after the simulation has terminated.

    Returns
    -------
    dict
        Keys:
          - ``total_boarding_time_s``    : int  – tick of the last seating
          - ``aisle_interference_events``: int  – sum across all passengers
          - ``seat_interference_events`` : int  – sum across all passengers
          - ``mean_wait_time_s``         : float – mean wait_ticks per passenger
          - ``max_wait_time_s``          : int  – max wait_ticks any passenger
          - ``passengers``               : list[Passenger] – for further analysis
    """
    from boarding.passenger import PassengerState

    # Validate all passengers are seated
    not_seated = [p for p in passengers if p.state != PassengerState.SEATED]
    if not_seated:
        ids = [p.id for p in not_seated[:5]]
        raise ValueError(
            f"collect_metrics called before all passengers are seated. "
            f"First unseated ids: {ids} (and {len(not_seated) - 5} more)."
            if len(not_seated) > 5 else
            f"collect_metrics called before all passengers are seated. "
            f"Unseated ids: {ids}."
        )

    seated_times = [p.seated_time for p in passengers]
    wait_times   = [p.wait_ticks   for p in passengers]

    total_boarding_time   = max(seated_times)
    aisle_events          = sum(p.aisle_interference_events for p in passengers)
    seat_events           = sum(p.seat_interference_events  for p in passengers)
    mean_wait             = sum(wait_times) / len(wait_times)
    max_wait              = max(wait_times)

    return {
        "total_boarding_time_s":     total_boarding_time,
        "aisle_interference_events": aisle_events,
        "seat_interference_events":  seat_events,
        "mean_wait_time_s":          mean_wait,
        "max_wait_time_s":           max_wait,
        "passengers":                passengers,   # kept for further analysis
    }
