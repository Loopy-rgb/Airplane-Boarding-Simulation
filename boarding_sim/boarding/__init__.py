"""
Convenience re-exports for the boarding simulation package.
Import the most-used public names so callers can write:

    from boarding import load_config, Cabin, Passenger
"""

from boarding.config_loader import load_config, apply_scenario  # noqa: F401
from boarding.cabin import Cabin, SeatType                      # noqa: F401
from boarding.passenger import Passenger, PassengerState        # noqa: F401
