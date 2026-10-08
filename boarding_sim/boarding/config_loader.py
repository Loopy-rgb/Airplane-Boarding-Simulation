"""
Loads config.yaml and merges a named scenario on top of the
base configuration.

Usage
-----
    cfg = load_config()                          # base defaults
    cfg = load_config("path/to/config.yaml")
    cfg = load_config(scenario="philippine")     # merge scenario overrides
"""

import copy
import pathlib
from typing import Optional

import yaml

# Default location of the config file, relative to this file's parent directory.
_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml"


def load_config(
    path: Optional[str | pathlib.Path] = None,
    scenario: Optional[str] = None,
) -> dict:
    """
    Load and return the simulation configuration as a plain Python dict.

    Parameters
    ----------
    path : str or Path, optional
        Path to the YAML config file.  Defaults to ``boarding_sim/config.yaml``.
    scenario : str, optional
        Name of a scenario defined under ``scenarios:`` in the config (e.g.
        ``"baseline"`` or ``"philippine"``).  When supplied, the scenario's
        overrides are deep-merged on top of the base config.

    Returns
    -------
    dict
        The merged configuration dictionary.  The ``scenarios`` key is stripped
        from the returned dict to avoid confusion.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If the requested scenario name is not defined in the config.
    """
    config_path = pathlib.Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # Work on a deep copy so callers never mutate the on-disk structure.
    cfg = copy.deepcopy(raw)

    if scenario is not None:
        cfg = apply_scenario(cfg, scenario)

    # Remove the scenario registry from the returned config – it's metadata,
    # not runtime state.
    cfg.pop("scenarios", None)

    return cfg


def apply_scenario(cfg: dict, scenario: str) -> dict:
    """
    Deep-merge the overrides for ``scenario`` on top of ``cfg``.

    The ``scenarios`` section of the config maps scenario names to partial
    config dicts.  Only the keys present in the scenario dict are overwritten;
    all other keys retain their base values.

    Parameters
    ----------
    cfg : dict
        The base configuration (must already contain a ``scenarios`` section).
    scenario : str
        The scenario name to apply.

    Returns
    -------
    dict
        A new dict with the scenario overrides applied.

    Raises
    ------
    KeyError
        If ``scenario`` is not found in ``cfg["scenarios"]``.
    """
    scenarios = cfg.get("scenarios", {})
    if scenario not in scenarios:
        available = list(scenarios.keys())
        raise KeyError(
            f"Unknown scenario '{scenario}'. "
            f"Available scenarios: {available}"
        )

    overrides = scenarios[scenario]
    merged = copy.deepcopy(cfg)
    _deep_merge(merged, overrides)
    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overrides: dict) -> None:
    """
    Recursively merge *overrides* into *base* in-place.

    Dicts are merged key-by-key; all other types are replaced wholesale.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
