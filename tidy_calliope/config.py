"""Configuration: the packaged defaults, with a user YAML file (or dict) merged on top."""

import copy
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("default_config.yaml")


def _read_yaml(path: Path) -> dict:
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:  # e.g. the Calliope environment ships ruamel.yaml only
        from ruamel.yaml import YAML

        with open(path) as f:
            return YAML(typ="safe").load(f) or {}


def _merge(base: dict, update: dict) -> dict:
    """Recursive dict update: nested dicts are merged, everything else (lists included) is replaced."""
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(config: str | Path | dict | None = None) -> dict:
    """Defaults merged with `config` (a YAML path or a dict). Unknown top-level keys are an error (typos)."""
    defaults = _read_yaml(DEFAULT_CONFIG)
    if config is None:
        return defaults
    user = config if isinstance(config, dict) else _read_yaml(Path(config))
    unknown = set(user) - set(defaults)
    if unknown:
        raise KeyError(f"Unknown configuration keys: {sorted(unknown)} (see {DEFAULT_CONFIG})")
    unknown_outputs = set(user.get("outputs", {})) - set(defaults["outputs"])
    if unknown_outputs:
        raise KeyError(f"Unknown outputs: {sorted(unknown_outputs)}; available: {sorted(defaults['outputs'])}")
    return _merge(defaults, user)
