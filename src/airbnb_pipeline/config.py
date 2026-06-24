"""Configuration loader from cities.yml."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from .utils import get_project_root

logger = logging.getLogger("airbnb_pipeline.config")

# Required top-level keys for each city configuration
REQUIRED_CITY_FIELDS = ["display_name", "country", "snapshot_date", "files"]
REQUIRED_FILE_FIELDS = ["url", "local_name"]


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    """
    Load the full cities.yml configuration file.

    Args:
        config_path: Optional path override. Defaults to <project_root>/config/cities.yml.

    Returns:
        Parsed YAML as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "cities.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", config_path)
    return config


def get_city_config(city_key: str, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Retrieve and validate configuration for a specific city.

    Args:
        city_key: The city identifier (e.g., 'bangkok').
        config: Optional pre-loaded config dict. If None, loads from default path.

    Returns:
        City configuration dictionary.

    Raises:
        KeyError: If the city key is not found.
        ValueError: If required fields are missing.
    """
    if config is None:
        config = load_config()

    cities = config.get("cities", {})
    if city_key not in cities:
        available = list(cities.keys())
        raise KeyError(
            f"City '{city_key}' not found in configuration. Available cities: {available}"
        )

    city_cfg = cities[city_key]
    _validate_city_config(city_key, city_cfg)

    return city_cfg


def _validate_city_config(city_key: str, city_cfg: Dict[str, Any]) -> None:
    """
    Validate that a city configuration contains all required fields.

    Raises:
        ValueError: If any required field is missing.
    """
    for field in REQUIRED_CITY_FIELDS:
        if field not in city_cfg:
            raise ValueError(
                f"City '{city_key}' is missing required field: '{field}'"
            )

    files = city_cfg["files"]
    for file_key, file_cfg in files.items():
        for field in REQUIRED_FILE_FIELDS:
            if field not in file_cfg:
                raise ValueError(
                    f"File '{file_key}' in city '{city_key}' is missing required field: '{field}'"
                )

    logger.debug("City config for '%s' validated successfully.", city_key)


def get_data_paths(city_key: str, city_cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    Return standard data directory paths for a city.

    Args:
        city_key: City identifier.
        city_cfg: City configuration dict.

    Returns:
        Dictionary with keys: raw_dir, processed_dir, warehouse_dir, reports_dir, figures_dir.
    """
    root = get_project_root()
    snapshot = city_cfg["snapshot_date"]

    return {
        "raw_dir": root / "data" / "raw" / city_key / snapshot,
        "processed_dir": root / "data" / "processed" / city_key,
        "warehouse_dir": root / "data" / "warehouse",
        "reports_dir": root / "reports",
        "figures_dir": root / "reports" / "figures",
    }
