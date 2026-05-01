from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_default_config_path()
        self.config = self._load_config()

    def _find_default_config_path(self) -> str:
        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "config.yaml",
            current_dir.parent / "config.yaml",
            Path(__file__).parent / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            "config.yaml not found in any of the expected locations"
        )

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            ) from error
        except yaml.YAMLError as error:
            raise ValueError(f"Error parsing YAML configuration: {error}") from error

        if config.get("hardware", {}).get("device") == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    config["hardware"]["device"] = "cpu"
            except ImportError:
                pass

        return config

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_dataset_config(self) -> Dict[str, Any]:
        return self.config.get("dataset", {})

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value


_config_loader = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    global _config_loader
    if _config_loader is None or config_path is not None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader
