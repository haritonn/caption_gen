import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            current_dir = Path.cwd()
            possible_paths = [
                current_dir / "config.yaml",
                current_dir.parent / "config.yaml",
                Path(__file__).parent / "config.yaml",
                Path(__file__).parent.parent / "config.yaml"
            ]

            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

            if config_path is None:
                raise FileNotFoundError("config.yaml not found in any of the expected locations")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_project_config(self) -> Dict[str, Any]:
        return self.config.get('project', {})

    def get_dataset_config(self) -> Dict[str, Any]:
        return self.config.get('dataset', {})

    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        return self.config.get('evaluation', {})

    def get_hardware_config(self) -> Dict[str, Any]:
        return self.config.get('hardware', {})

    def get_data_loading_config(self) -> Dict[str, Any]:
        return self.config.get('data_loading', {})

    def get_reproducibility_config(self) -> Dict[str, Any]:
        return self.config.get('reproducibility', {})

    def get_paths_config(self) -> Dict[str, Any]:
        return self.config.get('paths', {})

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

_config_loader = None

def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    global _config_loader
    if _config_loader is None or config_path is not None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader

def reload_config(config_path: Optional[str] = None):
    global _config_loader
    _config_loader = ConfigLoader(config_path)
    return _config_loader
