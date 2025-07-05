"""Configuration management for the rxiv corpus downloader.

This module provides centralized configuration management with support for:
- JSON-based configuration files with sensible defaults
- Environment variable overrides and command-line integration
- Automatic directory creation and path management
- Hierarchical configuration merging

Input:
    - config.json file (optional, defaults created if missing)
    - Environment variables and command-line overrides

Output:
    - Unified configuration object accessible throughout the application
    - Auto-generated config.json with default settings
    - Created directory structure as specified in paths configuration
"""

import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    "api": {
        "base_url": "https://api.biorxiv.org/details",
        "chunk_size": 100,
        "timeout": 30,
        "max_retries": 5,
        "retry_backoff": 2,
        "rate_limit_delay": 0.1
    },
    "sources": {
        "biorxiv": {
            "start_date": "2013-01-01",
            "enabled": True
        },
        "medrxiv": {
            "start_date": "2019-01-01",
            "enabled": True
        }
    },
    "download": {
        "num_threads": 8,
        "pdf_timeout": 30,
        "pdf_chunk_size": 1048576,  # 1MB
        "validate_pdfs": True,
        "calculate_hashes": True
    },
    "paths": {
        "data_dir": "data",
        "pdf_dir": "pdfs",
        "log_dir": "logs"
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "separate_logs": True
    }
}

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file, creating default if not exists"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    self._merge_config(self.config, user_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}, using defaults")
        else:
            self.save_config()
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default=None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'api.timeout')
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default if not found
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config_section = self.config
        
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        config_section[keys[-1]] = value
        self.save_config()
    
    def get_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get enabled sources with their configurations.
        
        Returns:
            Dictionary mapping source names to their configuration objects
        """
        return {name: config for name, config in self.config["sources"].items() 
                if config.get("enabled", True)}
    
    def get_end_date(self) -> str:
        """Get end date for downloads.
        
        Returns:
            End date string in YYYY-MM-DD format
        """
        return self.config.get("end_date", "2025-07-01")
    
    def create_directories(self) -> None:
        """Create necessary directories as specified in configuration."""
        for dir_key in ["data_dir", "pdf_dir", "log_dir"]:
            dir_path = Path(self.get(f"paths.{dir_key}"))
            dir_path.mkdir(parents=True, exist_ok=True)

# Global configuration instance
config = Config()