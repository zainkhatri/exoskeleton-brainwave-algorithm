"""
Configuration management for brainwave analysis.
Handles loading and validation of configuration parameters.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for brainwave analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            # Use default config
            config_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ['data', 'filtering', 'features', 'ml', 'visualization', 'paths']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data parameters
        data_config = self._config['data']
        if data_config['sample_rate'] <= 0:
            raise ValueError("Sample rate must be positive")
        if data_config['epoch_duration'] <= 0:
            raise ValueError("Epoch duration must be positive")
        if not 0 <= data_config['overlap'] < 1:
            raise ValueError("Overlap must be between 0 and 1")
        
        # Validate filtering parameters
        filter_config = self._config['filtering']
        if filter_config['low_cutoff'] >= filter_config['high_cutoff']:
            raise ValueError("Low cutoff must be less than high cutoff")
        if filter_config['filter_order'] <= 0:
            raise ValueError("Filter order must be positive")
        
        # Validate ML parameters
        ml_config = self._config['ml']
        if not 0 < ml_config['test_size'] < 1:
            raise ValueError("Test size must be between 0 and 1")
        if ml_config['cross_validation_folds'] < 2:
            raise ValueError("Cross validation folds must be at least 2")
        
        logger.info("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.sample_rate')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Set configuration {key} = {value}")
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, saves to original path.
        """
        save_path = Path(path) if path else self.config_path
        
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {save_path}")
    
    def create_directories(self):
        """Create necessary directories from configuration."""
        paths = self.get('paths', {})
        
        for path_key, path_value in paths.items():
            if path_key.endswith('_dir'):
                Path(path_value).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path_value}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment."""
        self.set(key, value)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)
