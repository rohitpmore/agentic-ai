"""
CLI Configuration Management
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CLIConfig:
    """CLI-specific configuration"""
    
    # Display settings
    use_color: bool = True
    show_progress: bool = True
    verbose_output: bool = False
    
    # Output settings
    default_format: str = "markdown"
    auto_save: bool = True
    output_filename_template: str = "report_{timestamp}"
    
    # Interaction settings
    confirm_actions: bool = True
    show_help_on_start: bool = False
    remember_last_query: bool = True
    
    # Performance settings
    streaming_enabled: bool = True
    parallel_execution: bool = True
    timeout_seconds: int = 300
    
    def save_to_file(self, filepath: str):
        """Save configuration to file"""
        
        config_dict = asdict(self)
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'CLIConfig':
        """Load configuration from file"""
        
        if not os.path.exists(filepath):
            return cls()  # Return default config
        
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def get_default_config_path(cls) -> str:
        """Get default configuration file path"""
        
        home_dir = Path.home()
        config_dir = home_dir / ".config" / "multiagent-research"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        return str(config_dir / "cli_config.yaml")


class CLIConfigManager:
    """Manager for CLI configuration"""
    
    def __init__(self):
        self.config_path = CLIConfig.get_default_config_path()
        self.config = CLIConfig.load_from_file(self.config_path)
    
    def save_config(self):
        """Save current configuration"""
        self.config.save_to_file(self.config_path)
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = CLIConfig()
        self.save_config()