"""
Configuration management for Worker Activity Monitoring system.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_type: str = 'standard'  # 'standard' or 'lightweight'
    num_classes: int = 7
    input_size: tuple = (224, 224)
    pretrained: bool = False
    dropout_rate: float = 0.5

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 10
    num_workers: int = 4
    use_augmentation: bool = True
    validation_split: float = 0.2

@dataclass
class DataConfig:
    """Data configuration parameters."""
    data_dir: str = 'data'
    image_size: tuple = (224, 224)
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

@dataclass
class InferenceConfig:
    """Inference configuration parameters."""
    confidence_threshold: float = 0.5
    smoothing_window: int = 5
    update_interval: float = 1.0
    batch_size: int = 32
    save_predictions: bool = True

@dataclass
class VisualizationConfig:
    """Visualization configuration parameters."""
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    color_palette: str = 'Set3'
    save_format: str = 'png'

@dataclass
class SystemConfig:
    """System configuration parameters."""
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    num_threads: int = 4
    memory_limit: Optional[int] = None
    log_level: str = 'INFO'

class ConfigManager:
    """
    Configuration manager for the Worker Activity Monitoring system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'config.json'
        self.class_names = [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        
        # Initialize default configurations
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
        self.inference_config = InferenceConfig()
        self.visualization_config = VisualizationConfig()
        self.system_config = SystemConfig()
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        path = config_path or self.config_path
        
        if not os.path.exists(path):
            print(f"Config file {path} not found. Using default configuration.")
            return self.to_dict()
        
        try:
            with open(path, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'model' in config_data:
                self.model_config = ModelConfig(**config_data['model'])
            if 'training' in config_data:
                self.training_config = TrainingConfig(**config_data['training'])
            if 'data' in config_data:
                self.data_config = DataConfig(**config_data['data'])
            if 'inference' in config_data:
                self.inference_config = InferenceConfig(**config_data['inference'])
            if 'visualization' in config_data:
                self.visualization_config = VisualizationConfig(**config_data['visualization'])
            if 'system' in config_data:
                self.system_config = SystemConfig(**config_data['system'])
            
            print(f"Configuration loaded from {path}")
            return config_data
            
        except Exception as e:
            print(f"Error loading config file {path}: {e}")
            print("Using default configuration.")
            return self.to_dict()
    
    def save_config(self, config_path: Optional[str] = None) -> str:
        """Save current configuration to JSON file."""
        path = config_path or self.config_path
        
        try:
            config_data = self.to_dict()
            
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"Configuration saved to {path}")
            return path
            
        except Exception as e:
            print(f"Error saving config file {path}: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'data': asdict(self.data_config),
            'inference': asdict(self.inference_config),
            'visualization': asdict(self.visualization_config),
            'system': asdict(self.system_config),
            'class_names': self.class_names
        }
    
    def update_config(self, section: str, **kwargs) -> None:
        """Update specific configuration section."""
        if section == 'model':
            for key, value in kwargs.items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        elif section == 'training':
            for key, value in kwargs.items():
                if hasattr(self.training_config, key):
                    setattr(self.training_config, key, value)
        elif section == 'data':
            for key, value in kwargs.items():
                if hasattr(self.data_config, key):
                    setattr(self.data_config, key, value)
        elif section == 'inference':
            for key, value in kwargs.items():
                if hasattr(self.inference_config, key):
                    setattr(self.inference_config, key, value)
        elif section == 'visualization':
            for key, value in kwargs.items():
                if hasattr(self.visualization_config, key):
                    setattr(self.visualization_config, key, value)
        elif section == 'system':
            for key, value in kwargs.items():
                if hasattr(self.system_config, key):
                    setattr(self.system_config, key, value)
        else:
            raise ValueError(f"Unknown configuration section: {section}")
    
    def get_device(self) -> str:
        """Get the appropriate device for computation."""
        if self.system_config.device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.system_config.device
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Validate model config
        if self.model_config.num_classes != len(self.class_names):
            errors.append(f"Model num_classes ({self.model_config.num_classes}) doesn't match class_names length ({len(self.class_names)})")
        
        if self.model_config.model_type not in ['standard', 'lightweight']:
            errors.append(f"Invalid model_type: {self.model_config.model_type}")
        
        # Validate training config
        if self.training_config.batch_size <= 0:
            errors.append(f"Invalid batch_size: {self.training_config.batch_size}")
        
        if self.training_config.learning_rate <= 0:
            errors.append(f"Invalid learning_rate: {self.training_config.learning_rate}")
        
        if self.training_config.epochs <= 0:
            errors.append(f"Invalid epochs: {self.training_config.epochs}")
        
        # Validate data config
        if not os.path.exists(self.data_config.data_dir):
            errors.append(f"Data directory does not exist: {self.data_config.data_dir}")
        
        # Validate inference config
        if not 0 <= self.inference_config.confidence_threshold <= 1:
            errors.append(f"Invalid confidence_threshold: {self.inference_config.confidence_threshold}")
        
        if self.inference_config.smoothing_window <= 0:
            errors.append(f"Invalid smoothing_window: {self.inference_config.smoothing_window}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed!")
        return True
    
    def print_config(self) -> None:
        """Print current configuration."""
        print("Current Configuration:")
        print("=" * 50)
        
        config_dict = self.to_dict()
        for section, params in config_dict.items():
            if section != 'class_names':
                print(f"\n{section.upper()}:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
        
        print(f"\nCLASS_NAMES:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")

def create_default_config_file(config_path: str = 'config.json') -> str:
    """Create a default configuration file."""
    config_manager = ConfigManager(config_path)
    return config_manager.save_config()

def load_config_from_file(config_path: str) -> ConfigManager:
    """Load configuration from file and return ConfigManager instance."""
    config_manager = ConfigManager(config_path)
    config_manager.load_config()
    return config_manager

if __name__ == "__main__":
    # Create and save default configuration
    config_path = create_default_config_file()
    print(f"Default configuration created at: {config_path}")
    
    # Load and display configuration
    config_manager = load_config_from_file(config_path)
    config_manager.print_config()
    
    # Validate configuration
    config_manager.validate_config()
