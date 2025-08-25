"""
Configuration settings for Jina Embeddings v4.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    name: str = "jinaai/jina-embeddings-v4"
    torch_dtype: str = "float32"
    trust_remote_code: bool = True
    cache_dir: Optional[str] = ".cache/models"
    
    
@dataclass
class DeviceConfig:
    """Device configuration."""
    preference: str = "auto"  # auto, cuda, mps, cpu
    enable_mps: bool = True
    use_mps_for_images: bool = False
    

@dataclass
class EmbeddingConfig:
    """Embedding generation configuration."""
    default_dim: int = 2048
    min_dim: int = 128
    max_dim: int = 2048
    default_text_task: str = "retrieval"
    default_image_task: str = "retrieval"
    normalize: bool = True
    

@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    batch_size: int = 8
    max_batch_size: int = 32
    num_workers: int = 2
    timeout_seconds: int = 300
    enable_amp: bool = False
    

@dataclass
class PathConfig:
    """File and directory paths."""
    assets_dir: str = "assets"
    samples_dir: str = "assets/samples"
    outputs_dir: str = "outputs"
    cache_dir: str = ".cache"
    

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization setup."""
        self._setup_logging()
        self._create_directories()
        self._load_environment_overrides()
    
    def _setup_logging(self):
        """Configure logging based on settings."""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format=self.logging.format
        )
        
        if self.logging.file:
            file_handler = logging.FileHandler(self.logging.file)
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            logging.getLogger().addHandler(file_handler)
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.paths.assets_dir,
            self.paths.samples_dir,
            self.paths.outputs_dir,
            self.paths.cache_dir,
            self.model.cache_dir
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            "JINA_MODEL_NAME": ("model", "name"),
            "JINA_DEVICE": ("device", "preference"),
            "JINA_BATCH_SIZE": ("performance", "batch_size", int),
            "JINA_LOG_LEVEL": ("logging", "level"),
            "JINA_CACHE_DIR": ("paths", "cache_dir"),
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Handle type conversion
                if len(config_path) > 2:
                    value = config_path[2](value)
                
                # Set the configuration value
                section = getattr(self, config_path[0])
                setattr(section, config_path[1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create nested config objects
        config = cls(
            model=ModelConfig(**data.get("model", {})),
            device=DeviceConfig(**data.get("device", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            performance=PerformanceConfig(**data.get("performance", {})),
            paths=PathConfig(**data.get("paths", {})),
            logging=LoggingConfig(**data.get("logging", {}))
        )
        
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 60)
        print("JINA EMBEDDINGS V4 CONFIGURATION")
        print("=" * 60)
        
        sections = [
            ("Model", self.model),
            ("Device", self.device),
            ("Embedding", self.embedding),
            ("Performance", self.performance),
            ("Paths", self.paths),
            ("Logging", self.logging)
        ]
        
        for section_name, section_config in sections:
            print(f"\n{section_name} Settings:")
            for key, value in asdict(section_config).items():
                print(f"  {key}: {value}")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        config_path: Path to configuration file (JSON)
        
    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        return Config.load(config_path)
    else:
        return Config()


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config