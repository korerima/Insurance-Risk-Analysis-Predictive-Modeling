"""
Configuration settings for the insurance risk analysis project.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Pydantic v2 settings configuration
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Project paths
    project_root: Path = Field(default=Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default=Path("Datasets"))
    models_dir: Path = Field(default=Path("models"))
    logs_dir: Path = Field(default=Path("logs"))
    
    # Data settings
    data_file: str = Field(default="MachineLearningRating_v4.csv")
    delimiter: str = Field(default="|")
    
    # Model settings
    random_state: int = Field(default=42)
    test_size: float = Field(default=0.2)
    validation_size: float = Field(default=0.1)
    
    # Feature engineering settings
    categorical_columns: list = Field(default=[
        "Province", "VehicleType", "Gender", "Title", 
        "Language", "LegalType", "Citizenship"
    ])
    
    numerical_columns: list = Field(default=[
        "TotalPremium", "TotalClaims", "LossRatio", "HadClaim"
    ])
    
    # Logging
    log_level: str = Field(default="INFO")
    
    # Model hyperparameters
    max_depth: int = Field(default=10)
    n_estimators: int = Field(default=100)
    learning_rate: float = Field(default=0.1)
    
    


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def create_directories(settings: Settings) -> None:
    """Create necessary directories if they don't exist."""
    for path in [settings.data_dir, settings.models_dir, settings.logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
