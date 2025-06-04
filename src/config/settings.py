"""
Configuration settings for the cryptocurrency advisor.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings."""
    
    # Model Configuration
    MODEL_PATH: str = Field(
        default="models/crypto_advisor",
        description="Path to the model files"
    )
    MODEL_NAME: str = Field(
        default="gpt2",
        description="Base model to use"
    )
    MAX_LENGTH: int = Field(
        default=512,
        description="Maximum sequence length"
    )
    TEMPERATURE: float = Field(
        default=0.7,
        description="Sampling temperature"
    )
    TOP_P: float = Field(
        default=0.9,
        description="Top-p sampling parameter"
    )
    
    # Exchange API Configuration
    EXCHANGE_API_KEY: Optional[str] = Field(
        default=None,
        description="Exchange API key"
    )
    EXCHANGE_API_SECRET: Optional[str] = Field(
        default=None,
        description="Exchange API secret"
    )
    EXCHANGE_ID: str = Field(
        default="binance",
        description="Exchange to use"
    )
    
    # API Server Configuration
    HOST: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    PORT: int = Field(
        default=8000,
        description="API server port"
    )
    DEBUG: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-here",
        description="Secret key for security"
    )
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # Monitoring
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    PROMETHEUS_PORT: int = Field(
        default=9090,
        description="Prometheus metrics port"
    )
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Export settings
__all__ = ["settings"] 