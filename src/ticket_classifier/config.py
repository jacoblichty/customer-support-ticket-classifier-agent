"""
Configuration management for the ticket classifier system.
"""

import os
from typing import Optional, List, Union
from pydantic import Field, field_validator, AliasChoices
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # API Configuration
    app_name: str = "Customer Support Ticket Classifier API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False)
    
    # Server Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    
    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = Field(default=None)
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-02-15-preview")
    azure_openai_deployment_name: str = Field(default="gpt-4")
    openai_temperature: float = Field(default=0.3)
    openai_max_tokens: int = Field(default=200)
    openai_timeout: float = Field(default=30.0)
    openai_max_retries: int = Field(default=3)
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=60)  # seconds
    
    # Batch Processing
    max_batch_size: int = Field(default=100)
    max_concurrent_requests: int = Field(default=10)
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # CORS Configuration
    cors_origins: Union[str, List[str]] = Field(default=["*"])
    cors_methods: List[str] = Field(default=["*"])
    cors_headers: List[str] = Field(default=["*"])
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_path: str = Field(default="/metrics")
    
    # Database (for future use)
    database_url: Optional[str] = Field(default=None)
    
    # Ticket Categories
    ticket_categories: List[str] = Field(
        default=[
            "technical_issue",
            "billing_inquiry",
            "account_management", 
            "feature_request",
            "general_inquiry",
            "complaint",
            "refund_request"
        ]
    )
    
    @field_validator("openai_temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return [str(origin).strip() for origin in v if str(origin).strip()]
        return v
    
    @field_validator("ticket_categories", mode="before")
    @classmethod
    def parse_ticket_categories(cls, v):
        if isinstance(v, str):
            return [cat.strip() for cat in v.split(",")]
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "json_schema_extra": {
            "example": {
                "azure_openai_api_key": "your-azure-openai-key",
                "azure_openai_endpoint": "https://your-resource.openai.azure.com/",
                "azure_openai_deployment_name": "gpt-4",
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "log_level": "INFO"
            }
        }
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    host: str = "127.0.0.1"
    workers: int = 1


class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    workers: int = 4
    

class UnitTestSettings(Settings):
    """Unit testing environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    azure_openai_api_key: str = "test-key"
    azure_openai_endpoint: str = "https://test.openai.azure.com/"
    host: str = "127.0.0.1"
    port: int = 8001


def get_environment_settings() -> Settings:
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return UnitTestSettings()
    else:
        return DevelopmentSettings()