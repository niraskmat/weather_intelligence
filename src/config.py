from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings configuration class.

    This class defines all configurable parameters for the application.
    Settings can be overridden via environment variables or .env file.
    All environment variables should match the field names (case-sensitive).

    Attributes:
        LOG_LEVEL: Controls the verbosity of application logging
                  Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        DATA_SEED: Random seed for reproducible data generation and analysis
                  Ensures consistent results across runs when set
        CACHE_RESULTS: Toggle for enabling/disabling result caching
                      When True, expensive computations are cached to disk
    """
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    DATA_SEED: int = Field(42, description="Random seed for data generation")
    #ANALYSIS_WINDOW_DAYS: int = Field(7, description="Default analysis window in days")
    CACHE_RESULTS: bool = Field(False, description="Enable or disable result caching")

    class Config:
        # Load environment variables from .env file located one directory up
        env_file = "../.env"
        env_file_encoding = 'utf-8'
        # Environment variables take precedence over .env file values
        # Field names must match environment variable names exactly

@lru_cache()
def get_settings() -> Settings:
    return Settings()

