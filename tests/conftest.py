import pytest
import pandas as pd
import numpy as np
import logging
from typing import Iterator
from fastapi.testclient import TestClient

from src.api.main import app
from src.analysis_engine import WeatherAnalyzer

# Configure logging for tests - generally minimal logging is preferred in tests
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """
    Create a FastAPI test client for API endpoint testing.

    The client is scoped to the module level to avoid recreating
    it for each test function, improving test performance.

    Yields:
        TestClient: A test client instance for making API requests
    """
    with TestClient(app) as test_client:
        logger.debug("Created FastAPI test client")
        yield test_client


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Create a synthetic environmental dataset with known patterns and anomalies.

    This fixture generates a controlled dataset spanning 4 weeks (672 hours) with:
    - Realistic seasonal patterns for temperature, humidity, and air pressure
    - Known anomalies at specific indices for testing anomaly detection
    - Daily and weekly cycles that mirror real environmental data
    - Sufficient data points for statistical analysis while remaining fast for testing

    The data includes:
    - Temperature: Daily sinusoidal pattern with 20°C mean, ±10°C amplitude
    - Humidity: Daily cycle centered at 50%, inverse correlation with temperature
    - Air pressure: Composite 48-hour and weekly cycles around 1013 hPa
    - Intentional anomalies: Extreme values at index 10 for all parameters

    Returns:
        pd.DataFrame: Indexed by hourly timestamps with columns:
            - temperature_c: Temperature in Celsius
            - humidity_percent: Relative humidity percentage
            - air_pressure_hpa: Air pressure in hectopascals
    """
    # Generate 8 weeks of hourly data (sufficient for pattern detection)
    hours = 24 * 7 * 4

    # Create timestamp index starting from a fixed date for reproducibility
    date_range = pd.date_range("2024-01-01", periods=hours, freq="h")

    # Generate temperature with daily cycle (24-hour period)
    # 4 complete cycles per week * 8 weeks = 32 total cycles
    temperature = np.sin(np.linspace(0, 2 * np.pi * 4 * 8, hours)) * 10 + 20

    # Generate humidity with daily cycle, centered at 50%
    humidity = 50 - 20 * np.sin(np.linspace(0, 2 * np.pi * 4 * 8, hours))

    # Generate air pressure with composite cycles
    # 48-hour cycle (semi-diurnal) + 168-hour weekly cycle
    pressure = (
            1013 +  # Standard atmospheric pressure base
            5 * np.sin(np.linspace(0, 2 * np.pi * hours / 48, hours)) +  # 48h cycle
            3 * np.sin(np.linspace(0, 2 * np.pi * hours / 168, hours))  # Weekly cycle
    )

    # Create the dataframe
    data = {
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "air_pressure_hpa": pressure
    }

    # Insert known anomalies at index 10 for testing anomaly detection
    # These extreme values should be easily detectable by statistical methods
    data["temperature_c"][10] = 100.0  # Extremely high temperature
    data["humidity_percent"][10] = 120.0  # Impossible humidity (>100%)
    data["air_pressure_hpa"][10] = 1100.0  # Very high pressure

    df = pd.DataFrame(data, index=date_range)
    df.index.name = "timestamp"

    logger.debug(f"Generated sample dataframe with {len(df)} records")
    logger.debug(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


@pytest.fixture
def simple_analyzer(sample_dataframe: pd.DataFrame) -> WeatherAnalyzer:
    """
    Create a pre-configured WeatherAnalyzer instance with sample data.

    This fixture provides a WeatherAnalyzer that has already been initialized
    with the sample dataset and has completed the full data cleaning pipeline

    The analyzer is ready for testing analysis functions without requiring
    additional setup or data processing steps.

    Args:
        sample_dataframe: The synthetic environmental dataset fixture

    Returns:
        WeatherAnalyzer: Fully initialized analyzer instance with processed data

    Example:
        def test_correlation_analysis(simple_analyzer):
            corr_matrix = simple_analyzer.get_correlation_matrix()
            assert corr_matrix.shape == (3, 3)
    """
    logger.debug("Initializing WeatherAnalyzer with sample data")
    analyzer = WeatherAnalyzer(sample_dataframe)
    logger.debug("WeatherAnalyzer initialization complete")
    return analyzer