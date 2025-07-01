import pytest
from http import HTTPStatus
from typing import Optional, Dict, Any
from fastapi.testclient import TestClient

from src.api.models import (
    AnomalyResponse,
    TrendResponse,
    TimeSeriesResponse,
    HealthResponse
)


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")

    # Verify successful response
    assert response.status_code == HTTPStatus.OK

    # Validate response structure using Pydantic model
    data = HealthResponse(**response.json())

    # Verify system is healthy and all components are ready
    assert data.status == "healthy"
    assert data.analyzer_ready is True
    assert data.viz_engine_ready is True


def test_data_summary(client: TestClient) -> None:
    response = client.get("/data/summary")

    # Verify successful response
    assert response.status_code == HTTPStatus.OK

    data = response.json()
    # Verify all required summary fields are present
    assert "total_records" in data, "Summary must include total record count"
    assert "basic_statistics" in data, "Summary must include statistical measures"


@pytest.mark.parametrize("parameter", [None, "temperature_c"])
def test_timeseries_data(client: TestClient, parameter: Optional[str]) -> None:
    # Prepare query parameters based on test scenario
    params = {}
    if parameter:
        params["parameter"] = parameter

    # Make request with optional parameter filtering
    response = client.get("/data/timeseries", params=params)

    # Verify successful response
    assert response.status_code == HTTPStatus.OK

    # Validate response structure using Pydantic model
    data = TimeSeriesResponse(**response.json())

    # Verify response contains expected data structure
    assert isinstance(data.count, int), "Count must be an integer"
    assert isinstance(data.data, list), "Data must be a list of time series points"

    # Verify count matches actual data length
    assert data.count == len(data.data), "Count field must match data length"


def test_correlation_endpoint(client: TestClient) -> None:
    # Define correlation analysis request
    request_payload = {
        "method": "pearson",
    }

    response = client.post("/analysis/correlation", json=request_payload)

    # Verify successful analysis
    assert response.status_code == HTTPStatus.OK

    data = response.json()

    # Verify response contains expected analysis results
    assert data["method_used"] == "pearson", "Response must confirm method used"
    assert "correlation_matrix" in data, "Must include correlation matrix"
    assert isinstance(data["correlation_matrix"], list), "Matrix must be a list"

    # Verify correlation matrix structure
    required_fields = ["variable1", "variable2", "correlation"]
    for correlation_item in data["correlation_matrix"]:
        for field in required_fields:
            assert field in correlation_item, f"Correlation item must include {field}"


@pytest.mark.parametrize("method", ["mad", "z_score"])
def test_anomaly_detection(client: TestClient, method: str) -> None:
    # Define anomaly detection request with specific method
    request_payload = {
        "method": method,
        "parameters": ["temperature_c"]
    }

    # Submit anomaly detection request
    response = client.post("/analysis/anomalies", json=request_payload)

    # Verify successful detection
    assert response.status_code == HTTPStatus.OK

    # Validate response using Pydantic model
    data = AnomalyResponse(**response.json())

    # Verify temperature parameter is included in results
    assert "temperature_c" in data.parameters, "Must include requested parameter results"


def test_trend_analysis(client: TestClient) -> None:
    response = client.get("/analysis/trends/temperature_c")

    # Verify successful analysis
    assert response.status_code == HTTPStatus.OK

    # Validate response structure using Pydantic model
    data = TrendResponse(**response.json())

    # Verify trend analysis includes required components
    assert data.patterns, "Trend analysis must include pattern information"
    assert data.extremes, "Trend analysis must include extreme value analysis"


@pytest.mark.parametrize("chart_type", [
    "timeseries",
    "correlation",
    "distribution",
    "decomposition",
    "anomaly_detection"
])
def test_visualizations(client: TestClient, chart_type: str) -> None:
    """
    Test visualization generation endpoint for different chart types.

    This comprehensive parametrized test validates visualization generation
    across all supported chart types:
    - Time series plots with trend analysis
    - Correlation heatmaps
    - Distribution histograms and fitting
    - Seasonal decomposition plots
    - Anomaly detection visualizations

    Verifies that each visualization type:
    - Returns successful HTTP response
    - Includes correct chart type identification
    - Contains base64-encoded image data
    - Provides properly structured response

    Args:
        client: FastAPI test client fixture
        chart_type: Type of visualization to generate

    Raises:
        AssertionError: If visualization generation fails or response is malformed
    """
    # Request specific visualization type
    response = client.get(f"/visualizations/{chart_type}")

    # Verify successful generation
    assert response.status_code == HTTPStatus.OK

    data = response.json()

    # Verify response structure
    assert data["chart_type"] == chart_type, "Response must confirm chart type"
    assert isinstance(data["images"], dict), "Must include image data dictionary"

    # Verify images contain base64-encoded data
    for image_key, image_data in data["images"].items():
        if isinstance(image_data, str):
            # Simple base64 image string
            assert len(image_data) > 0, f"Image data for {image_key} cannot be empty"
        elif isinstance(image_data, dict):
            # Complex image object (e.g., distribution with metadata)
            assert "image" in image_data, f"Image object for {image_key} must contain image data"
        else:
            pytest.fail(f"Unexpected image data type for {image_key}: {type(image_data)}")


def test_invalid_visualization_type(client: TestClient) -> None:
    # Request invalid visualization type
    response = client.get("/visualizations/invalid_chart")

    # Verify proper error response
    assert response.status_code == HTTPStatus.BAD_REQUEST, \
        "Invalid chart type should return 400 Bad Request"