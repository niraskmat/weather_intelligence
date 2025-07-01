import pandas as pd
import numpy as np
import pytest

from src.analysis_engine import WeatherAnalyzer

@pytest.mark.parametrize("method", [
    "mad", "z_score", "iqr"
])
def test_get_anomalies(sample_dataframe: pd.DataFrame, method: str) -> None:
    # Initialize analyzer with sample data containing known anomalies
    analyzer = WeatherAnalyzer(sample_dataframe)

    # Detect anomalies
    if method == "mad":
        # Using MAD method with threshold of 5
        anomalies, threshold = analyzer.get_anomalies(
            col="temperature_c",
            method=method,
            threshold=5
        )
    else:
        # Using z-score or iqr with default threshold
        anomalies, threshold = analyzer.get_anomalies(
            col="temperature_c",
            method=method,
        )

    # Verify anomalies were detected
    assert not anomalies.empty, "No anomalies detected when anomalies should be present"

    # Verify anomalies dataframe has required structure
    assert "score" in anomalies.columns, "Anomalies dataframe missing 'score' column"

    # Verify the artificial temperature spike (100°C) was detected
    assert anomalies["temperature_c"].max() > 50, (
        "Temperature spike not detected - max anomalous temperature too low"
    )

    # Verify threshold is reasonable
    assert threshold > 0, "Threshold should be positive"

@pytest.mark.parametrize("lag", [0,6,12,24])
def test_correlation_matrix(sample_dataframe: pd.DataFrame, lag: int) -> None:
    # Initialize analyzer with sample data
    analyzer = WeatherAnalyzer(sample_dataframe)

    # Calculate Pearson correlation matrix
    matrix = analyzer.get_correlation_matrix(method="pearson", lag=lag,
                                             lag_column="temperature_c_filled")

    # Verify matrix dimensions (3 variables: temperature, humidity, pressure)
    assert matrix.shape == (3, 3), f"Expected 3x3 matrix, got {matrix.shape}"

    # Verify diagonal elements are 1.0 (perfect self-correlation)
    diagonal_values = np.diag(matrix)
    assert np.allclose(diagonal_values, 1.0), (
        f"Diagonal should be 1.0, got {diagonal_values}"
    )

    # Verify matrix is pandas DataFrame with proper structure
    assert isinstance(matrix, pd.DataFrame), "Correlation matrix should be DataFrame"


def test_moving_averages(sample_dataframe: pd.DataFrame) -> None:
    # Initialize analyzer with sample data
    analyzer = WeatherAnalyzer(sample_dataframe)

    # Calculate 6-hour moving average for filled temperature data
    result = analyzer.calculate_moving_averages(
        col="temperature_c_filled",
        windows=[6]  # 6-hour window
    )

    # Verify result structure contains expected window
    assert "6hour" in result, "Expected '6hour' key in moving average results"

    # Extract moving average series
    ma_series = result["6hour"]["temperature_c_filled"]

    # Verify data type and structure
    assert isinstance(ma_series, pd.Series), (
        "Moving average should return pandas Series"
    )

    # Verify data completeness (moving average should fill most points)
    missing_count = ma_series.isna().sum()
    total_points = len(ma_series)
    assert missing_count < total_points, (
        f"Too many missing values in moving average: {missing_count}/{total_points}"
    )


def test_get_trends_structure(sample_dataframe: pd.DataFrame) -> None:
    # Initialize analyzer with sample data
    analyzer = WeatherAnalyzer(sample_dataframe)

    # Perform trend analysis on temperature data
    trends = analyzer.get_trends("temperature_c")

    # Verify top-level structure
    assert "patterns" in trends, "Trends result missing 'patterns' key"
    assert "extremes" in trends, "Trends result missing 'extremes' key"

    # Verify patterns structure
    assert isinstance(trends["patterns"], list), (
        "'patterns' should be a list of cycle analyses"
    )

    # Verify extremes structure contains global statistics
    assert "global_min" in trends["extremes"], (
        "Extremes should contain 'global_min' information"
    )


def test_fit_distributions_valid(sample_dataframe: pd.DataFrame) -> None:
    # Initialize analyzer with sample data
    analyzer = WeatherAnalyzer(sample_dataframe)

    # Fit statistical distributions to temperature data
    fits = analyzer.fit_distributions("temperature_c")

    # Verify return type and non-empty results
    assert isinstance(fits, list), "Distribution fits should return a list"
    assert len(fits) > 0, "Should successfully fit at least one distribution"

    # Verify structure of first (best) fit result
    dist_name, ks_stat, p_value, params = fits[0]

    # Verify data types of fit components
    assert isinstance(dist_name, str), (
        f"Distribution name should be string, got {type(dist_name)}"
    )
    assert isinstance(params, tuple), (
        f"Distribution parameters should be tuple, got {type(params)}"
    )
    assert isinstance(ks_stat, (int, float)), (
        f"KS statistic should be numeric, got {type(ks_stat)}"
    )
    assert isinstance(p_value, (int, float)), (
        f"P-value should be numeric, got {type(p_value)}"
    )

    # Verify statistical validity
    assert 0 <= ks_stat <= 1, f"KS statistic should be between 0 and 1, got {ks_stat}"
    assert 0 <= p_value <= 1, f"P-value should be between 0 and 1, got {p_value}"


@pytest.mark.parametrize("parameter", [
    "temperature_c",
    "humidity_percent",
    "air_pressure_hpa"
])
def test_identify_cycles(sample_dataframe: pd.DataFrame, parameter: str) -> None:
    # Initialize analyzer with sample data containing known anomalies
    analyzer = WeatherAnalyzer(sample_dataframe)

    # run Fourier analysis to get periods
    results = analyzer.identify_cycles(sample_dataframe[parameter])

    # Verify return type and structure
    assert isinstance(results, pd.DataFrame), "Should be DataFrame"
    assert not results.empty, "No cycles detected when cycles should be present"

    # Verify correct periods are found
    if parameter in ["temperature_c", "humidity_percent"]:
        period = results["period_full_days_in_hours"].to_list()[0]
        assert results["period_full_days_in_hours"].to_list()[0] == 24, f"Period should be 24 hours, got {period}"
    elif parameter in ["air_pressure_hpa"]:
        period1 = results["period_full_days_in_hours"].to_list()[0]
        period2 = results["period_full_days_in_hours"].to_list()[1]
        assert results["period_full_days_in_hours"].to_list()[0] == 48, f"Period should be 48 hours, got {period1}"
        assert results["period_full_days_in_hours"].to_list()[1] == 168, f"Period should be 168 hours, got {period2}"

