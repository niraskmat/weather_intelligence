import base64

from src.visualization import VisualizationEngine
from src.analysis_engine import WeatherAnalyzer


def test_timeseries_plot(simple_analyzer: WeatherAnalyzer) -> None:
    # Initialize visualization engine with the test analyzer
    viz = VisualizationEngine(simple_analyzer)

    # Generate time series plot for filled temperature data
    b64_img = viz.timeseries("temperature_c_filled")

    # Validate return type and basic structure
    assert isinstance(b64_img, str), "Time series plot should return base64 string"

    # Check for valid PNG base64 encoding
    # PNG files start with specific byte sequence when base64 encoded
    png_signature_b64 = base64.b64encode(b"\x89PNG").decode()[:5]
    assert (b64_img.startswith("iVBOR") or
            b64_img.startswith(png_signature_b64)), "Should return valid PNG image data"


def test_correlation_plot(simple_analyzer: WeatherAnalyzer) -> None:
    # Initialize visualization engine
    viz = VisualizationEngine(simple_analyzer)

    # Generate correlation heatmap using Pearson method
    b64_img = viz.correlation("pearson")

    # Validate basic return characteristics
    assert isinstance(b64_img, str), "Correlation plot should return base64 string"
    assert len(b64_img) > 100, "Base64 image should contain substantial data"


def test_best_distribution_plot(simple_analyzer: WeatherAnalyzer) -> None:
    # Initialize visualization engine
    viz = VisualizationEngine(simple_analyzer)

    # Perform distribution fitting and generate visualization
    dist_name, b64_img = viz.best_distribution("temperature_c")

    # Validate distribution identification
    assert isinstance(dist_name, str), "Distribution name should be string"
    assert len(dist_name) > 0, "Distribution name should not be empty"

    # Validate image generation
    assert isinstance(b64_img, str), "Distribution plot should return base64 string"
    assert len(b64_img) > 100, "Base64 image should contain substantial data"


def test_decomposition_plot(simple_analyzer: WeatherAnalyzer) -> None:
    # Initialize visualization engine
    viz = VisualizationEngine(simple_analyzer)

    # Generate seasonal decomposition visualization
    # Uses pre-computed STL decomposition results from analyzer
    b64_img = viz.decomposition("temperature_c_filled")

    # Validate successful image generation
    assert isinstance(b64_img, str), "Decomposition plot should return base64 string"
    assert len(b64_img) > 100, "Base64 image should contain substantial data"
