from src.visualization import VisualizationEngine
import base64


def test_timeseries_plot(simple_analyzer):
    viz = VisualizationEngine(simple_analyzer)
    b64_img = viz.timeseries("temperature_c_filled")

    assert isinstance(b64_img, str)
    assert b64_img.startswith("iVBOR") or b64_img[:5] == base64.b64encode(b"\x89PNG").decode()[:5]


def test_correlation_plot(simple_analyzer):
    viz = VisualizationEngine(simple_analyzer)
    b64_img = viz.correlation("pearson")

    assert isinstance(b64_img, str)
    assert len(b64_img) > 100


def test_best_distribution_plot(simple_analyzer):
    viz = VisualizationEngine(simple_analyzer)
    dist_name, b64_img = viz.best_distribution("temperature_c")

    assert isinstance(dist_name, str)
    assert isinstance(b64_img, str)
    assert len(b64_img) > 100


def test_decomposition_plot(simple_analyzer):
    viz = VisualizationEngine(simple_analyzer)
    b64_img = viz.decomposition("temperature_c_filled")

    assert isinstance(b64_img, str)
    assert len(b64_img) > 100


