import logging
from fastapi import HTTPException

from src.analysis_engine import WeatherAnalyzer
from src.visualization import VisualizationEngine


# Configure module logger
logger = logging.getLogger(__name__)

# ============ APPLICATION STATE MANAGEMENT ============

# Global application state dictionary containing initialized components
# This will be populated by main.py during the application startup sequence
# Structure: {
#   "analyzer": WeatherAnalyzer instance or None,
#   "visualizer": VisualizationEngine instance or None
# }
state = {"analyzer": None,
         "visualizer": None}

# ============ DEPENDENCY PROVIDER FUNCTIONS ============


async def get_analyzer() -> WeatherAnalyzer:
    """
    FastAPI dependency provider for the weather analysis engine.

    This function serves as a dependency injection provider that ensures
    the WeatherAnalyzer component is properly initialized before being
    injected into API endpoint handlers. It performs validation to prevent
    endpoints from attempting to use uninitialized components.

    Returns:
        WeatherAnalyzer: The initialized weather analysis engine instance

    Raises:
        HTTPException: 503 Service Unavailable if the analyzer is not initialized

    Example:
        ```python
        @router.get("/analysis/summary")
        async def get_summary(analyzer: WeatherAnalyzer = Depends(get_analyzer)):
            return analyzer.get_data_summary()
        ```

    Note:
        The analyzer must be initialized during application startup via the
        lifespan manager in main.py. If this dependency is called before
        initialization, it will raise a 503 error.
    """
    logger.debug("Dependency injection requested for WeatherAnalyzer")

    # Log current state for debugging purposes
    logger.debug(f"Current analyzer state: {state['analyzer'] is not None}")

    # Validate that the analyzer has been properly initialized
    if state["analyzer"] is None:
        logger.error("WeatherAnalyzer dependency requested but component not initialized")
        logger.error("This indicates the application startup sequence may have failed")
        raise HTTPException(
            status_code=503,
            detail="Data analyzer not initialized. Please check application startup logs."
        )

    logger.debug("Successfully providing WeatherAnalyzer dependency")
    return state["analyzer"]


async def get_visualizer() -> VisualizationEngine:
    """
    FastAPI dependency provider for the visualization engine.

    This function serves as a dependency injection provider that ensures
    the VisualizationEngine component is properly initialized before being
    injected into API endpoint handlers. It validates component availability
    and provides appropriate error handling for uninitialized states.

    Returns:
        VisualizationEngine: The initialized visualization engine instance

    Raises:
        HTTPException: 503 Service Unavailable if the visualizer is not initialized

    Example:
        ```python
        @router.get("/visualizations/timeseries")
        async def get_plot(visualizer: VisualizationEngine = Depends(get_visualizer)):
            return visualizer.timeseries("temperature_c")
        ```

    Note:
        The visualizer must be initialized during application startup via the
        lifespan manager in main.py. The visualizer depends on the analyzer
        being initialized first, so both components must be available.
    """
    logger.debug("Dependency injection requested for VisualizationEngine")

    # Log current state for debugging purposes
    logger.debug(f"Current visualizer state: {state['visualizer'] is not None}")

    # Validate that the visualizer has been properly initialized
    if state["visualizer"] is None:
        logger.error("VisualizationEngine dependency requested but component not initialized")
        logger.error("This may indicate analyzer initialization failed or visualizer setup error")
        raise HTTPException(
            status_code=503,
            detail="Visualizer not initialized. Please check application startup logs."
        )

    logger.debug("Successfully providing VisualizationEngine dependency")
    return state["visualizer"]