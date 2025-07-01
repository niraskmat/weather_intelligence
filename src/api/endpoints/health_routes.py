import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from src.api.models import HealthResponse
from src.api.dependencies import get_analyzer, get_visualizer
from src.analysis_engine import WeatherAnalyzer
from src.visualization import VisualizationEngine

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check(
        analyzer: Optional[WeatherAnalyzer] = Depends(get_analyzer),
        visualizer: Optional[VisualizationEngine] = Depends(get_visualizer)
) -> HealthResponse:
    """
    Perform system health check.

    This endpoint verifies that all critical components of the Weather Data
    Intelligence Platform are properly initialized and ready to serve requests.

    Args:
        analyzer (Optional[WeatherAnalyzer]): Weather data analysis engine.
            Injected via dependency injection. May be None if not initialized.
        visualizer (Optional[VisualizationEngine]): Data visualization engine.
            Injected via dependency injection. May be None if not initialized.

    Returns:
        HealthResponse: Structured health status including:
            - status: Overall system status ("healthy", "degraded", "unhealthy")
            - timestamp: Current system time for monitoring synchronization
            - analyzer_ready: Boolean indicating analyzer availability
            - viz_engine_ready: Boolean indicating visualizer availability

    Raises:
        HTTPException: 503 Service Unavailable if critical components are not ready
        HTTPException: 500 Internal Server Error if health check itself fails

    Example:
        >>> # Healthy system response
        {
            "status": "healthy",
            "timestamp": "2024-01-01T12:00:00.123456",
            "analyzer_ready": true,
            "viz_engine_ready": true
        }

        >>> # Degraded system response
        {
            "status": "degraded",
            "timestamp": "2024-01-01T12:00:00.123456",
            "analyzer_ready": false,
            "viz_engine_ready": true
        }
    """
    logger.info("Performing system health check")

    try:
        # Get current timestamp for response
        current_time = datetime.now()
        logger.debug(f"Health check initiated at: {current_time}")

        # Check analyzer component availability
        analyzer_ready = analyzer is not None
        if analyzer_ready:
            logger.debug("Weather analyzer engine: READY")
            # Additional check: verify analyzer has data loaded
            try:
                data_count = len(analyzer.df) if hasattr(analyzer, 'df') else 0
                logger.debug(f"Analyzer data points available: {data_count}")
                if data_count == 0:
                    logger.warning("Analyzer ready but no data loaded")
                    analyzer_ready = False
            except Exception as e:
                logger.warning(f"Error checking analyzer data: {e}")
                analyzer_ready = False
        else:
            logger.warning("Weather analyzer engine: NOT READY")

        # Check visualization engine availability
        viz_ready = visualizer is not None
        if viz_ready:
            logger.debug("Visualization engine: READY")
            # Additional check: verify visualizer has access to analyzer
            try:
                if hasattr(visualizer, 'analyzer') and visualizer.analyzer is not None:
                    logger.debug("Visualizer has valid analyzer reference")
                else:
                    logger.warning("Visualizer missing analyzer reference")
                    viz_ready = False
            except Exception as e:
                logger.warning(f"Error checking visualizer configuration: {e}")
                viz_ready = False
        else:
            logger.warning("Visualization engine: NOT READY")

        # Determine overall system status
        if analyzer_ready and viz_ready:
            status = "healthy"
            logger.info("System health check: ALL SYSTEMS OPERATIONAL")
        elif analyzer_ready or viz_ready:
            status = "degraded"
            logger.warning("System health check: DEGRADED - Some components unavailable")
        else:
            status = "unhealthy"
            logger.error("System health check: UNHEALTHY - Critical components unavailable")

        # Create health response
        health_response = HealthResponse(
            status=status,
            timestamp=current_time,
            analyzer_ready=analyzer_ready,
            viz_engine_ready=viz_ready
        )

        # Log final status
        logger.info(
            f"Health check completed - Status: {status}, "
            f"Analyzer: {'OK' if analyzer_ready else 'FAIL'}, "
            f"Visualizer: {'OK' if viz_ready else 'FAIL'}"
        )

        # Return appropriate HTTP status based on health
        if status == "unhealthy":
            logger.error("Returning HTTP 503 due to unhealthy system status")
            raise HTTPException(
                status_code=503,
                detail="Critical system components are not available"
            )
        elif status == "degraded":
            logger.warning("System degraded but continuing operation")

        return health_response

    except HTTPException:
        # Re-raise HTTP exceptions (like 503 above)
        raise

    except Exception as e:
        # Handle unexpected errors during health check
        logger.error(f"Unexpected error during health check: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Health check failed due to internal error"
        )
