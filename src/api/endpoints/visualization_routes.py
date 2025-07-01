from fastapi import APIRouter, Depends, HTTPException
from src.api.models import (
    TimeSeriesVisualizationResponse,
    CorrelationMatrixVisualizationResponse,
    DistributionVisualizationResponse,
    DecompositionVisualizationResponse,
    AnomalyVisualizationResponse
)

from typing import Union, List
from src.visualization import VisualizationEngine
from src.api.dependencies import get_visualizer
import logging


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/visualizations",
    tags=["visualizations"],
    responses={
        400: {"description": "Visualization type not found"},
        500: {"description": "Internal server error during chart generation"}
    }
)

# Define supported weather parameters for visualization
FILLED_PARAMETERS: List[str] = [
    "temperature_c_filled",
    "humidity_percent_filled",
    "air_pressure_hpa_filled"
]

# Raw parameters (with potential missing values) for anomaly detection
RAW_PARAMETERS: List[str] = [
    "temperature_c",
    "humidity_percent",
    "air_pressure_hpa"
]

# Supported correlation analysis methods
CORRELATION_METHODS: List[str] = ["pearson", "spearman", "kendall"]

# Valid chart types supported by the visualization engine
VALID_CHART_TYPES: List[str] = [
    'timeseries',
    'correlation',
    'distribution',
    'decomposition',
    'anomaly_detection'
]


@router.get(
    "/{chart_type}",
    response_model=Union[
        TimeSeriesVisualizationResponse,
        CorrelationMatrixVisualizationResponse,
        DistributionVisualizationResponse,
        DecompositionVisualizationResponse,
        AnomalyVisualizationResponse
    ],
    summary="Generate weather data visualization",
    description="""
    Generate and return various types of weather data visualizations as base64-encoded images.

    Supported chart types:
    - **timeseries**: Time series plots with moving averages for all parameters
    - **correlation**: Correlation heatmaps using different methods (pearson, spearman, kendall)
    - **distribution**: Distribution analysis with best-fit statistical distributions
    - **decomposition**: Seasonal decomposition plots showing trend, seasonal, and residual components
    - **anomaly_detection**: Anomaly detection visualizations highlighting outliers

    All images are returned as base64-encoded PNG strings for easy web integration.
    """
)
async def get_visualization(
        chart_type: str,
        viz_engine: VisualizationEngine = Depends(get_visualizer)
) -> Union[
    TimeSeriesVisualizationResponse,
    CorrelationMatrixVisualizationResponse,
    DistributionVisualizationResponse,
    DecompositionVisualizationResponse,
    AnomalyVisualizationResponse
]:
    """
    Generate and return chart images as base64-encoded strings.

    This endpoint serves as the main visualization API, routing requests to the appropriate
    visualization generation method based on the requested chart type.

    Args:
        chart_type (str): Type of visualization to generate. Must be one of:
                         'timeseries', 'correlation', 'distribution',
                         'decomposition', 'anomaly_detection'
        viz_engine (VisualizationEngine): Injected visualization engine dependency

    Returns:
        Union[Various Response Models]: Appropriate response model containing:
            - chart_type: The type of chart generated
            - images: Dictionary of base64-encoded PNG images

    Raises:
        HTTPException: 400 if chart_type is invalid
        HTTPException: 500 if visualization generation fails
    """
    logger.info(f"Received visualization request for chart type: {chart_type}")

    try:
        # Validate chart type against supported options
        if chart_type not in VALID_CHART_TYPES:
            logger.warning(f"Invalid chart type requested: {chart_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chart type '{chart_type}'. Valid types: {VALID_CHART_TYPES}"
            )

        logger.debug(f"Generating {chart_type} visualization with {len(FILLED_PARAMETERS)} parameters")

        # Initialize response structure
        plots = {"chart_type": chart_type,
                 "images": {}}
        # Route to appropriate visualization generation method
        if chart_type == 'timeseries':
            logger.info("Generating time series plots for all filled parameters")

            # Generate time series plot for each weather parameter
            for col in FILLED_PARAMETERS:
                logger.debug(f"Creating time series plot for parameter: {col}")
                try:
                    image_b64: str = viz_engine.timeseries(col)
                    plots["images"][col] = image_b64
                    logger.debug(f"Successfully generated time series plot for {col}")
                except Exception as e:
                    logger.error(f"Failed to generate time series plot for {col}: {str(e)}")
                    raise

            logger.info(f"Generated {len(plots['images'])} time series plots successfully")
            return TimeSeriesVisualizationResponse(**plots)

        elif chart_type == 'correlation':
            logger.info("Generating correlation heatmaps for all correlation methods")

            # Generate correlation matrix for each statistical method
            for method in CORRELATION_METHODS:
                plots["images"][method] = {}
                for lag in [0, 6, 12, 24]:
                    plots["images"][method][lag] = {}
                    if lag != 0:
                        lag_columns = FILLED_PARAMETERS
                    else:
                        lag_columns = ["none"]
                    for lag_column in lag_columns:
                        logger.debug(f"Creating correlation heatmap using method: {method}")
                        try:
                            image_b64: str = viz_engine.correlation(method, lag, lag_column)
                            plots["images"][method][lag][lag_column] = image_b64
                            logger.debug(f"Successfully generated correlation heatmap for {method} with lag {lag} and fixed {lag_column}")
                        except Exception as e:
                            logger.error(f"Failed to generate correlation heatmap for {method}, {lag}, {lag_column}: {str(e)}")
                            raise

            logger.info(f"Generated correlation heatmaps successfully")
            return CorrelationMatrixVisualizationResponse(**plots)

        elif chart_type == 'distribution':
            logger.info("Generating distribution analysis plots for all parameters")

            # Generate distribution analysis for each parameter
            for col in FILLED_PARAMETERS:
                logger.debug(f"Analyzing distribution for parameter: {col}")
                try:
                    best_fit: str
                    image_b64: str
                    best_fit, image_b64 = viz_engine.best_distribution(col)
                    plots["images"][col] = {"image": image_b64, "best_fit": best_fit}
                    logger.debug(f"Best distribution for {col}: {best_fit}")
                except Exception as e:
                    logger.error(f"Failed to generate distribution plot for {col}: {str(e)}")
                    raise

            logger.info(f"Generated {len(plots['images'])} distribution plots successfully")
            return DistributionVisualizationResponse(**plots)

        elif chart_type == 'decomposition':
            logger.info("Generating seasonal decomposition plots for all parameters")

            # Generate seasonal decomposition for each parameter
            for col in FILLED_PARAMETERS:
                logger.debug(f"Creating seasonal decomposition plot for parameter: {col}")
                try:
                    image_b64: str = viz_engine.decomposition(col)
                    plots["images"][col] = image_b64
                    logger.debug(f"Successfully generated decomposition plot for {col}")
                except Exception as e:
                    logger.error(f"Failed to generate decomposition plot for {col}: {str(e)}")
                    raise

            logger.info(f"Generated {len(plots['images'])} decomposition plots successfully")
            return DecompositionVisualizationResponse(**plots)

        elif chart_type == 'anomaly_detection':
            logger.info("Generating anomaly detection plots for all raw parameters")

            # Generate anomaly detection plots using raw (non-filled) data
            for col in RAW_PARAMETERS:
                logger.debug(f"Creating anomaly detection plot for parameter: {col}")
                try:
                    image_b64: str = viz_engine.anomalies(col)
                    plots["images"][col] = image_b64
                    logger.debug(f"Successfully generated anomaly plot for {col}")
                except Exception as e:
                    logger.error(f"Failed to generate anomaly plot for {col}: {str(e)}")
                    raise

            logger.info(f"Generated {len(plots['images'])} anomaly detection plots successfully")
            return AnomalyVisualizationResponse(**plots)

    except HTTPException as e:
        # Re-raise HTTPExceptions as-is to maintain proper status codes
        logger.error(f"HTTP error in visualization endpoint: {e.detail}")
        raise e

    except Exception as e:
        # Log unexpected errors and return 500 status
        logger.error(f"Unexpected error in get_visualization for {chart_type}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while generating {chart_type} visualization: {str(e)}"
        )