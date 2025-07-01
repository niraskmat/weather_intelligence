from fastapi import APIRouter, Depends, HTTPException
from typing import List
import logging

from src.api.dependencies import get_analyzer
from src.analysis_engine import WeatherAnalyzer
from src.api.models import (
    CorrelationData,
    CorrelationResponse,
    CorrelationRequest,
    AnomalyDataCollection,
    AnomalyDataPoint,
    AnomalyRequest,
    AnomalyResponse,
    TrendResponse
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
    responses={
        500: {"description": "Internal server error during analysis"},
        503: {"description": "Analysis engine not available"}
    }
)


@router.post(
    "/correlation",
    response_model=CorrelationResponse,
    summary="Perform correlation analysis between weather variables",
    description="""
    Calculate correlation coefficients between all pairs of weather variables
    using the specified statistical method (Pearson, Spearman, or Kendall).

    Returns a correlation matrix showing relationships between:
    - Temperature vs Humidity
    - Temperature vs Air Pressure  
    - Humidity vs Air Pressure
    """
)
async def analyze_correlations(
        request: CorrelationRequest,
        analyzer: WeatherAnalyzer = Depends(get_analyzer)
) -> CorrelationResponse:
    """
    Perform correlation analysis between environmental variables.

    This endpoint calculates pairwise correlations between all numeric weather
    parameters using the specified correlation method. The analysis helps identify
    relationships and dependencies between different environmental factors.

    Args:
        request (CorrelationRequest): Configuration specifying correlation method
        analyzer (WeatherAnalyzer): Injected weather data analysis engine

    Returns:
        CorrelationResponse: Correlation matrix with pairwise correlation coefficients

    Raises:
        HTTPException: 500 if correlation calculation fails
        HTTPException: 503 if analyzer is not initialized
    """
    logger.info(f"Starting correlation analysis using method: {request.method}")

    try:
        # Calculate correlation matrix using the specified method
        logger.debug("Computing correlation matrix from analyzer")
        correlation_results = analyzer.get_correlation_matrix(method=request.method,
                                                              lag=request.lag,
                                                              lag_column=request.lag_column)

        logger.debug(f"Correlation matrix shape: {correlation_results.shape}")
        logger.debug(f"Correlation matrix:\n{correlation_results}")

        # Transform correlation matrix into API response format
        correlation_data = []

        # Iterate through all variable pairs in the correlation matrix
        for var1 in correlation_results.index:
            for var2 in correlation_results.columns:
                correlation_value: float = correlation_results.loc[var1, var2]

                # Create correlation data point for this variable pair
                correlation_data.append(
                    CorrelationData(
                        variable1=var1,
                        variable2=var2,
                        correlation=correlation_value
                    )
                )

                logger.debug(f"Correlation {var1} vs {var2}: {correlation_value:.4f}")

        logger.info(f"Successfully calculated {len(correlation_data)} correlation pairs")

        # Construct and return the response
        response = CorrelationResponse(
            correlation_matrix=correlation_data,
            method_used=request.method,
            lag=request.lag,
            lag_column=request.lag_column
        )

        logger.info(f"Correlation analysis completed successfully using {request.method} method")
        return response

    except Exception as e:
        error_msg = f"Failed to perform correlation analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post(
    "/anomalies",
    response_model=AnomalyResponse,
    summary="Detect anomalies in environmental sensor data",
    description="""
    Identify statistical anomalies in weather data using various detection methods.

    Supports multiple detection algorithms:
    - MAD (Median Absolute Deviation) - robust to outliers
    - Z-Score - standard deviation based detection

    Returns detected anomalies with timestamps, values, and anomaly scores.
    """
)
async def detect_anomalies(
        request: AnomalyRequest,
        analyzer: WeatherAnalyzer = Depends(get_analyzer)
) -> AnomalyResponse:
    """
    Detect statistical anomalies in environmental sensor data.

    This endpoint identifies data points that deviate significantly from normal
    patterns using various statistical and machine learning methods. Anomalies
    could indicate sensor malfunctions, extreme weather events, or data quality issues.

    Args:
        request (AnomalyRequest): Anomaly detection configuration including:
            - method: Statistical method to use (mad, z_score)
            - thresholds: Custom thresholds per parameter (optional)
            - parameters: Specific parameters to analyze (optional, defaults to all)

        analyzer (WeatherAnalyzer): Injected weather data analysis engine

    Returns:
        AnomalyResponse: Detected anomalies grouped by parameter with scores

    Raises:
        HTTPException: 500 if anomaly detection fails
        HTTPException: 503 if analyzer is not initialized
    """
    logger.info(f"Starting anomaly detection using method: {request.method}")

    try:
        # Set default parameters if none specified
        parameters: List[str] = request.parameters
        if parameters is None:
            parameters = ["temperature_c", "humidity_percent", "air_pressure_hpa"]
            logger.debug("Using default parameters for anomaly detection")

        logger.debug(f"Analyzing parameters: {parameters}")

        # Set default thresholds if none specified
        thresholds = request.thresholds
        if thresholds is None:
            thresholds = {}

        # Ensure all parameters have threshold entries (None = auto-calculate)
        for parameter in parameters:
            if parameter not in thresholds:
                thresholds[parameter] = None

        logger.debug(f"Using thresholds: {thresholds}")

        # Process each parameter for anomaly detection
        anomaly_results = {}

        for col in parameters:
            logger.info(f"Detecting anomalies in parameter: {col}")

            try:
                # Get anomalies for this specific parameter
                anomaly_df, threshold_used = analyzer.get_anomalies(
                    col,
                    method=request.method,
                    threshold=thresholds[col]
                )

                logger.debug(f"Found {len(anomaly_df)} anomalies in {col} with threshold {threshold_used:.4f}")

                # Convert anomaly dataframe to API response format
                detected_anomalies = []

                for _, row in anomaly_df.reset_index().iterrows():
                    anomaly_point = AnomalyDataPoint(
                        timestamp=row['timestamp'],
                        value=row[col],
                        score=row['score']
                    )
                    detected_anomalies.append(anomaly_point)

                    logger.debug(f"Anomaly at {row['timestamp']}: value={row[col]:.2f}, score={row['score']:.4f}")

                # Create anomaly collection for this parameter
                anomaly_collection = AnomalyDataCollection(
                    n_anomalies=len(detected_anomalies),
                    threshold=threshold_used,
                    anomalies=detected_anomalies
                )

                anomaly_results[col] = anomaly_collection
                logger.info(f"Successfully processed {len(detected_anomalies)} anomalies for {col}")

            except Exception as param_error:
                logger.warning(f"Failed to detect anomalies in {col}: {str(param_error)}")
                # Continue with other parameters rather than failing completely
                continue

        # Ensure we have results for at least one parameter
        if not anomaly_results:
            error_msg = "No anomaly results generated for any requested parameters"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Calculate total anomalies across all parameters
        total_anomalies = sum(collection.n_anomalies for collection in anomaly_results.values())

        logger.info(
            f"Anomaly detection completed: {total_anomalies} total anomalies across {len(anomaly_results)} parameters")

        # Construct and return the response
        response = AnomalyResponse(
            method_used=request.method,
            parameters=anomaly_results
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        error_msg = f"Failed to detect anomalies: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get(
    "/trends/{parameter}",
    response_model=TrendResponse,
    summary="Analyze trends and patterns for a specific weather parameter",
    description="""
    Perform trend analysis for a single weather parameter including:

    - Seasonal decomposition to identify cyclical patterns
    - Extreme value analysis (min/max per season and globally)
    - Peak and valley timing analysis with statistical uncertainty
    - Multi-scale pattern recognition (daily, weekly, yearly cycles)

    Supported parameters: temperature_c, humidity_percent, air_pressure_hpa
    """
)
async def get_trend_analysis(
        parameter: str,
        analyzer: WeatherAnalyzer = Depends(get_analyzer)
) -> TrendResponse:
    """
    Perform trend analysis for a specific weather parameter.

    This endpoint analyzes long-term trends, seasonal patterns, and cyclical
    behavior in environmental data. It uses seasonal decomposition techniques
    to separate trend, seasonal, and residual components, providing insights
    into recurring patterns and extreme values.

    Args:
        parameter (str): Weather parameter to analyze. Must be one of:
            - 'temperature_c': Temperature in Celsius
            - 'humidity_percent': Relative humidity percentage
            - 'air_pressure_hpa': Air pressure in hectopascals

        analyzer (WeatherAnalyzer): Injected weather data analysis engine

    Returns:
        TrendResponse: Comprehensive trend analysis including:
            - patterns: List of cyclical patterns at different time scales
            - extremes: Seasonal and global extreme values with timestamps

    Raises:
        HTTPException: 400 if parameter is invalid
        HTTPException: 500 if trend analysis fails
        HTTPException: 503 if analyzer is not initialized
    """
    logger.info(f"Starting trend analysis for parameter: {parameter}")

    # Define valid parameters for trend analysis
    valid_parameters: List[str] = ['temperature_c', 'humidity_percent', 'air_pressure_hpa']

    # Validate the requested parameter
    if parameter not in valid_parameters:
        error_msg = f"Invalid parameter '{parameter}'. Valid options: {valid_parameters}"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    logger.debug(f"Parameter '{parameter}' validated successfully")

    try:
        # Perform trend analysis using the analyzer engine
        logger.debug("Calling analyzer.get_trends() method")
        trend_results = analyzer.get_trends(parameter)

        # Log summary of trend analysis results
        if 'patterns' in trend_results:
            num_patterns = len(trend_results['patterns'])
            logger.debug(f"Identified {num_patterns} cyclical patterns")

            for i, pattern in enumerate(trend_results['patterns']):
                period_hours = pattern.get('period', 'unknown')
                cycles = pattern.get('cycles', 'unknown')
                logger.debug(f"Pattern {i + 1}: {period_hours}h period, {cycles} cycles")

        if 'extremes' in trend_results:
            extremes = trend_results['extremes']
            global_min = extremes.get('global_min', 'unknown')
            global_max = extremes.get('global_max', 'unknown')
            logger.debug(f"Global extremes: min={global_min}, max={global_max}")

        logger.info(f"Trend analysis completed successfully for {parameter}")

        # Create and return the response
        response = TrendResponse(**trend_results)
        return response

    except Exception as e:
        error_msg = f"Failed to perform trend analysis for {parameter}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)