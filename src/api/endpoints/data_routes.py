import logging
import json
from datetime import datetime, date

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any

from src.api.models import DataSummaryResponse, DateRange, MissingDataCount
from src.api.models import TimeSeriesResponse, TimeSeriesDataPoint
from src.api.dependencies import get_analyzer
from src.analysis_engine import WeatherAnalyzer

logger = logging.getLogger(__name__)

# Create router instance for data-related endpoints
router = APIRouter(
    prefix="/data",
    tags=["data"],
    responses={
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
        503: {"description": "Data analyzer not available"}
    }
)


@router.get("/summary", response_model=DataSummaryResponse)
async def get_data_summary(
        analyzer: WeatherAnalyzer = Depends(get_analyzer)
) -> DataSummaryResponse:
    """
    Get comprehensive summary statistics of the environmental dataset.

    This endpoint provides an overview of the dataset including:
    - Total number of records in the dataset
    - Date range (start and end dates) of the data
    - Count of missing values for each parameter
    - Basic descriptive statistics (mean, std, min, max, quartiles) for all numeric columns

    Args:
        analyzer (WeatherAnalyzer): Injected weather analysis engine dependency

    Returns:
        DataSummaryResponse: Comprehensive dataset summary containing:
            - total_records: Number of data points
            - date_range: Start and end timestamps
            - missing_data_count: Missing value counts per parameter
            - basic_statistics: Descriptive statistics for all parameters

    Raises:
        HTTPException: 500 status code if data summary generation fails

    Note:
        503 status code is automatically raised by the dependency system
        if the analyzer is not properly initialized
    """
    logger.info("Processing data summary request")

    try:
        # Generate comprehensive data summary using analyzer
        logger.debug("Retrieving dataset summary from analyzer")
        summary: Dict[str, Any] = analyzer.get_data_summary()

        logger.debug("Processing summary data for response formatting")

        # Extract and validate data information
        data_info = summary.get('data_info', {})

        # Create structured response with proper type conversion
        response = DataSummaryResponse(
            total_records=data_info['total_records'],
            date_range=DateRange(
                start=datetime.fromisoformat(data_info['date_range']['start']),
                end=datetime.fromisoformat(data_info['date_range']['end'])
            ),
            missing_data_count=MissingDataCount(**data_info['missing_data']),
            basic_statistics=summary['basic_statistics']
        )

        logger.info(
            f"Data summary generated successfully - {response.total_records} records "
            f"from {response.date_range.start} to {response.date_range.end}"
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except KeyError as e:
        logger.error(f"Missing required field in data summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Data summary missing required field: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error generating data summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate data summary: {str(e)}"
        )


@router.get("/timeseries", response_model=TimeSeriesResponse)
async def get_timeseries_data(
        start_date: Optional[date] = Query(
            None,
            description="Start date for filtering (YYYY-MM-DD format). If not provided, returns data from beginning.",
            example="2024-01-01"
        ),
        end_date: Optional[date] = Query(
            None,
            description="End date for filtering (YYYY-MM-DD format). If not provided, returns data until end.",
            example="2024-12-31"
        ),
        parameter: Optional[str] = Query(
            None,
            description="Specific parameter to filter (temperature_c, humidity_percent, air_pressure_hpa). If not provided, returns all parameters.",
            example="temperature_c"
        ),
        limit: Optional[int] = Query(
            None,
            description="Maximum number of records to return. If not provided, returns all matching records.",
            ge=1,
            le=10000,
            example=1000
        ),
        analyzer: WeatherAnalyzer = Depends(get_analyzer)
) -> TimeSeriesResponse:
    """
    Retrieve filtered environmental time series data.

    This endpoint provides access to the raw environmental sensor data with optional filtering
    by date range and specific parameters. The data includes timestamps and measurements for
    temperature, humidity, and air pressure.

    Args:
        start_date (Optional[date]): Start date for filtering data (inclusive)
        end_date (Optional[date]): End date for filtering data (inclusive)
        parameter (Optional[str]): Specific parameter to include in response
        limit (Optional[int]): Maximum number of records to return (1-10000)
        analyzer (WeatherAnalyzer): Injected weather analysis engine dependency

    Returns:
        TimeSeriesResponse: Filtered time series data containing:
            - data: List of time series data points with timestamps and measurements
            - count: Number of records returned
            - filters_applied: Dictionary showing which filters were applied

    Raises:
        HTTPException: 400 status code for invalid parameter names or date ranges
        HTTPException: 500 status code for data processing errors

    Note:
        503 status code is automatically raised by the dependency system
        if the analyzer is not properly initialized
    """
    logger.info(
        f"Processing timeseries data request - start_date: {start_date}, "
        f"end_date: {end_date}, parameter: {parameter}, limit: {limit}"
    )

    try:
        # Validate input parameters
        if start_date and end_date and start_date > end_date:
            logger.warning(f"Invalid date range: start_date {start_date} > end_date {end_date}")
            raise HTTPException(
                status_code=400,
                detail="start_date must be earlier than or equal to end_date"
            )

        # Get a copy of the dataset for filtering
        logger.debug("Retrieving dataset from analyzer")
        df = analyzer.df.copy()

        if df.empty:
            logger.warning("Dataset is empty")
            return TimeSeriesResponse(
                data=[],
                count=0,
                filters_applied={
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "parameter": parameter
                }
            )

        logger.debug(f"Original dataset size: {len(df)} records")

        # Apply date filtering
        if start_date:
            logger.debug(f"Applying start_date filter: {start_date}")
            initial_count = len(df)
            df = df[df.index.date >= start_date]
            logger.debug(f"After start_date filter: {len(df)} records (removed {initial_count - len(df)})")

        if end_date:
            logger.debug(f"Applying end_date filter: {end_date}")
            initial_count = len(df)
            df = df[df.index.date <= end_date]
            logger.debug(f"After end_date filter: {len(df)} records (removed {initial_count - len(df)})")

        # Apply parameter filtering
        available_columns = ['temperature_c', 'humidity_percent', 'air_pressure_hpa']
        if parameter:
            if parameter not in available_columns:
                logger.warning(f"Invalid parameter requested: {parameter}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid parameter '{parameter}'. Available parameters: {', '.join(available_columns)}"
                )

            logger.debug(f"Applying parameter filter: {parameter}")
            # Keep only the specified parameter column
            df = df[[col for col in df.columns if col == parameter]]

        # Convert to JSON-compatible format for consistent serialization
        logger.debug("Converting dataframe to JSON format")
        records = json.loads(df.reset_index().to_json(orient="records"))

        # Convert to response format
        logger.debug("Converting records to TimeSeriesDataPoint objects")
        data_points = []

        for record in records:
            try:
                # Create data point with proper null handling
                point = TimeSeriesDataPoint(
                    timestamp=record.get('timestamp'),
                    temperature_c=record.get('temperature_c') if 'temperature_c' in record else None,
                    humidity_percent=record.get('humidity_percent') if 'humidity_percent' in record else None,
                    air_pressure_hpa=record.get('air_pressure_hpa') if 'air_pressure_hpa' in record else None
                )
                data_points.append(point)
            except Exception as e:
                logger.warning(f"Skipping invalid record due to parsing error: {e}")
                continue

        # Create comprehensive response
        response = TimeSeriesResponse(
            data=data_points,
            count=len(data_points),
            filters_applied={
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "parameter": parameter
            }
        )

        logger.info(
            f"Timeseries data request completed successfully - returned {response.count} records"
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Unexpected error in timeseries endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve timeseries data: {str(e)}"
        )