"""
Pydantic Models for Weather Data Intelligence Platform API

This module defines all the data models used throughout the Weather Data Intelligence Platform.
It includes request/response models, enums, and data structures for environmental sensor data
analysis and visualization.

The models provide:
- Type validation and serialization via Pydantic
- API request/response structure definition
- Enum definitions for controlled vocabularies
- Example data for API documentation
- Clear field descriptions and constraints
"""

from pydantic import BaseModel, Field, confloat
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timedelta
from enum import Enum


# ============ ENUMS ============

class CorrelationMethod(str, Enum):
    """
    Statistical correlation methods supported by the analysis engine.

    - PEARSON: Linear correlation coefficient (assumes normal distribution)
    - SPEARMAN: Rank-based correlation (non-parametric, robust to outliers)
    - KENDALL: Tau correlation (robust, good for small samples)
    """
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class AnomalyMethod(str, Enum):
    """
    Anomaly detection methods for identifying outliers in time series data.

    - MAD: Median Absolute Deviation (robust to outliers)
    - ZSCORE: Z-score method (assumes normal distribution)
    - IQR: Interquartile Range method (quartile-based)
    """
    MAD = "mad"
    ZSCORE = "z_score"
    IQR = "iqr"


class ParameterType(str, Enum):
    """
    Environmental sensor parameter types available in the dataset.

    Raw parameters contain original sensor readings with potential anomalies and missing values.
    Filled parameters are processed versions with anomalies removed and missing data imputed.
    """
    # Raw sensor parameters
    TEMPERATURE = "temperature_c"
    HUMIDITY = "humidity_percent"
    PRESSURE = "air_pressure_hpa"

    # Processed parameters (anomalies removed, missing data filled)
    TEMPERATURE_filled = "temperature_c_filled"
    HUMIDITY_filled = "humidity_percent_filled"
    PRESSURE_filled = "air_pressure_hpa_filled"


class VisualizationType(str, Enum):
    """
    Chart types supported by the visualization engine.

    Each type generates specific visualizations for environmental data analysis.
    """
    TIMESERIES = "timeseries"  # Time series plots with moving averages
    CORRELATION = "correlation"  # Correlation heatmaps between variables
    DISTRIBUTION = "distribution"  # Statistical distribution fitting
    DECOMPOSITION = "decomposition"  # Seasonal decomposition analysis
    ANOMALY = "anomaly_detection"  # Anomaly detection visualization


# ============ BASIC DATA MODELS ============

class TimeSeriesDataPoint(BaseModel):
    """
    Single environmental sensor reading with timestamp.

    Represents one hour of environmental measurements from all sensors.
    Missing values are represented as None when sensors malfunction.
    """
    timestamp: datetime = Field(..., description="ISO timestamp of the measurement")
    temperature_c: Optional[float] = Field(None, description="Temperature in Celsius", ge=-50, le=100)
    humidity_percent: Optional[float] = Field(None, description="Relative humidity percentage", ge=0, le=150)
    air_pressure_hpa: Optional[float] = Field(None, description="Air pressure in hectopascals", ge=800, le=1200)

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-07-15T14:30:00Z",
                "temperature_c": 23.5,
                "humidity_percent": 65.2,
                "air_pressure_hpa": 1013.25
            }
        }


class DateRange(BaseModel):
    """
    Date range specification for filtering and analysis.

    Used to define time periods for data queries and analysis windows.
    """
    start: datetime = Field(..., description="Start date and time (inclusive)")
    end: datetime = Field(..., description="End date and time (inclusive)")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-12-31T23:59:59Z"
            }
        }


class MissingDataCount(BaseModel):
    """
    Count of missing data points for each environmental parameter.

    Tracks data quality by counting how many measurements are missing
    due to sensor malfunctions or data transmission issues.
    """
    temperature_c: int = Field(..., ge=0, description="Number of missing temperature readings")
    humidity_percent: int = Field(..., ge=0, description="Number of missing humidity readings")
    air_pressure_hpa: int = Field(..., ge=0, description="Number of missing pressure readings")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "temperature_c": 175,
                "humidity_percent": 180,
                "air_pressure_hpa": 165
            }
        }


class SeasonExtremes(BaseModel):
    """
    Extreme values (min/max) for a specific season with timestamps.

    Captures the most extreme environmental conditions during each season,
    useful for understanding seasonal patterns and climate extremes.
    """
    max: float = Field(..., description="Maximum value recorded in the season")
    max_time: datetime = Field(..., description="Timestamp when maximum occurred")
    min: float = Field(..., description="Minimum value recorded in the season")
    min_time: datetime = Field(..., description="Timestamp when minimum occurred")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "max": 35.7,
                "max_time": "2024-07-15T15:00:00Z",
                "min": -12.3,
                "min_time": "2024-01-20T06:00:00Z"
            }
        }


class ExtremesData(BaseModel):
    """
    Comprehensive extreme value analysis across all seasons and globally.

    Provides seasonal breakdown of extreme values plus overall dataset extremes
    for understanding climate patterns and variability.
    """
    spring: Optional[SeasonExtremes] = Field(None, description="Spring season extremes (Mar-May)")
    summer: Optional[SeasonExtremes] = Field(None, description="Summer season extremes (Jun-Aug)")
    fall: Optional[SeasonExtremes] = Field(None, description="Fall season extremes (Sep-Nov)")
    winter: Optional[SeasonExtremes] = Field(None, description="Winter season extremes (Dec-Feb)")
    global_max: float = Field(..., description="Highest value in entire dataset")
    global_max_time: datetime = Field(..., description="Timestamp of global maximum")
    global_min: float = Field(..., description="Lowest value in entire dataset")
    global_min_time: datetime = Field(..., description="Timestamp of global minimum")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "spring": {
                    "max": 28.5,
                    "max_time": "2024-05-20T14:00:00Z",
                    "min": 2.1,
                    "min_time": "2024-03-05T06:00:00Z"
                },
                "global_max": 35.7,
                "global_max_time": "2024-07-15T15:00:00Z",
                "global_min": -15.3,
                "global_min_time": "2024-01-20T06:00:00Z"
            }
        }


class DecompositionStats(BaseModel):
    """
    Statistical analysis of time series decomposition components.

    Provides statistical measures of cyclical patterns identified through
    STL (Seasonal and Trend decomposition using Loess) analysis.
    Includes timing statistics for peaks/valleys and amplitude measures.
    """
    mean_amplitude: float = Field(..., description="Average amplitude of cycles")
    mean_peak_time_day: timedelta = Field(..., description="Average time of daily peaks. In time since start of day.")
    mean_peak_time_week: timedelta = Field(..., description="Average time of weekly peaks. In time since start of week.")
    mean_peak_time_year: timedelta = Field(..., description="Average time of yearly peaks. In time since start of year.")
    mean_valley_time_day: timedelta = Field(..., description="Average time of daily valleys. In time since start of day.")
    mean_valley_time_week: timedelta = Field(..., description="Average time of weekly valleys. In time since start of week.")
    mean_valley_time_year: timedelta = Field(..., description="Average time of yearly valleys. In time since start of year.")
    std_amplitude: float = Field(..., description="Standard deviation of cycle amplitudes")
    std_peak_time_day: timedelta = Field(..., description="Standard deviation of daily peak timing")
    std_peak_time_week: timedelta = Field(..., description="Standard deviation of weekly peak timing")
    std_peak_time_year: timedelta = Field(..., description="Standard deviation of yearly peak timing")
    std_valley_time_day: timedelta = Field(..., description="Standard deviation of daily valley timing")
    std_valley_time_week: timedelta = Field(..., description="Standard deviation of weekly valley timing")
    std_valley_time_year: timedelta = Field(..., description="Standard deviation of yearly valley timing")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "mean_amplitude": 8.5,
                "mean_peak_time_day": "PT14H30M",  # 14:30 in ISO 8601 duration
                "mean_valley_time_day": "PT6H0M",  # 06:00 in ISO 8601 duration
                "std_amplitude": 2.1,
                "std_peak_time_day": "PT2H15M",  # ±2h 15m variation
                "std_valley_time_day": "PT1H45M"  # ±1h 45m variation
            }
        }


class PatternData(BaseModel):
    """
    Cyclical pattern analysis results for specific time periods.

    Contains analysis of repeating patterns in environmental data,
    such as daily temperature cycles or weekly pressure variations.
    """
    cycles: int = Field(..., ge=1, description="Number of complete cycles identified")
    period: int = Field(..., ge=1, description="Period length in hours")
    decomposition: DecompositionStats = Field(..., description="Statistics from decomposed seasonal component")
    raw_data: DecompositionStats = Field(..., description="Statistics from raw time series data")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "cycles": 365,
                "period": 24,
                "decomposition": {
                    "mean_amplitude": 8.5,
                    "mean_peak_time_day": "PT14H30M",
                    "std_amplitude": 2.1
                },
                "raw_data": {
                    "mean_amplitude": 9.2,
                    "mean_peak_time_day": "PT14H45M",
                    "std_amplitude": 2.8
                }
            }
        }


class CorrelationData(BaseModel):
    """
    Correlation coefficient between two environmental variables.

    Represents the statistical relationship between pairs of environmental
    parameters using various correlation methods.
    """
    variable1: str = Field(..., description="First variable name")
    variable2: str = Field(..., description="Second variable name")
    correlation: float = Field(..., ge=-1.0, le=1.0, description="Correlation coefficient (-1 to 1)")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "variable1": "temperature_c_filled",
                "variable2": "humidity_percent_filled",
                "correlation": -0.65
            }
        }


class AnomalyDataPoint(BaseModel):
    """
    Individual anomaly detection result with scoring.

    Represents a single data point identified as anomalous,
    including the anomaly score for ranking severity.
    """
    timestamp: datetime = Field(..., description="When the anomaly occurred")
    value: float = Field(..., description="The anomalous sensor value")
    score: float = Field(..., ge=0.0, description="Anomaly score (higher = more anomalous)")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:00:00Z",
                "value": 45.2,
                "score": 4.5
            }
        }


class AnomalyDataCollection(BaseModel):
    """
    Complete anomaly detection results for a parameter.

    Contains all anomalies detected for a specific environmental parameter,
    along with detection metadata and thresholds used.
    """
    n_anomalies: int = Field(..., ge=0, description="Total number of anomalies detected")
    threshold: float = Field(..., gt=0, description="Detection threshold used")
    anomalies: List[AnomalyDataPoint] = Field(..., description="List of detected anomalies")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "n_anomalies": 3,
                "threshold": 3.0,
                "anomalies": [
                    {
                        "timestamp": "2024-01-15T14:00:00Z",
                        "value": 45.2,
                        "score": 4.5
                    },
                    {
                        "timestamp": "2024-03-22T09:30:00Z",
                        "value": -25.1,
                        "score": 3.8
                    }
                ]
            }
        }


# ============ REQUEST MODELS ============

class CorrelationRequest(BaseModel):
    """
    Request parameters for correlation analysis between environmental variables.

    Allows selection of different correlation methods and time lag compared to a specified variable
    distribution assumptions and analysis requirements.
    """
    method: CorrelationMethod = Field(
        default=CorrelationMethod.PEARSON,
        description="Statistical correlation method to use"
    )
    lag: int = Field(
        default=0,
        description="Lag in hours"
    )
    lag_column: str = Field(
        default=None,
        description="variable to fix while lagging other"
    )
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "method": "pearson",
                "lag": 0,
                "lag_column": "none"
            },
            "examples": {
                "pearson_correlation": {
                    "summary": "Pearson correlation analysis",
                    "description": "Linear correlation assuming normal distribution",
                    "value": {"method": "pearson"}
                },
                "spearman_correlation": {
                    "summary": "Spearman rank correlation with time lag",
                    "description": "Non-parametric correlation robust to outliers",
                    "value": {"method": "spearman", "lag": 6, "lag_column": "temperature_c_filled"}
                },
                "kendall_correlation": {
                    "summary": "Kendall tau correlation",
                    "description": "Robust correlation good for small samples",
                    "value": {"method": "kendall"}
                }
            }
        }


# Custom constraint for anomaly detection thresholds
AnomalyThreshold = confloat(ge=1.0, le=10.0)


class AnomalyRequest(BaseModel):
    """
    Request parameters for anomaly detection in environmental sensor data.

    Configures detection method, thresholds, and which parameters to analyze.
    Supports per-parameter threshold customization for different sensor types.
    """
    method: AnomalyMethod = Field(
        default=AnomalyMethod.MAD,
        description="Anomaly detection algorithm to use"
    )
    thresholds: Optional[Dict[ParameterType, AnomalyThreshold]] = Field(
        default=None,
        description="Custom detection thresholds per parameter (1.0-10.0). If not provided, auto-calculated."
    )
    parameters: Optional[List[ParameterType]] = Field(
        default=None,
        description="Environmental parameters to analyze. If not provided, analyzes all parameters."
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "method": "mad",
                "thresholds": {
                    "temperature_c": 3.0,
                    "humidity_percent": 2.5
                },
                "parameters": ["temperature_c", "humidity_percent"]
            },
            "examples": {
                "basic_detection": {
                    "summary": "Basic anomaly detection",
                    "description": "Use MAD method with auto-calculated thresholds on all parameters",
                    "value": {"method": "mad"}
                },
                "custom_temperature": {
                    "summary": "Temperature-focused analysis",
                    "description": "Z-score method with custom threshold for temperature only",
                    "value": {
                        "method": "z_score",
                        "thresholds": {"temperature_c": 3.0},
                        "parameters": ["temperature_c"]
                    }
                },
                "sensitive_detection": {
                    "summary": "High sensitivity detection",
                    "description": "Lower thresholds to catch more subtle anomalies",
                    "value": {
                        "method": "mad",
                        "thresholds": {
                            "temperature_c": 2.0,
                            "humidity_percent": 2.0,
                            "air_pressure_hpa": 2.5
                        }
                    }
                }
            }
        }


# ============ RESPONSE MODELS ============

class HealthResponse(BaseModel):
    """
    API health check response indicating system status.

    Provides real-time status of the analysis engine and visualization
    components to ensure the system is ready to process requests.
    """
    status: str = Field(..., description="Overall system health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    analyzer_ready: bool = Field(..., description="Whether the analysis engine is initialized")
    viz_engine_ready: bool = Field(..., description="Whether the visualization engine is ready")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-07-01T12:00:00Z",
                "analyzer_ready": True,
                "viz_engine_ready": True
            }
        }


class DataSummaryResponse(BaseModel):
    """
    Comprehensive summary of the environmental sensor dataset.

    Provides overview statistics, data quality metrics, and basic
    descriptive statistics for all environmental parameters.
    """
    total_records: int = Field(..., ge=0, description="Total number of data records")
    date_range: DateRange = Field(..., description="Time span of the dataset")
    missing_data_count: MissingDataCount = Field(..., description="Missing data statistics")
    basic_statistics: Dict[str, Any] = Field(..., description="Descriptive statistics for all parameters")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "total_records": 8760,
                "date_range": {
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-12-31T23:00:00Z"
                },
                "missing_data_count": {
                    "temperature_c": 175,
                    "humidity_percent": 180,
                    "air_pressure_hpa": 165
                },
                "basic_statistics": {
                    "temperature_c": {
                        "count": 8585,
                        "mean": 12.5,
                        "std": 8.2,
                        "min": -15.3,
                        "25%": 6.1,
                        "50%": 12.8,
                        "75%": 19.2,
                        "max": 35.7
                    },
                    "humidity_percent": {
                        "count": 8580,
                        "mean": 65.4,
                        "std": 18.7,
                        "min": 15.2,
                        "max": 95.8
                    }
                }
            }
        }


class TimeSeriesResponse(BaseModel):
    """
    Time series data response with filtering information.

    Returns environmental sensor data for specified time ranges and parameters,
    along with metadata about applied filters and result counts.
    """
    data: List[TimeSeriesDataPoint] = Field(..., description="Time series data points")
    count: int = Field(..., ge=0, description="Number of data points returned")
    filters_applied: Dict[str, Any] = Field(..., description="Summary of filters applied to the data")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "temperature_c": -9.63,
                        "humidity_percent": 79.3,
                        "air_pressure_hpa": 1025.56
                    },
                    {
                        "timestamp": "2024-01-01T01:00:00Z",
                        "temperature_c": -8.92,
                        "humidity_percent": 78.1,
                        "air_pressure_hpa": 1025.21
                    }
                ],
                "count": 2,
                "filters_applied": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-01",
                    "parameter": None
                }
            }
        }


class CorrelationResponse(BaseModel):
    """
    Statistical correlation analysis results between environmental variables.

    Contains correlation matrix showing relationships between all pairs
    of environmental parameters using the specified method.
    """
    correlation_matrix: List[CorrelationData] = Field(
        ...,
        description="Pairwise correlations between all variables"
    )
    method_used: str = Field(..., description="Correlation method that was applied")
    lag: int = Field(..., description="Lag in hours used")
    lag_column: str = Field(..., description="fixed variable")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "correlation_matrix": [
                    {
                        "variable1": "temperature_c_filled",
                        "variable2": "humidity_percent_filled",
                        "correlation": -0.65
                    },
                    {
                        "variable1": "temperature_c_filled",
                        "variable2": "air_pressure_hpa_filled",
                        "correlation": 0.23
                    },
                    {
                        "variable1": "humidity_percent_filled",
                        "variable2": "air_pressure_hpa_filled",
                        "correlation": -0.12
                    }
                ],
                "method_used": "pearson",
                "lag": "6",
                "lag_column": "temperature_c_filled"
            }
        }


class AnomalyResponse(BaseModel):
    """
    Comprehensive anomaly detection results for environmental parameters.

    Returns detected anomalies for each analyzed parameter, including
    detection metadata and anomaly scoring information.
    """
    method_used: str = Field(..., description="Anomaly detection method that was applied")
    parameters: Dict[ParameterType, AnomalyDataCollection] = Field(
        ...,
        description="Anomaly detection results for each parameter"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "method_used": "mad",
                "parameters": {
                    "temperature_c": {
                        "n_anomalies": 3,
                        "threshold": 3.0,
                        "anomalies": [
                            {
                                "timestamp": "2024-01-15T14:00:00Z",
                                "value": 45.2,
                                "score": 4.5
                            },
                            {
                                "timestamp": "2024-03-22T09:30:00Z",
                                "value": -25.1,
                                "score": 3.8
                            }
                        ]
                    },
                    "humidity_percent": {
                        "n_anomalies": 1,
                        "threshold": 2.5,
                        "anomalies": [
                            {
                                "timestamp": "2024-06-10T16:20:00Z",
                                "value": 105.3,
                                "score": 6.2
                            }
                        ]
                    }
                }
            }
        }


class TrendResponse(BaseModel):
    """
    Comprehensive trend analysis results for environmental parameters.

    Includes cyclical pattern analysis, seasonal extremes, and statistical
    characterization of temporal patterns in the data.
    """
    extremes: ExtremesData = Field(..., description="Seasonal and global extreme values")
    patterns: List[PatternData] = Field(..., description="Identified cyclical patterns and their statistics")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "extremes": {
                    "spring": {
                        "max": 28.5,
                        "max_time": "2024-05-20T14:00:00Z",
                        "min": 2.1,
                        "min_time": "2024-03-05T06:00:00Z"
                    },
                    "global_max": 35.7,
                    "global_max_time": "2024-07-15T15:00:00Z",
                    "global_min": -15.3,
                    "global_min_time": "2024-01-20T06:00:00Z"
                },
                "patterns": [
                    {
                        "cycles": 365,
                        "period": 24,
                        "decomposition": {
                            "mean_amplitude": 8.5,
                            "mean_peak_time_day": "PT14H30M"
                        },
                        "raw_data": {
                            "mean_amplitude": 9.2,
                            "mean_peak_time_day": "PT14H45M"
                        }
                    }
                ]
            }
        }


# ============ VISUALIZATION MODELS ============

class BaseVisualizationResponse(BaseModel):
    """
    Base class for all visualization responses.

    Provides common structure for visualization endpoints with chart type identification.
    """
    chart_type: VisualizationType = Field(..., description="Type of visualization generated")


class TimeSeriesVisualizationResponse(BaseVisualizationResponse):
    """
    Time series visualization response with base64-encoded plot images.

    Contains time series plots for environmental parameters with moving averages
    and trend analysis. Images are provided as base64-encoded PNG data.
    """
    chart_type: Literal[VisualizationType.TIMESERIES] = VisualizationType.TIMESERIES
    images: Dict[str, str] = Field(
        ...,
        description="Base64-encoded PNG images keyed by parameter name"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chart_type": "timeseries",
                "images": {
                    "temperature_c_filled": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "humidity_percent_filled": "iVBORw0KGgoAAAANSUhEUgAA..."
                }
            }
        }


class CorrelationMatrixVisualizationResponse(BaseVisualizationResponse):
    """
    Correlation matrix heatmap visualization response.

    Contains correlation heatmaps for different correlation methods and lags
    showing relationships between environmental parameters.
    """
    chart_type: Literal[VisualizationType.CORRELATION] = VisualizationType.CORRELATION
    images: Dict[str, Dict[int, Dict[str, str]]] = Field(
        ...,
        description="Base64-encoded PNG heatmaps keyed by correlation method"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chart_type": "correlation",
                "images": {
                    "pearson": {
                        "0": {
                            "none": "iVBORw0KGgoAAAANSUhEUgAA..."
                        }
                    },
                    "spearman": {
                        "0": {
                            "none": "iVBORw0KGgoAAAANSUhEUgAA..."
                        }
                    },
                    "kendall": {
                        "0": {
                            "none": "iVBORw0KGgoAAAANSUhEUgAA..."
                        },
                        "6": {
                            "temperature_c_filled": "iVBORw0KGgoAAAANSUhEUgAA...",
                            "humidity_percent_filled": "iVBORw0KGgoAAAANSUhEUgAA...",
                            "air_pressure_hpa_filled": "iVBORw0KGgoAAAANSUhEUgAA..."
                        }
                    }
                }
            }
        }


class DistributionVisualizationResponse(BaseVisualizationResponse):
    """
    Statistical distribution analysis visualization response.

    Contains distribution fitting plots showing how well different statistical
    distributions match the environmental parameter data.
    """
    chart_type: Literal[VisualizationType.DISTRIBUTION] = VisualizationType.DISTRIBUTION
    images: Dict[str, Dict[str, str]] = Field(
        ...,
        description="Distribution plots and metadata keyed by parameter name"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chart_type": "distribution",
                "images": {
                    "temperature_c_filled": {
                        "image": "iVBORw0KGgoAAAANSUhEUgAA...",
                        "best_fit": "norm"
                    },
                    "humidity_percent_filled": {
                        "image": "iVBORw0KGgoAAAANSUhEUgAA...",
                        "best_fit": "beta"
                    }
                }
            }
        }


class DecompositionVisualizationResponse(BaseVisualizationResponse):
    """
    Time series decomposition visualization response.

    Contains seasonal decomposition plots showing trend, seasonal,
    and residual components of environmental time series data.
    """
    chart_type: Literal[VisualizationType.DECOMPOSITION] = VisualizationType.DECOMPOSITION
    images: Dict[str, str] = Field(
        ...,
        description="Base64-encoded decomposition plots keyed by parameter name"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chart_type": "decomposition",
                "images": {
                    "temperature_c_filled": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "humidity_percent_filled": "iVBORw0KGgoAAAANSUhEUgAA..."
                }
            }
        }


class AnomalyVisualizationResponse(BaseVisualizationResponse):
    """
    Anomaly detection visualization response.

    Contains time series plots highlighting detected anomalies in
    environmental sensor data with visual emphasis on outlier points.
    """
    chart_type: Literal[VisualizationType.ANOMALY] = VisualizationType.ANOMALY
    images: Dict[str, str] = Field(
        ...,
        description="Base64-encoded anomaly detection plots keyed by parameter name"
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "chart_type": "anomaly_detection",
                "images": {
                    "temperature_c": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "humidity_percent": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "air_pressure_hpa": "iVBORw0KGgoAAAANSUhEUgAA..."
                }
            }
        }