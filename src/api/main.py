"""
Weather Data Intelligence Platform - Main Application Entry Point

This module serves as the main entry point for the Weather Data Intelligence Platform,
a FastAPI-based scientific data analysis pipeline for environmental sensor data.

The application provides:
- RESTful API endpoints for data access, analysis, and visualization
- Weather data processing and visualization
- Configurable logging and error handling

Author: Nicolai Rask Mathiesen
Version: 1.0.0
"""

import pandas as pd
import uvicorn
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.endpoints import analysis_routes, data_routes, visualization_routes, health_routes
from src.analysis_engine import WeatherAnalyzer
from src.visualization import VisualizationEngine
import src.api.dependencies
from src.config import get_settings

# ============ LOGGING CONFIGURATION ============
# Bootstrap logger with a default level temporarily
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load settings
try:
    settings = get_settings()
    logger.info("Application settings successfully loaded")
    logger.debug(f"Settings configuration: {settings}")
except Exception as e:
    logger.error(f"Critical error: Failed to load application settings - {e}")
    logger.error("Application cannot start without valid configuration")
    raise  # don't proceed if settings are broken

# get log level
log_level_str = getattr(settings, "LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# Reset root logger with correct level
logger.info("Reconfiguring logging system with settings-based log level")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(log_level)  # Also adjust module-level logger

logger.info(f"Logging system configured successfully at level: {log_level_str}")

# ============ APPLICATION LIFESPAN MANAGEMENT ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Handles the initialization and cleanup of application resources including:
    - Loading and validating sample data
    - Initializing analysis and visualization engines
    - Setting up application state

    Args:
        app (FastAPI): The FastAPI application instance

    Yields:
        None: Control back to the application during its runtime

    Raises:
        Exception: If critical startup operations fail
    """

    # ============ STARTUP SEQUENCE ============
    logger.info("Starting Weather Data Intelligence Platform...")
    logger.info("Initializing application components...")

    try:
        # Load and validate sample data
        logger.info("Loading sample environmental data...")
        sample_data = load_sample_data()  # This function would load your actual data

        if sample_data.empty:
            logger.warning("No data loaded - application will run with empty dataset")
        else:
            logger.info(f"Successfully loaded {len(sample_data)} data points")
            logger.debug(f"Data columns: {list(sample_data.columns)}")
            logger.debug(f"Data date range: {sample_data.index.min()} to {sample_data.index.max()}")

        # Initialize analysis engine
        logger.info("Initializing weather analysis engine...")
        src.api.dependencies.state["analyzer"] = WeatherAnalyzer(sample_data)
        logger.info("Weather analysis engine initialized successfully")

        # Initialize visualization engine
        logger.info("Initializing visualization engine...")
        src.api.dependencies.state["visualizer"] = VisualizationEngine(src.api.dependencies.state["analyzer"])
        logger.info("Visualization engine initialized successfully")

        logger.info("Application startup completed successfully")
    except FileNotFoundError as e:
        logger.error(f"Data file not found during startup: {e}")
        logger.error("Application will continue but functionality may be limited")
    except Exception as e:
        logger.error(f"Critical error during application startup: {e}")
        logger.error("This may cause application instability")
        raise # prevent startup

    # Yield control to the running application
    yield

    # ============ SHUTDOWN SEQUENCE ============
    logger.info("Initiating Weather Data Intelligence Platform shutdown...")
    try:
        # Cleanup application state
        if "analyzer" in src.api.dependencies.state:
            logger.info("Cleaning up analysis engine resources...")
            del src.api.dependencies.state["analyzer"]

        if "visualizer" in src.api.dependencies.state:
            logger.info("Cleaning up visualization engine resources...")
            del src.api.dependencies.state["visualizer"]

        logger.info("Application shutdown completed successfully")

    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")

# ============ FASTAPI APPLICATION SETUP ============

# Create FastAPI app
app = FastAPI(
    title="Weather Data Intelligence Platform",
    description="""
    Scientific data analysis pipeline for environmental sensor data.
    
    This API provides endpoints for:
    - Environmental data access and querying
    - Statistical analysis and trend detection  
    - Data visualization and reporting
    - System health monitoring""",
    version="1.0.0",
    lifespan=lifespan
)

# Register API route modules
logger.info("Registering API route modules...")

app.include_router(health_routes.router)
app.include_router(data_routes.router)
app.include_router(analysis_routes.router)
app.include_router(visualization_routes.router)
logger.info("All API routes registered successfully")

# ============ MIDDLEWARE CONFIGURATION ============

# Add CORS middleware for cross-origin requests
logger.info("Configuring CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, we should probably specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured successfully")

# ============ UTILITY FUNCTIONS ============

def load_sample_data() -> pd.DataFrame:
    df = pd.read_json("../../data/environmental_sensor_data.json")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


# ============ MAIN ============

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        reload=True,
    )