# Environmental Data Intelligence Platform

A comprehensive scientific data analysis pipeline for environmental sensor data with REST API interface, implementing advanced statistical analysis, time series decomposition, anomaly detection, and data visualization capabilities.

## 🔬 Scientific Approach

### Statistical Methods Used

This platform implements a comprehensive suite of scientific analysis methods for environmental data:

#### **Time Series Decomposition**
- **STL (Seasonal and Trend decomposition using Loess)**: For single seasonal patterns
- **MSTL (Multiple Seasonal-Trend decomposition using Loess)**: For complex multi-seasonal patterns
- Automatically detects and quantifies:
  - Daily temperature cycles (24-hour periods)
  - Weekly weather patterns (168-hour periods)  
  - Seasonal variations (yearly cycles)
  - Long-term trends and residual noise

#### **Anomaly Detection**
- **MAD (Median Absolute Deviation)**: Robust to outliers, uses 99.84th percentile threshold
- **Z-Score Method**: Standard statistical approach for normally distributed data
- **Isolation Forest**: Advanced ML-based anomaly detection (configurable)
- **IQR Method**: Interquartile range-based outlier detection

#### **Missing Data Reconstruction**
Sophisticated imputation pipeline:
1. Temporal interpolation of cleaned data
2. STL decomposition of interpolated series
3. Signal reconstruction using trend + seasonal components
4. Residual interpolation for natural variation preservation

#### **Distribution Fitting & Analysis**
- Fits multiple statistical distributions: Normal, Exponential, Gamma, Log-normal, Beta, Weibull
- Kolmogorov-Smirnov goodness-of-fit testing
- Best-fit distribution identification with parameter estimation

#### **Correlation Analysis**
- Pearson correlation (linear relationships)
- Spearman correlation (monotonic relationships)  
- Kendall's tau (rank-based correlation)

#### **Circular Statistics**
- Mean time calculations for cyclical data (daily/weekly/yearly patterns)
- Circular standard deviation for timing uncertainty quantification
- Handles wraparound effects in temporal data

## 🏗️ Architecture Overview

### System Design

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Analysis        │    │ Visualization   │
│   REST API      │◄──►│  Engine          │◄──►│ Engine          │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Pydantic      │    │  Scientific      │    │  Matplotlib/    │
│   Models        │    │  Libraries       │    │  Seaborn        │
│                 │    │  (NumPy, SciPy)  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Interaction

#### **WeatherAnalyzer** (`analysis_engine.py`)
Core scientific computing engine:
- Data preprocessing and cleaning pipeline
- Statistical analysis and decomposition
- Anomaly detection algorithms
- Distribution fitting and trend analysis
- Caching for expensive computations via joblib

#### **VisualizationEngine** (`visualization.py`)
Extends WeatherAnalyzer for visualization:
- Time series plots with moving averages
- Correlation heatmaps and distribution plots
- Seasonal decomposition visualizations
- Anomaly detection scatter plots
- Base64 image encoding for API responses

#### **FastAPI Application** (`main.py`)
- Asynchronous request handling
- Lifespan management for resource initialization
- CORS middleware for cross-origin requests
- Comprehensive logging and error handling

#### **Configuration Management** (`config.py`)
- Environment variable-based configuration
- Pydantic settings validation
- Caching control and logging levels

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- Git

### Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd environmental-data-intelligence-platform
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate Sample Data**
```bash
cd src
python data_generator.py
```

5. **Configure Environment** (Optional)
Create a `.env` file in the project root:
```env
LOG_LEVEL=INFO
DATA_SEED=42
CACHE_RESULTS=true
```

### Running the Application

#### Development Mode
```bash
cd src
python main.py
```

#### Production Mode with Uvicorn
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

#### Docker Deployment
```bash
# Build the container
docker build -t weather-intelligence .

# Run the container
docker run -p 8000:8000 weather-intelligence
```

### Accessing the API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Data Summary**: http://localhost:8000/data/summary

## 📚 API Documentation

### Core Endpoints

#### Health & Status
```http
GET /health
```
Returns system health status and component readiness.

#### Data Access
```http
GET /data/summary
```
Comprehensive dataset statistics and metadata.

```http
GET /data/timeseries?start_date=2024-01-01&end_date=2024-01-31&parameter=temperature_c
```
Filtered time series data with optional date range and parameter selection.

#### Statistical Analysis
```http
POST /analysis/correlation
Content-Type: application/json

{
  "method": "pearson"
}
```

```http
POST /analysis/anomalies
Content-Type: application/json

{
  "method": "mad",
  "thresholds": {
    "temperature_c": 3.0
  },
  "parameters": ["temperature_c", "humidity_percent"]
}
```

```http
GET /analysis/trends/temperature_c
```

#### Visualizations
```http
GET /visualizations/{chart_type}
```
Where `chart_type` can be:
- `timeseries`: Time series plots with moving averages
- `correlation`: Correlation heatmaps  
- `distribution`: Distribution fitting plots
- `decomposition`: Seasonal decomposition plots
- `anomaly_detection`: Anomaly detection visualizations

### Example Usage

```python
import requests

# Get data summary
response = requests.get("http://localhost:8000/data/summary")
summary = response.json()

# Detect anomalies
anomaly_request = {
    "method": "mad",
    "parameters": ["temperature_c"]
}
response = requests.post("http://localhost:8000/analysis/anomalies", 
                        json=anomaly_request)
anomalies = response.json()

# Generate time series visualization
response = requests.get("http://localhost:8000/visualizations/timeseries")
visualization = response.json()
# visualization["images"] contains base64-encoded plots
```

## 📊 Analysis Results

### Sample Insights from Generated Data

Based on the synthetic temperate climate dataset (8,760 hourly measurements):

#### **Temporal Patterns Detected**
- **Daily Temperature Cycle**: Clear 24-hour pattern with peaks at ~15:00 and valleys at ~06:00
- **Seasonal Variation**: 15°C amplitude between summer and winter temperatures
- **Humidity Anti-correlation**: Strong negative correlation with temperature (-0.65 Pearson)
- **Pressure Cycles**: Multi-day weather system variations (48-hour and weekly patterns)

#### **Anomaly Detection Results**
- **Temperature anomalies**: 0.5% of data (sensor spikes ±20-40°C from normal)
- **Humidity anomalies**: Impossible readings >100% or <3%
- **Pressure anomalies**: Extreme deviations ±50-100 hPa from normal

#### **Data Quality Assessment**
- **Completeness**: 97.8% complete data after simulated sensor outages
- **Missing data**: Successfully imputed using STL-based reconstruction
- **Seasonal extremes**: Winter min: -15.3°C, Summer max: 35.7°C

#### **Distribution Analysis**
- **Temperature**: Best fit to normal distribution (temperate climate)
- **Humidity**: Beta distribution (bounded 0-100%)
- **Pressure**: Normal distribution around 1013.25 hPa

## 🔧 Technical Implementation Details

### Performance Optimizations
- **Joblib caching**: Expensive STL decompositions cached to disk
- **Async FastAPI**: Non-blocking request handling
- **Efficient data structures**: NumPy arrays for numerical computations
- **Memory management**: Automatic matplotlib figure cleanup

### Error Handling
- **Comprehensive validation**: Pydantic models for all API inputs/outputs
- **Graceful degradation**: System continues with warnings for non-critical errors
- **HTTP status codes**: Proper REST API error responses
- **Logging**: Structured logging at multiple levels

### Testing Strategy
- **Unit tests**: Core analytical functions tested with pytest
- **Integration tests**: Full API endpoint testing
- **Statistical validation**: Known patterns verified in synthetic data
- **Edge case handling**: Missing data and outlier scenarios tested

## 🧪 Scientific Assumptions and Limitations

### Assumptions Made

#### **Data Generation**
- **Temperate climate simulation**: Moderate seasonal variation, winter humidity peaks
- **Sensor characteristics**: Gaussian noise with σ=2.5°C for temperature
- **Missing data patterns**: Random 2% + periodic outages (realistic sensor behavior)
- **Anomaly injection**: 0.5% of data with sensor malfunction signatures

#### **Statistical Methods**
- **STL decomposition**: Assumes additive seasonal components
- **Anomaly thresholds**: 99.84th percentile (≈3σ equivalent) for balance
- **Distribution fitting**: Limited to common meteorological distributions
- **Circular statistics**: Assumes unimodal cyclical patterns

### Limitations

#### **Technical Constraints**
- **Memory usage**: Full dataset loaded into memory (suitable for hourly yearly data)
- **Real-time processing**: Not optimized for streaming data
- **Geographic scope**: Single-location analysis (no spatial correlation)

#### **Scientific Limitations**
- **Model complexity**: Linear trend assumption in STL decomposition
- **Correlation analysis**: Does not imply causation
- **Missing data**: Long gaps (>24h) may have reduced imputation accuracy
- **Anomaly detection**: May miss novel anomaly types not in training patterns

### Future Enhancements
- **Multivariate anomaly detection**: Consider variable interactions
- **Fourier analysis**: Frequency domain pattern identification
- **Machine learning integration**: Weather prediction models
- **Real-time processing**: Streaming analysis capabilities

## 🛠️ Development

### Code Quality Standards
- **Type hints**: Full static typing with mypy compatibility
- **Documentation**: Comprehensive docstrings for all public methods
- **Error handling**: Defensive programming with informative error messages
- **Modularity**: Clear separation of concerns between components

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is developed as part of the PhaseTree technical assessment.

---

**Author**: Nicolai Rask Mathiesen  
**Date**: July 2025  
**Version**: 1.0.0