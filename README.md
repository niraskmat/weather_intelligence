# Environmental Data Intelligence Platform

A scientific data analysis pipeline for environmental sensor data with REST API interface, implementing advanced statistical analysis, time series decomposition, anomaly detection, and data visualization capabilities.

## 🔬 Scientific Approach

### Statistical Methods Used

#### **Fourier Analysis**
- Using FFT, peaks in the frequency spectrum are identified and converted to periods which are later used in the decomposition.   

#### **Time Series Decomposition**
- **STL (Seasonal and Trend decomposition using Loess)**: For single seasonal patterns
- **MSTL (Multiple Seasonal-Trend decomposition using Loess)**: For multi-seasonal patterns, i.e. if more than one period is detected in the Fourier Analysis
- Automatically detects and quantifies:
  - Cycles (24-hour, 48-hour, and 168-hour periods) 
  - Seasonal variations (yearly cycles) as the trend
  - Residual noise

#### **Anomaly Detection**
- **MAD (Median Absolute Deviation)**: Robust to outliers
- **Z-Score Method**: Standard statistical approach for normally distributed data
- **IQR Method**: Interquartile range-based outlier detection
- Uses 99.84th percentile threshold by default for MAD and Z-score (assuming we know the sensor have an expected 0.16% miss-reading rate)

#### **Missing Data Reconstruction**
Imputation pipeline:
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
git clone https://github.com/niraskmat/weather_intelligence.git
cd weather_intelligence
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

4. **Configure Environment** (Optional)
Create a `.env` file in the project root:
```env
LOG_LEVEL=INFO
DATA_SEED=42
CACHE_RESULTS=True
```

### Running the Application

#### Development Mode
```bash
export PYTHONPATH=.
python src/api/main.py
```

### Testing
```bash
# Run all tests
pytest
```

#### Production Mode with Uvicorn
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
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
Dataset statistics and metadata.

```http
GET /data/timeseries?start_date=2024-01-01&end_date=2024-01-31&parameter=temperature_c
```
Filtered time series data with optional date range and parameter selection.

#### Statistical Analysis
Get correlation matrix with or without time lag (in hours). For no time lag set lag to 0.
```http
POST /analysis/correlation
Content-Type: application/json

{
  "method": "pearson",
  "lag": 6,
  "lag_column": "temperature_c_filled" 
}
```
Get anomalies for chosen parameters using different methods and specified thresholds. By default it will use mad, auto set thresholds and return results for all three variables.
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
Get trends for temperature_c, humidity_percent, or air_pressure_hpa. returns comprehensive analysis of cycles and seasonal extremes.
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

Based on the synthetic temperate climate dataset (8,760 hourly measurements - 1 year):

#### **Patterns Detected**
- **Daily Temperature Cycle**: Clear 24-hour pattern with peaks at ~12:00 and valleys at ~00:00
- **Daily Temperature difference**: The decomposed daily cycle shows a 19°C mean difference between day and night 
- **Seasonal Variation**: 33°C amplitude between summer and winter temperatures (looking at the yearly trend)
- **Humidity Anti-correlation**: Strong negative correlation with a 12-hour lag to temperature (-0.94 Spearman)
- **Pressure Cycles**: Multi-day weather system variations (48-hour and 168-hour (weekly) patterns)
- **Pressure Days**: Pressure is highest around Tuesday evening and lowest around Saturday morning...

## 🔧 Technical Implementation Details

### Performance Optimizations
- **Joblib caching**: Expensive STL decompositions and distribution fitting cached to disk. More methods could benefit but this shows the idea.
- **Async FastAPI**: Non-blocking request handling
- **Efficient data structures**: NumPy arrays and Pandas dataframes for numerical computations
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

#### **Statistical Methods**
- **STL decomposition**: Assumes additive seasonal components
- **Automatic Anomaly thresholds**: 99.84th percentile equivalent to 0.16% sensor error rate
- **Distribution fitting**: Limited to common meteorological distributions
- **Circular statistics**: Assumes unimodal cyclical patterns

### Limitations

#### **Technical Constraints**
- **Memory usage**: Full dataset loaded into memory (suitable for hourly yearly data)
- **Real-time processing**: Not optimized for streaming data
- **Performance**: Most methods are implemented with use of optimized methods through scipy, numpy, etc. but we still might run into performance issues in some scenarios. Particularly if the dataset is larger, e.g. if we 10 years with minute resolution (factor 600) - we would need to optimize or change to less demanding methods. 
- **Endpoint caching**: is not implemented. For production it would be beneficial on visulization and some analysis endpoints - could be done with Redis for multi instance support. 
- **Multi instance caching**: The current joblib caching doesn't support multi instance caching.

#### **Scientific Limitations**
- **Missing data**: Long gaps (>24h) may have reduced imputation accuracy
- **STL/MSTL decomposition**: 
  - Not suitable for streaming scenarios - would like need another method if we were working with live data from a sensor.
  - Might be too slow for some scenarios with more data
  - Right now the MSTL is hardcoded to two cycles - needs to be generalized 
- Assumes hourly data many places, would need adaptation to work with different data point intervals
- Assumes one year worth of data in some places - needs to be generalized

#### **Missing Elements**
- Only Fourier Analysis was implemented from the advanced statistics - lack of time to do more
- Analysis_windows_days was not implemented - I am not sure how it was supposed to work
- No use of the generator script, the API is depending on loading the pre-generated data.
- The API endpoints support asynchronous behaviour but none of the underlying analysis functions support async.

### Future Enhancements
There is enough to deal with in the previous section. But I have made some more analysis on anomalies including a probability plot and a plot of sorted anomaly scores which would help adjust anomaly detection. I have also made some plots for inspecting imputation quality which could be served to the user.
But time is up!

---

**Author**: Nicolai Rask Mathiesen  
**Date**: July 2025  
**Version**: 1.0.0