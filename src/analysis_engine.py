"""
Weather Data Analysis Engine

This module provides comprehensive weather data analysis capabilities including:
- Seasonal decomposition using STL/MSTL
- Anomaly detection using MAD and Z-score methods
- Missing data imputation
- Correlation analysis
- Distribution fitting
- Trend and pattern analysis
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import STL, MSTL
from statsmodels.robust.scale import mad
from typing import Any, Dict, List, Optional, Tuple, Union

from src.cache import memory

# Configure logging
logger = logging.getLogger(__name__)


class WeatherAnalyzer:
    """
    Weather data analyzer with anomaly detection, trend analysis,
    and missing data imputation capabilities.

    This class provides methods for:
    - Cleaning and preprocessing weather data
    - Detecting and removing anomalies
    - Seasonal decomposition and trend analysis
    - Statistical analysis and distribution fitting
    - Correlation analysis between weather parameters

    Attributes:
        df (pd.DataFrame): The main weather dataset with timestamp index
        residuals (dict): Residuals from seasonal decomposition for each parameter
        decomposition (dict): STL/MSTL decomposition results for each parameter
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the WeatherAnalyzer with weather data.

        - Performs Fourier analysis to identify cycles.
        - Performs STL/MSTL to identify patterns and residuals. Residuals are used to identify and remove anomalies.
        - Missing data is imputed using patterns from STL/MSTL to reconstruct the missing data points.
        - Adds _cl version of each parameter to self.df which is the observed data with anomalies removed.
        - Adds _filled version of each parameter self.df which is the _cl data with all missing data points imputed.

        Args:
            data (pd.DataFrame): Weather dataset with timestamp index and columns:
                - temperature_c: Temperature in Celsius
                - humidity_percent: Relative humidity percentage
                - air_pressure_hpa: Air pressure in hectopascals
        Attributes:
            df (pd.DataFrame): The main weather dataset with timestamp index
            residuals (dict): Residuals from seasonal decomposition for each parameter
            decomposition (dict): STL/MSTL decomposition results for each parameter
        """
        logger.info("Initializing WeatherAnalyzer with %d records", len(data))
        self.df = data.copy()
        self.residuals = {}
        self.decomposition = {}
        self.found_periods = {}

        logger.info("identifying frequencies with Fourier analysis")
        for col in ["temperature_c", "humidity_percent", "air_pressure_hpa"]:
            fft_result = self.identify_cycles(self.df[col].copy())
            self.found_periods[col] = fft_result.period_full_days_in_hours.to_list()

        # Process each weather parameter according to configuration
        for col, periods in self.found_periods.items():
            logger.info("Processing column: %s with periods: %s", col, periods)
            self.get_clean_data(col, periods=periods)

        logger.info("WeatherAnalyzer initialization complete")


    def fit_distributions(self, col: str,
                          dist_names: Optional[List[str]] = None) -> List[Tuple[str, float, float, Tuple]]:
        """
        Fit multiple statistical distributions to weather data.

        Args:
            col (str): Column name to analyze
            dist_names (list, optional): List of distribution names to fit.
                Defaults to common distributions: 'norm', 'expon', 'gamma',
                                                  'lognorm', 'beta', 'weibull_min'.

        Returns:
            list: Sorted list of tuples (dist_name, ks_stat, p_value, params)
                ordered by goodness of fit (lowest KS statistic first)
        """
        logger.info("Fitting distributions for column: %s", col)
        data = self.df[col].to_numpy()
        return fit_distributions_cached(data, dist_names=dist_names)

    def get_best_fit(self, fits: List[Tuple[str, float, float, Tuple]]) -> Tuple[str, Tuple]:
        """
        Extract the best fitting distribution from fit results.

        Args:
            fits (list): Results from fit_distributions method

        Returns:
            tuple: (distribution_name, parameters) for best fit
        """
        best_fit = fits[0]  # First item has lowest KS statistic
        dist_name, _, _, params = best_fit
        logger.info("Best distribution fit: %s with params: %s", dist_name, params)
        return dist_name, params

    def get_correlation_matrix(self, method: str = "pearson", lag: int = 0, lag_column: str = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between cleaned weather parameters.

        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            lag (str): time lag in hours

        Returns:
            pd.DataFrame: Correlation matrix between weather parameters
        """
        logger.info("Calculating correlation matrix using %s method", method)

        # Use cleaned/filled data for correlation analysis
        cols = ["temperature_c_filled", "humidity_percent_filled", "air_pressure_hpa_filled"]
        df_clean = self.df[cols].copy()
        if lag != 0:
            for col in cols:
                if col != lag_column:
                    df_clean[col] = df_clean[col].shift(lag)

        corr_matrix = df_clean.corr(method=method)

        logger.info("Correlation matrix calculated successfully")
        return corr_matrix

    def calculate_moving_averages(self, col: str,
                                  windows: List[int] = [7 * 24, 30 * 24]) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calculate moving averages for specified time windows.

        Args:
            col (str): Column name to calculate moving averages for
            windows (list): List of window sizes in hours (default: 7-day and 30-day)

        Returns:
            dict: Dictionary with moving averages for each window size
                Format: {window_name: {col: moving_average_series}}
        """
        logger.info("Calculating moving averages for %s with windows: %s hours", col, windows)
        results = {}

        for window in windows:
            # Create human-readable window name
            window_name = f"{window // 24}day" if window >= 24 else f"{window}hour"
            results[window_name] = {}

            logger.debug("Computing %s moving average", window_name)

            # Calculate centered moving average with minimum periods for edge handling
            results[window_name][col] = self.df[col].rolling(
                window=window,
                min_periods=max(1, window // 2),  # Require at least half the window
                center=True,
            ).mean()

        logger.info("Moving averages calculated for %d windows", len(windows))
        return results

    def get_trends(self, col: str) -> Dict[str, Any]:
        """
        Analyze trends and patterns in weather data using seasonal decomposition.

        This method identifies:
        - Cyclical patterns at different time scales
        - Peak and valley timing with std to quantify uncertainty
        - Seasonal extremes and global extremes
        - Amplitude statistics for different cycles

        Args:
            col (str): Weather parameter column to analyze

        Returns:
            dict: Comprehensive trend analysis containing:
                - patterns: List of cycle analyses for different periods
                - extremes: Seasonal and global extreme values with timestamps
        """
        logger.info("Analyzing trends for column: %s", col)

        periods = self.found_periods[col]
        decomp = self.decomposition[f"{col}_filled"]
        trends = {}
        patterns = []

        # Analyze configured seasonal patterns
        seasonal = decomp.seasonal.reset_index()
        season_col = "season"

        logger.info("Analyzing %d configured periods for %s", len(periods), col)

        for period in periods:
            logger.debug("Processing period: %d hours", period)

            cycle = {"period": period, "cycles": len(seasonal) // period}

            # Handle multiple seasonal components
            if len(periods) > 1:
                season_col = f"seasonal_{period}"

            # Analyze both decomposed seasonal component and raw data
            cycle["decomposition"] = self.analyze_cycle(seasonal, season_col, period)
            cycle["raw_data"] = self.analyze_cycle(self.df.reset_index(), f"{col}_filled", period)
            patterns.append(cycle)

        # Analyze yearly trends
        logger.info("Analyzing yearly trends")
        period = 24 * 365  # One year in hours
        cycle = {"period": period, "cycles": len(decomp.trend) // period}
        cycle["decomposition"] = self.analyze_cycle(decomp.trend.reset_index(), "trend", period)
        cycle["raw_data"] = self.analyze_cycle(self.df.reset_index(), f"{col}_filled", period)
        patterns.append(cycle)

        trends["patterns"] = patterns

        # Calculate seasonal extremes
        logger.info("Calculating seasonal extremes")
        df_extreme = self.df[f"{col}_filled"].to_frame()
        df_extreme["season"] = df_extreme.index.map(assign_season)

        # Group by season and compute min/max with timestamps
        extremes_res = df_extreme.groupby("season").agg({
            f"{col}_filled": ["min", "idxmin", "max", "idxmax"],
        })

        # Format extremes results
        extremes = {
            season.lower(): {
                "min": row[(f"{col}_filled", "min")],
                "min_time": row[(f"{col}_filled", "idxmin")],
                "max": row[(f"{col}_filled", "max")],
                "max_time": row[(f"{col}_filled", "idxmax")],
            }
            for season, row in extremes_res.iterrows()
        }

        # Add global extremes
        global_min = df_extreme[f"{col}_filled"].min()
        global_min_time = df_extreme[f"{col}_filled"].idxmin()
        global_max = df_extreme[f"{col}_filled"].max()
        global_max_time = df_extreme[f"{col}_filled"].idxmax()

        extremes["global_min"] = global_min
        extremes["global_min_time"] = global_min_time
        extremes["global_max"] = global_max
        extremes["global_max_time"] = global_max_time

        trends["extremes"] = extremes

        logger.info("Trend analysis complete for %s", col)
        return trends

    def analyze_cycle(self, df: pd.DataFrame,
                      seasonal_col: str,
                      period_hours: int) -> Dict[str, Union[pd.Timedelta, float]]:
        """
        Analyze cyclical patterns in time series data.

        This method identifies peak and valley timing patterns across multiple
        cycles and calculates statistical measures of cycle characteristics.

        Args:
            df (pd.DataFrame): DataFrame with timestamp and seasonal data
            seasonal_col (str): Column name containing seasonal values
            period_hours (int): Length of one cycle in hours

        Returns:
            dict: Cycle analysis results including:
                - Peak/valley timing statistics for day, week, year periods
                - Amplitude statistics (mean and standard deviation)
        """
        logger.debug("Analyzing cycle for period: %d hours", period_hours)

        # Trim data to complete cycles only
        trimmed_len = len(df) - (len(df) % period_hours)
        df = df.iloc[:trimmed_len]
        logger.debug("Trimmed data to %d records (%d complete cycles)",
                     trimmed_len, trimmed_len // period_hours)

        # Assign cycle numbers to each data point
        df["cycle"] = np.repeat(np.arange(len(df) // period_hours), period_hours)

        # Find peaks and valleys for each cycle
        peaks = df.loc[df.groupby("cycle")[seasonal_col].idxmax()]
        valleys = df.loc[df.groupby("cycle")[seasonal_col].idxmin()]

        # Calculate amplitude statistics
        amplitude = peaks[seasonal_col].to_numpy() - valleys[seasonal_col].to_numpy()
        mean_amplitude = np.mean(amplitude)
        std_amplitude = np.std(amplitude)

        logger.debug("Cycle amplitude - mean: %.3f, std: %.3f", mean_amplitude, std_amplitude)

        # Calculate timing statistics for different periods using circular statistics
        mean_peak_time, peak_std = mean_time_of_period(peaks["timestamp"], period="day")
        mean_valley_time, valley_std = mean_time_of_period(valleys["timestamp"], period="day")
        mean_peak_time_week, peak_std_week = mean_time_of_period(peaks["timestamp"], period="week")
        mean_valley_time_week, valley_std_week = mean_time_of_period(valleys["timestamp"], period="week")
        mean_peak_time_year, peak_std_year = mean_time_of_period(peaks["timestamp"], period="year")
        mean_valley_time_year, valley_std_year = mean_time_of_period(valleys["timestamp"], period="year")

        results = {
            "mean_peak_time_day": mean_peak_time,
            "std_peak_time_day": peak_std,
            "mean_valley_time_day": mean_valley_time,
            "std_valley_time_day": valley_std,
            "mean_peak_time_week": mean_peak_time_week,
            "std_peak_time_week": peak_std_week,
            "mean_valley_time_week": mean_valley_time_week,
            "std_valley_time_week": valley_std_week,
            "mean_peak_time_year": mean_peak_time_year,
            "std_peak_time_year": peak_std_year,
            "mean_valley_time_year": mean_valley_time_year,
            "std_valley_time_year": valley_std_year,
            "mean_amplitude": mean_amplitude,
            "std_amplitude": std_amplitude
        }

        logger.debug("Cycle analysis complete")
        return results

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the weather dataset.

        Returns:
            dict: Dataset summary containing:
                - data_info: Record count, date range, missing data counts
                - basic_statistics: Descriptive statistics for all columns
        """
        logger.info("Generating data summary")

        summary = {
            'data_info': {
                'total_records': len(self.df),
                'date_range': {
                    'start': self.df.index.min().isoformat(),
                    'end': self.df.index.max().isoformat()
                },
                'missing_data': {
                    'temperature_c': self.df['temperature_c'].isna().sum(),
                    'humidity_percent': self.df['humidity_percent'].isna().sum(),
                    'air_pressure_hpa': self.df['air_pressure_hpa'].isna().sum()
                }
            },
            'basic_statistics': self.df.describe().to_dict()
        }

        logger.info("Data summary generated - %d records from %s to %s",
                    summary['data_info']['total_records'],
                    summary['data_info']['date_range']['start'],
                    summary['data_info']['date_range']['end'])

        return summary

    def get_clean_data(self, col: str, periods: List[int]) -> None:
        """
        Complete data cleaning pipeline for a weather parameter.

        This method orchestrates the full cleaning process:
        1. Seasonal decomposition
        2. Anomaly detection using residuals
        3. Anomaly removal
        4. Missing data imputation

        Args:
            col (str): Weather parameter column name
            periods (list): List of seasonal periods for decomposition
        """
        logger.info("Starting data cleaning pipeline for %s", col)

        # Step 1: Seasonal decomposition
        logger.debug("Performing STL decomposition")
        decomp = self.get_STL_decomposition(self.df[col], col, periods=periods)

        # Step 2: Extract residuals for anomaly detection
        logger.debug("Extracting residuals")
        self.residuals[col] = decomp.resid

        # Step 3: Detect anomalies using residuals
        logger.debug("Detecting anomalies")
        anomalies, threshold = self.get_anomalies(col)
        logger.info("Detected %d anomalies for %s (threshold: %.3f)",
                    len(anomalies), col, threshold)

        # Step 4: Remove detected anomalies
        logger.debug("Removing anomalies")
        self.remove_anomalies(anomalies, col=col)

        # Step 5: Impute missing data
        logger.debug("Imputing missing data")
        self.construct_missing_data(col=col, periods=periods)

        logger.info("Data cleaning pipeline complete for %s", col)

    def get_STL_decomposition(self, df_col: pd.Series, col: str, periods: List[int] = [24]) -> Union[Any, Any]:
        """
        Perform Seasonal and Trend decomposition using Loess (STL).

        Uses MSTL for multiple seasonal periods or STL for single period.
        Results are cached for performance.

        Args:
            df_col (pd.Series): Time series data to decompose
            col (str): Column name for storing results
            periods (list): List of seasonal periods in hours

        Returns:
            STLResults or MSTLResults: Decomposition results with trend, seasonal, residual
        """
        logger.debug("Performing STL decomposition for %s with periods: %s", col, periods)

        result = cached_stl_decomposition(df_col, tuple(periods))
        self.decomposition[col] = result

        logger.debug("STL decomposition complete for %s", col)
        return result

    def get_anomalies(
        self,
        col: str,
        method: str = "mad",
        threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, float]:
        """
        Detect anomalies using statistical methods on residuals.

        Args:
            col (str): Weather parameter column name
            method (str): Anomaly detection method ('mad', 'z_score', or "IQR")
            threshold (float, optional): Custom threshold. If None, uses 99.84th percentile

        Returns:
            tuple: (anomalies_dataframe, threshold_used)
                - anomalies_dataframe: DataFrame with anomalous records and scores
                - threshold_used: The threshold value applied
        """
        logger.debug("Detecting anomalies for %s using %s method", col, method)

        residuals = self.residuals[col]

        # Calculate anomaly scores based on method
        if method == "mad":
            # Median Absolute Deviation method - robust to outliers
            mad_val = mad(residuals)
            median = np.median(residuals)
            scores = np.abs(residuals - median) / mad_val
            logger.debug("Using MAD method - MAD value: %.3f, median: %.3f", mad_val, median)
        elif method == "z_score":
            # Standard Z-score method
            scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
            logger.debug("Using Z-score method")
        elif method == 'iqr':
            # Interquartile Range method - Traditional approach
            Q1 = residuals.quantile(0.25)
            Q3 = residuals.quantile(0.75)
            IQR = Q3 - Q1

            # Set default threshold for IQR (typically 1.5 for outliers, 3.0 for extreme outliers)
            if threshold is None:
                threshold = 2.5
                logger.debug("Using default IQR threshold: %.1f", threshold)

            # Calculate bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Calculate scores (how far beyond bounds, normalized by IQR)
            scores = np.maximum(
                (lower_bound - residuals) / IQR,  # How far below lower bound
                (residuals - upper_bound) / IQR  # How far above upper bound
            )

            # Ensure non-negative scores (points within bounds get score = 0)
            scores = np.maximum(scores, 0)
            logger.debug("Using IQR method - Q1: %.3f, Q3: %.3f, IQR: %.3f, bounds: [%.3f, %.3f]",
                         Q1, Q3, IQR, lower_bound, upper_bound)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")

        # Set threshold if not provided
        if threshold is None:
            threshold = np.percentile(scores, 99.84)
            logger.debug("Auto-calculated threshold: %.3f", threshold)

        # Identify anomalies above threshold
        if method == "iqr":
            anomalies = scores > 0
        else:
            anomalies = scores > threshold
        anomalies = anomalies.loc[anomalies]

        # Create detailed anomalies dataframe
        anomalies_full = self.df.loc[self.df.index.isin(anomalies.index)].copy()
        anomalies_full["score"] = scores.loc[scores.index.isin(anomalies.index)]

        logger.info("Detected %d anomalies for %s (%.2f%% of data)",
                    len(anomalies_full), col, 100 * len(anomalies_full) / len(self.df))

        return anomalies_full[[col, "score"]], threshold

    def remove_anomalies(
        self,
        anomalies: pd.DataFrame,
        col: str = "temperature_c"
    ) -> pd.DataFrame:
        """
        Remove detected anomalies by setting them to NaN.

        Creates a new column with '_cl' suffix containing cleaned data.

        Args:
            anomalies (pd.DataFrame): DataFrame with anomalous records
            col (str): Weather parameter column name

        Returns:
            pd.DataFrame: Updated dataframe with anomalies set to NaN
        """
        col_cl = f"{col}_cl"
        logger.debug("Removing %d anomalies from %s -> %s", len(anomalies), col, col_cl)

        # Create cleaned column as copy of original
        self.df[col_cl] = self.df[col]

        # Set anomalous values to NaN
        self.df.loc[self.df.index.isin(anomalies.index.tolist()), col_cl] = np.nan

        missing_after = self.df[col_cl].isna().sum()
        logger.info("Anomaly removal complete - %d missing values in %s", missing_after, col_cl)

        return self.df

    def construct_missing_data(
        self,
        col: str = "temperature_c",
        periods: List[int] = [24]
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Impute missing data using seasonal decomposition and interpolation.

        This sophisticated imputation method:
        1. Interpolates the cleaned data temporally
        2. Performs STL decomposition on interpolated data
        3. Reconstructs signal using trend + seasonal components
        4. Adds interpolated residuals for natural variation

        Args:
            col (str): Weather parameter column name
            periods (list): Seasonal periods for decomposition

        Returns:
            tuple: (updated_dataframe, decomposition_results)
        """
        col_cl = f"{col}_cl"
        filled_col = f'{col}_filled'

        logger.info("Starting missing data imputation for %s", col)
        missing_before = self.df[col_cl].isna().sum()
        logger.debug("Missing values before imputation: %d", missing_before)

        # Step 1: Temporal interpolation of cleaned data
        logger.debug("Performing temporal interpolation")
        interp = self.df[col_cl].interpolate('time')

        # Step 2: STL decomposition of interpolated series
        logger.debug("Decomposing interpolated series")
        res = self.get_STL_decomposition(interp, filled_col, periods=periods)

        # Step 3: Reconstruct baseline signal (trend + seasonal)
        logger.debug("Reconstructing baseline signal")
        if isinstance(res.seasonal, pd.Series):
            # Single seasonal component
            baseline = res.trend + res.seasonal
        elif isinstance(res.seasonal, pd.DataFrame):
            # Multiple seasonal components - sum them
            baseline = res.trend + res.seasonal.sum(axis=1)

        # Step 4: Handle residuals with interpolation for small gaps
        logger.debug("Processing residuals")
        residual_interp = res.resid.copy()
        # Set residuals to NaN where original data was missing
        residual_interp[self.df[col_cl].isna()] = np.nan
        # Interpolate residuals
        residual_interp = residual_interp.interpolate()

        # Step 5: Final reconstruction
        self.df[filled_col] = baseline + residual_interp

        # Verify imputation results
        missing_after = self.df[filled_col].isna().sum()
        imputed_count = missing_before - missing_after

        logger.info("Missing data imputation complete for %s", col)
        logger.info("Imputed %d values, %d still missing", imputed_count, missing_after)

        return self.df, res

    def identify_cycles(self, observed_data: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Identify dominant cyclical patterns in time series data using Fast Fourier Transform (FFT).

        This method performs frequency domain analysis to detect repeating patterns in environmental
        sensor data such as daily temperature cycles, weekly pressure variations, etc. It uses FFT
        to transform the time series into the frequency domain, identifies significant peaks in the
        magnitude spectrum, and converts these back to time periods.

        The analysis workflow:
        1. Detrend the data by removing the mean
        2. Apply FFT to convert to frequency domain
        3. Filter to positive frequencies only (remove mirror frequencies)
        4. Detect peaks in the magnitude spectrum above a threshold
        5. Convert dominant frequencies back to time periods
        6. Round periods to full days and remove duplicates

        Args:
            observed_data (Union[pd.Series, np.ndarray]): Time series data with hourly observations.
                Should contain numeric values representing sensor measurements over time.
                Missing values should be handled before calling this method.

        Returns:
            pd.DataFrame: DataFrame containing identified cyclical patterns with columns:
                - 'period_hours' (float): Exact period length in hours
                - 'freq_hr' (float): Frequency in cycles per hour
                - 'amplitude' (float): Magnitude of the frequency component (strength of cycle)
                - 'period_full_days_in_hours' (int): Period rounded to nearest full day(s) in hours

                Sorted by amplitude (strongest cycles first) with duplicates removed.

        Raises:
            ValueError: If observed_data is empty or contains only NaN values
            TypeError: If observed_data is not a numeric array-like object

        Notes:
            - Assumes hourly time steps (time_step_hours = 1)
            - Peak detection threshold is set to 10% of maximum amplitude
            - Periods are rounded to full days to identify standard meteorological cycles
            - Only positive frequencies are considered (negative frequencies are redundant)
            - The method works best with data spanning multiple complete cycles
        """
        # Validate input data
        if len(observed_data) == 0:
            raise ValueError("Input data cannot be empty")

        # Convert to numpy array if pandas Series for consistent handling
        if isinstance(observed_data, pd.Series):
            data_array = observed_data.values
        else:
            data_array = np.asarray(observed_data)

        # Check for all NaN values
        if np.all(np.isnan(data_array)):
            raise ValueError("Input data contains only NaN values")

        # Remove NaN values for FFT analysis
        clean_data = data_array[~np.isnan(data_array)]

        N = len(clean_data)
        time_step_hours = 1  # Hourly data assumption

        # ============ FAST FOURIER TRANSFORM ============
        # Detrend by removing mean to focus on cyclical variations
        yf = fft(clean_data - clean_data.mean())

        # Generate frequency array (cycles per hour)
        xf = fftfreq(N, time_step_hours)

        # ============ FREQUENCY DOMAIN ANALYSIS ============
        # Consider only positive frequencies to avoid redundant negative frequency components
        pos_mask = xf > 0
        xf = xf[pos_mask]
        yf_magnitude = np.abs(yf[pos_mask])  # Take magnitude of complex FFT result

        # ============ PEAK DETECTION ============
        # Find peaks in magnitude spectrum that represent significant cyclical patterns
        # Threshold set to 10% of maximum amplitude to filter out noise
        peak_threshold = np.max(yf_magnitude) * 0.1
        peaks, _ = find_peaks(yf_magnitude, height=peak_threshold)

        # Extract dominant frequencies and their corresponding amplitudes
        dominant_freqs = xf[peaks]
        dominant_amps = yf_magnitude[peaks]

        # Convert frequencies (cycles/hour) to periods (hours/cycle)
        dominant_periods = 1 / dominant_freqs

        # Round periods to full days for meteorological interpretation
        # This helps identify standard cycles like daily (24h), weekly (168h), etc.
        periods_full_days = np.round(dominant_periods / 24)
        periods_full_days_hours = periods_full_days * 24

        # ============ CREATE RESULTS DATAFRAME ============
        periods_df = pd.DataFrame({
            'period_hours': dominant_periods,  # Exact period in hours
            'freq_hr': dominant_freqs,  # Frequency in cycles per hour
            'amplitude': dominant_amps,  # Strength of cyclical pattern
            "period_full_days_in_hours": periods_full_days_hours,  # Rounded to full days
        })

        # Sort by amplitude (strongest cycles first) for easier interpretation
        periods_df = periods_df.sort_values(by='amplitude', ascending=False)

        # Convert to integer for cleaner output and remove duplicates
        periods_df["period_full_days_in_hours"] = periods_df["period_full_days_in_hours"].astype(int)

        # Remove duplicate periods (keeps the one with highest amplitude due to sorting)
        periods_df = periods_df.drop_duplicates(subset="period_full_days_in_hours")

        return periods_df


@memory.cache
def fit_distributions_cached(
    data: np.ndarray,
    dist_names: Optional[List[str]] = None
) -> List[Tuple[str, float, float, Tuple[float, ...]]]:
    """
    Fit multiple statistical distributions to data with caching.

    Uses Kolmogorov-Smirnov test to evaluate goodness of fit.
    Results are cached to avoid repeated computation.

    Args:
        data (np.ndarray): Data to fit distributions to
        dist_names (list, optional): Distribution names to try

    Returns:
        list: Sorted list of (dist_name, ks_stat, p_value, params) tuples
    """
    if dist_names is None:
        dist_names = ['norm', 'expon', 'gamma', 'lognorm', 'beta', 'weibull_min']

    logger.debug("Fitting %d distributions to %d data points", len(dist_names), len(data))

    results = []
    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        try:
            # Fit distribution parameters
            params = dist.fit(data)
            # Test goodness of fit using Kolmogorov-Smirnov test
            ks_stat, p_value = stats.kstest(data, dist_name, args=params)
            results.append((dist_name, ks_stat, p_value, params))
            logger.debug("Fitted %s: KS=%.4f, p=%.4f", dist_name, ks_stat, p_value)
        except Exception as e:
            logger.warning("Failed to fit %s distribution: %s", dist_name, e)

    # Sort by KS statistic (lower is better)
    results.sort(key=lambda x: x[1])
    logger.debug("Distribution fitting complete, best fit: %s", results[0][0] if results else "None")

    return results


@memory.cache
def cached_stl_decomposition(
    data: pd.Series,
    periods: Tuple[int, ...]
) -> Union[Any, Any]:  # STLResults or MSTLResults
    """
    Perform STL decomposition with caching for performance.

    Uses MSTL for multiple periods or STL for single period.

    Args:
        data (pd.Series): Time series data to decompose
        periods (tuple): Seasonal periods (tuple for hashability in cache)

    Returns:
        STLResults or MSTLResults: Decomposition results
    """
    logger.debug("Performing cached STL decomposition with periods: %s", periods)

    # Remove missing values before decomposition
    data = data.dropna()
    logger.debug("Data after dropna: %d points", len(data))

    if len(periods) > 1:
        # Multiple seasonal periods - use MSTL
        logger.debug("Using MSTL for multiple periods")
        stl = MSTL(data, periods=list(periods), windows=[19, 15])
        result = stl.fit()
    else:
        # Single seasonal period - use STL
        logger.debug("Using STL for single period")
        stl = STL(data, period=periods[0], robust=True)
        result = stl.fit()

    logger.debug("STL decomposition complete")
    return result


def mean_time_of_period(
    timestamps: pd.Series,
    period: str = "day"
) -> Tuple[pd.Timedelta, pd.Timedelta]:
    """
    Calculate mean time within a period using circular statistics.

    This function handles the circular nature of time (e.g., 23:59 and 00:01
    are close in time). It converts times to angles on a unit circle and
    calculates the mean angle.

    Args:
        timestamps (pd.Series): Series of timestamps
        period (str): Period type - 'day', 'week', or 'year'

    Returns:
        tuple: (mean_time, circular_standard_deviation)
            Both as pd.Timedelta objects
    """
    logger.debug("Calculating mean time for period: %s", period)

    if period == "day":
        # Convert to seconds since midnight
        seconds_since_start = (
                timestamps.dt.hour * 3600 +
                timestamps.dt.minute * 60 +
                timestamps.dt.second
        )
        period_seconds = 24 * 60 * 60
    elif period == "week":
        # Convert to seconds since start of week (Sunday)
        week_start = timestamps.dt.to_period('W-SUN').dt.start_time
        seconds_since_start = (timestamps - week_start).dt.total_seconds()
        period_seconds = 7 * 24 * 60 * 60
    elif period == "year":
        # Convert to seconds since start of year
        year_start = timestamps.dt.to_period('Y').dt.start_time
        seconds_since_start = (timestamps - year_start).dt.total_seconds()
        period_seconds = 365 * 24 * 60 * 60
    else:
        raise ValueError(f"Unknown period: {period}")

    # Convert times to angles on unit circle (0 to 2π)
    angles = 2 * np.pi * seconds_since_start / period_seconds

    # Convert to unit circle coordinates
    x = np.cos(angles)
    y = np.sin(angles)

    # Calculate mean coordinates (centroid)
    mean_x = x.mean()
    mean_y = y.mean()

    # Resultant vector length (measure of concentration)
    R = np.sqrt(mean_x ** 2 + mean_y ** 2)

    # Mean angle (direction of resultant vector)
    mean_angle = np.arctan2(mean_y, mean_x)
    if mean_angle < 0:
        mean_angle += 2 * np.pi

    # Convert mean angle back to time
    mean_seconds = mean_angle * period_seconds / (2 * np.pi)
    mean_time = pd.to_timedelta(mean_seconds, unit='s')

    # Circular standard deviation
    # R close to 1 means times are concentrated, close to 0 means dispersed
    circular_std_rad = np.sqrt(-2 * np.log(R)) if R > 0 else np.pi

    # Convert to time units
    circular_std_sec = circular_std_rad * (period_seconds / (2 * np.pi))
    circular_std_time = pd.to_timedelta(circular_std_sec, unit='s')

    logger.debug("Mean time calculated: %s ± %s", mean_time, circular_std_time)
    return mean_time, circular_std_time


def assign_season(date: Union[pd.Timestamp, Any]) -> str:
    """
    Assign meteorological season to a given date.

    Uses fixed date ranges:
    - Spring: March 1 - May 31
    - Summer: June 1 - August 31
    - Fall: September 1 - November 30
    - Winter: December 1 - February 28/29

    Args:
        date (pd.Timestamp or datetime): Date to classify

    Returns:
        str: Season name ('Spring', 'Summer', 'Fall', 'Winter')
    """
    Y = 2024  # Dummy year for date comparison (handles leap years)

    # Define season boundaries
    spring = (pd.Timestamp(f"{Y}-03-01"), pd.Timestamp(f"{Y}-05-31"))
    summer = (pd.Timestamp(f"{Y}-06-01"), pd.Timestamp(f"{Y}-08-31"))
    fall = (pd.Timestamp(f"{Y}-09-01"), pd.Timestamp(f"{Y}-11-30"))
    # Winter is the remaining months (Dec, Jan, Feb)

    # Create date with dummy year for comparison
    d = pd.Timestamp(f"{Y}-{date.month:02d}-{date.day:02d}")

    if spring[0] <= d <= spring[1]:
        return "Spring"
    elif summer[0] <= d <= summer[1]:
        return "Summer"
    elif fall[0] <= d <= fall[1]:
        return "Fall"
    else:
        return "Winter"


if __name__ == "__main__":
    # Load and prepare sample data
    df = pd.read_json("../data/environmental_sensor_data.json")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Initialize analyzer
    weather_analyzer = WeatherAnalyzer(df)
    cols = ["temperature_c", "humidity_percent", "air_pressure_hpa"]
    for col in cols:
        dft = df.dropna(subset=[col]).copy()
        periods = weather_analyzer.identify_cycles(dft[col])
    summary = weather_analyzer.get_data_summary()
    anomalies = weather_analyzer.get_anomalies("temperature_c")
    trends = weather_analyzer.get_trends("temperature_c")
    trends2 = weather_analyzer.get_trends("air_pressure_hpa")
    mov_avg = weather_analyzer.calculate_moving_averages("temperature_c_filled")
    corr_matrix = weather_analyzer.get_correlation_matrix("pearson")
    corr_matrix2 = weather_analyzer.get_correlation_matrix("pearson", lag = 16, lag_column="temperature_c_filled")