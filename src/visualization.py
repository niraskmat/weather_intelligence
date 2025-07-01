"""
Weather Data Visualization Engine

This module provides comprehensive visualization capabilities for weather data analysis,
including time series plots, correlation heatmaps, distribution analysis, seasonal
decomposition, and anomaly detection visualizations.
"""
import io
import base64
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Any

from src.analysis_engine import WeatherAnalyzer


# Configure logger for this module
logger = logging.getLogger(__name__)


class VisualizationEngine(WeatherAnalyzer):
    """
    A visualization engine for weather data analysis.

    This class extends WeatherAnalyzer to provide various plotting capabilities
    including time series analysis, correlation visualization, distribution fitting,
    seasonal decomposition, and anomaly detection plots.

    Attributes:
        analyzer (WeatherAnalyzer): The underlying weather analysis engine
    """

    def __init__(self, analyzer: WeatherAnalyzer) -> None:
        """
        Initialize the visualization engine with a weather analyzer.

        Args:
            analyzer (WeatherAnalyzer): Pre-configured weather analyzer instance
        """
        logger.info("Initializing VisualizationEngine")
        self.analyzer = analyzer
        logger.debug(f"Visualization engine initialized with analyzer for {len(analyzer.df)} data points")

    def timeseries(self, col: str) -> str:
        """
        Generate a time series plot with moving averages and trend analysis.

        Creates a comprehensive time series visualization showing the original data,
        7-day moving average, and 30-day moving average to highlight trends and patterns.

        Args:
            col (str): Column name to plot from the weather data

        Returns:
            str: Base64 encoded PNG image of the time series plot

        Raises:
            KeyError: If the specified column doesn't exist in the dataset
            ValueError: If the column contains no valid data
        """
        logger.info(f"Generating time series plot for column: {col}")

        try:
            df = self.analyzer.df

            # Validate column exists and has data
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in dataset")

            if df[col].isna().all():
                raise ValueError(f"Column '{col}' contains no valid data")

            # Calculate moving averages for trend analysis
            logger.debug(f"Calculating moving averages for {col}")
            res = self.analyzer.calculate_moving_averages(col)

            # Create the time series plot
            plt.figure(figsize=(15, 8))

            # Plot original data with transparency
            plt.plot(df.index, df[col], label="Observed", alpha=0.6, color='gray')

            # Plot moving averages for trend identification
            plt.plot(df.index, res["7day"][col], label="7-day Moving Avg",
                     color="blue", linestyle='--', linewidth=2)
            plt.plot(df.index, res["30day"][col], label="30-day Moving Avg",
                     color="red", linestyle='--', linewidth=2)

            # Customize plot appearance
            plt.title(f"Time Series Analysis: {col}", fontsize=16, fontweight='bold')
            plt.xlabel("Date", fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            logger.debug(f"Time series plot generated successfully for {col}")

            # Convert plot to base64 image
            img_base64 = self.render_plot_to_base64()
            logger.info(f"Time series visualization completed for {col}")

            return img_base64

        except Exception as e:
            logger.error(f"Failed to generate time series plot for {col}: {str(e)}")
            raise

    def correlation(self, method: str = 'pearson', lag: int = 0, lag_column: str = None) -> str:
        """
        Generate a correlation heatmap for all numeric variables in the dataset.

        Creates a color-coded heatmap showing correlations between all numeric
        variables using the specified correlation method.

        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            str: Base64 encoded PNG image of the correlation heatmap

        Raises:
            ValueError: If the correlation method is not supported
        """
        logger.info(f"Generating correlation heatmap using method: {method}")

        try:
            # Validate correlation method
            valid_methods = ['pearson', 'spearman', 'kendall']
            if method not in valid_methods:
                raise ValueError(f"Method '{method}' not supported. Use one of: {valid_methods}")

            # Calculate correlation matrix
            logger.debug("Computing correlation matrix")
            corr_matrix = self.analyzer.get_correlation_matrix(method, lag, lag_column)

            # Create heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle

            sns.heatmap(corr_matrix,
                        annot=True,
                        fmt=".2f",
                        cmap="coolwarm",
                        center=0,
                        mask=mask,
                        square=True,
                        cbar_kws={"shrink": 0.8})

            plt.title(f"Correlation Heatmap ({method.capitalize()}, {lag} hour lag, fixed {lag_column})",
                      fontsize=14, fontweight='bold')
            plt.tight_layout()

            logger.debug("Correlation heatmap generated successfully")

            # Convert to base64 image
            img_base64 = self.render_plot_to_base64()
            logger.info(f"Correlation heatmap completed using {method} method, {lag} lag, {lag_column} lag column")

            return img_base64

        except Exception as e:
            logger.error(f"Failed to generate correlation heatmap: {str(e)}")
            raise

    def best_distribution(self, col: str) -> Tuple[str, str]:
        """
        Identify and visualize the best-fitting statistical distribution for a column.

        Fits multiple statistical distributions to the data and returns the best fit
        along with a visualization comparing the data histogram to the fitted distribution.

        Args:
            col (str): Column name to analyze

        Returns:
            Tuple[str, str]: Best distribution name and base64 encoded plot image

        Raises:
            KeyError: If the specified column doesn't exist
            ValueError: If the column has insufficient data for distribution fitting
        """
        logger.info(f"Finding best distribution fit for column: {col}")

        try:
            # Validate column
            if col not in self.analyzer.df.columns:
                raise KeyError(f"Column '{col}' not found in dataset")

            # Fit distributions and find best match
            logger.debug(f"Fitting distributions to {col}")
            fits = self.analyzer.fit_distributions(col)

            if not fits:
                raise ValueError(f"No valid distributions could be fitted to column '{col}'")

            best_fit, _ = self.analyzer.get_best_fit(fits)

            # Generate visualization for the best fit
            logger.debug(f"Best fit distribution: {best_fit}")
            fit = fits[0]  # Best fit is the first in the sorted list
            img_base64 = self.plot_distribution(fit, col)

            logger.info(f"Best distribution analysis completed for {col}: {best_fit}")

            return best_fit, img_base64

        except Exception as e:
            logger.error(f"Failed to find best distribution for {col}: {str(e)}")
            raise

    def distributions(self, col: str) -> Dict[str, Any]:
        """
        Generate visualizations for all fitted distributions of a column.

        Fits multiple statistical distributions and creates individual plots
        for each distribution, allowing comparison of different fits.

        Args:
            col (str): Column name to analyze

        Returns:
            Dict[str, Any]: Dictionary containing best fit name and all distribution plots

        Raises:
            KeyError: If the specified column doesn't exist
            ValueError: If no distributions can be fitted to the data
        """
        logger.info(f"Generating all distribution plots for column: {col}")

        try:
            # Fit all distributions
            logger.debug(f"Fitting multiple distributions to {col}")
            fits = self.analyzer.fit_distributions(col)

            if not fits:
                raise ValueError(f"No distributions could be fitted to column '{col}'")

            best_fit, params = self.analyzer.get_best_fit(fits)

            # Generate plots for each distribution
            results = {"best_fit": best_fit, "distributions": {}}

            logger.debug(f"Generating plots for {len(fits)} distributions")
            for i, fit in enumerate(fits):
                dist_name = fit[0]
                logger.debug(f"Plotting distribution {i + 1}/{len(fits)}: {dist_name}")
                results["distributions"][dist_name] = self.plot_distribution(fit, col)

            logger.info(f"All distribution plots completed for {col}. Best fit: {best_fit}")

            return results

        except Exception as e:
            logger.error(f"Failed to generate distribution plots for {col}: {str(e)}")
            raise

    def plot_distribution(self, fit: Tuple[str, float, Any, Tuple], col: str) -> str:
        """
        Create a distribution plot comparing data histogram with fitted distribution.

        Generates a plot showing the data histogram overlaid with the probability
        density function of the fitted statistical distribution.

        Args:
            fit (Tuple): Distribution fit information (name, score, test_stat, params)
            col (str): Column name being analyzed

        Returns:
            str: Base64 encoded PNG image of the distribution plot

        Raises:
            AttributeError: If the distribution is not available in scipy.stats
        """
        logger.debug(f"Plotting distribution fit for {col}")

        try:
            df = self.analyzer.df

            # Create x values for the distribution curve
            x_min, x_max = df[col].min(), df[col].max()
            x = np.linspace(x_min, x_max, 1000)

            # Extract distribution information
            dist_name, _, _, params = fit

            # Get the distribution from scipy.stats
            try:
                dist = getattr(stats, dist_name)
            except AttributeError:
                raise AttributeError(f"Distribution '{dist_name}' not found in scipy.stats")

            # Calculate probability density function
            pdf = dist.pdf(x, *params)

            # Create the plot
            plt.figure(figsize=(10, 6))

            # Plot histogram of actual data
            sns.histplot(df[col].dropna(), bins=50, stat="density",
                         label="Observed Data", color="skyblue", alpha=0.7)

            # Plot fitted distribution curve
            plt.plot(x, pdf, label=f"{dist_name} Fit", color="red", linewidth=2)

            # Customize plot
            plt.xlabel(col, fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.title(f"Distribution Fit: {dist_name} for {col}",
                      fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            logger.debug(f"Distribution plot created for {dist_name}")

            # Convert to base64 image
            img_base64 = self.render_plot_to_base64()

            return img_base64

        except Exception as e:
            logger.error(f"Failed to plot distribution {fit[0]} for {col}: {str(e)}")
            raise

    def decomposition(self, col: str) -> str:
        """
        Generate a seasonal decomposition plot for time series data.

        Creates a multi-panel plot showing the original time series decomposed
        into trend, seasonal, and residual components using STL decomposition.

        Args:
            col (str): Column name to decompose

        Returns:
            str: Base64 encoded PNG image of the decomposition plot

        Raises:
            KeyError: If the column doesn't exist or hasn't been decomposed
            AttributeError: If decomposition data is not available
        """
        logger.info(f"Generating seasonal decomposition plot for column: {col}")

        try:
            # Check if decomposition exists
            if not hasattr(self.analyzer, 'decomposition') or col not in self.analyzer.decomposition:
                raise KeyError(f"Decomposition not available for column '{col}'. "
                               "Run decomposition analysis first.")

            decomp = self.analyzer.decomposition[col]

            # Set figure size for decomposition plot
            plt.rcParams["figure.figsize"] = (15, 10)

            # Generate the decomposition plot
            logger.debug("Creating decomposition plot")
            fig = decomp.plot()

            # Add overall title
            fig.suptitle(f"Seasonal Decomposition: {col}", fontsize=16, fontweight='bold')

            # Reset matplotlib parameters
            plt.rcdefaults()

            # Convert to base64 without using render_plot_to_base64 (since we have fig object)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            buf.seek(0)

            img_base64 = base64.b64encode(buf.read()).decode('utf-8')

            logger.info(f"Seasonal decomposition plot completed for {col}")

            return img_base64

        except Exception as e:
            logger.error(f"Failed to generate decomposition plot for {col}: {str(e)}")
            raise

    def anomalies(self, col: str) -> str:
        """
        Generate an anomaly detection visualization.

        Creates a scatter plot highlighting detected anomalies in the time series
        data, showing both the normal data points and the identified outliers.

        Args:
            col (str): Column name to analyze for anomalies

        Returns:
            str: Base64 encoded PNG image of the anomaly detection plot

        Raises:
            KeyError: If the column doesn't exist
            ValueError: If no anomalies are detected or data is insufficient
        """
        logger.info(f"Generating anomaly detection plot for column: {col}")

        try:
            # Validate column exists
            if col not in self.analyzer.df.columns:
                raise KeyError(f"Column '{col}' not found in dataset")

            # Get anomalies from analyzer
            logger.debug(f"Detecting anomalies in {col}")
            anomalies, _ = self.analyzer.get_anomalies(col)

            # Clean data for visualization
            df = self.analyzer.df.dropna(subset=[col]).copy()

            if df.empty:
                raise ValueError(f"No valid data available for column '{col}'")

            # Create the anomaly plot
            plt.figure(figsize=(15, 8))

            # Plot normal data points
            plt.plot(df.index, df[col], color='blue', alpha=0.7,
                     linewidth=1, label='Normal Data')

            # Highlight anomalies if any exist
            if not anomalies.empty:
                plt.scatter(x=anomalies.index, y=anomalies[col],
                            color="red", s=50, alpha=0.8,
                            label=f"Anomalies ({len(anomalies)})", zorder=5)
                logger.info(f"Found {len(anomalies)} anomalies in {col}")
            else:
                logger.info(f"No anomalies detected in {col}")

            # Customize plot
            plt.title(f'Anomaly Detection: {col}', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Convert to base64 image
            img_base64 = self.render_plot_to_base64()

            logger.info(f"Anomaly detection plot completed for {col}")

            return img_base64

        except Exception as e:
            logger.error(f"Failed to generate anomaly plot for {col}: {str(e)}")
            raise

    def render_plot_to_base64(self) -> str:
        """
        Convert the current matplotlib figure to a base64 encoded PNG image.

        This utility method handles the conversion of matplotlib plots to base64
        encoded strings for web display or API responses. It automatically closes
        the figure to free memory.

        Returns:
            str: Base64 encoded PNG image data

        Raises:
            RuntimeError: If no active matplotlib figure exists
        """
        logger.debug("Converting matplotlib figure to base64")

        try:
            # Check if there's an active figure
            if plt.get_fignums():
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                plt.close()  # Close figure to free memory
                buf.seek(0)

                # Encode to base64
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')

                logger.debug("Successfully converted plot to base64")
                return img_base64
            else:
                raise RuntimeError("No active matplotlib figure to convert")

        except Exception as e:
            logger.error(f"Failed to convert plot to base64: {str(e)}")
            # Ensure figure is closed even if conversion fails
            plt.close('all')
            raise