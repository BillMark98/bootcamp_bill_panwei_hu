"""
Outlier Detection and Handling Module
Homework 7 - Stage 07: Outliers, Risk, and Assumptions
Author: Panwei Hu
Date: 2025-08-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple, Any
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


class OutlierDetector:
    """Comprehensive outlier detection using multiple methods"""
    
    @staticmethod
    def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
        -----------
        series : pd.Series
            Numeric series to evaluate
        k : float
            Multiplier for IQR to set fences (default 1.5)
            
        Returns:
        --------
        pd.Series
            Boolean mask where True indicates an outlier
            
        Assumptions:
        - Distribution is reasonably summarized by quartiles
        - k=1.5 is standard, but can be adjusted for strictness
        - Works well for skewed distributions
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            # Handle case where IQR is zero (all values the same)
            return pd.Series(False, index=series.index)
        
        lower_fence = q1 - k * iqr
        upper_fence = q3 + k * iqr
        
        outliers = (series < lower_fence) | (series > upper_fence)
        
        return outliers
    
    @staticmethod
    def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Parameters:
        -----------
        series : pd.Series
            Numeric series to evaluate
        threshold : float
            Z-score threshold (default 3.0)
            
        Returns:
        --------
        pd.Series
            Boolean mask where True indicates an outlier
            
        Assumptions:
        - Data is approximately normally distributed
        - Sensitive to heavy tails and extreme values
        - Mean and std are representative of the distribution
        """
        if len(series.dropna()) < 2:
            return pd.Series(False, index=series.index)
        
        mu = series.mean()
        sigma = series.std(ddof=0)
        
        if sigma == 0:
            # Handle case where std is zero
            return pd.Series(False, index=series.index)
        
        z_scores = np.abs((series - mu) / sigma)
        outliers = z_scores > threshold
        
        return outliers
    
    @staticmethod
    def detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """
        Detect outliers using Modified Z-score (based on median).
        More robust to outliers than standard Z-score.
        
        Parameters:
        -----------
        series : pd.Series
            Numeric series to evaluate
        threshold : float
            Modified Z-score threshold (default 3.5)
            
        Returns:
        --------
        pd.Series
            Boolean mask where True indicates an outlier
        """
        if len(series.dropna()) < 2:
            return pd.Series(False, index=series.index)
        
        median = series.median()
        mad = np.median(np.abs(series - median))
        
        if mad == 0:
            # Use standard deviation as fallback
            mad = series.std() * 0.6745  # 0.6745 is the 75th percentile of standard normal
        
        if mad == 0:
            return pd.Series(False, index=series.index)
        
        modified_z_scores = 0.6745 * (series - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        
        return outliers
    
    @staticmethod
    def detect_outliers_isolation_forest(series: pd.Series, contamination: float = 0.1) -> pd.Series:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
        -----------
        series : pd.Series
            Numeric series to evaluate
        contamination : float
            Expected proportion of outliers (default 0.1)
            
        Returns:
        --------
        pd.Series
            Boolean mask where True indicates an outlier
        """
        if len(series.dropna()) < 10:
            return pd.Series(False, index=series.index)
        
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(X) == -1
        
        return pd.Series(outliers, index=series.index)
    
    @staticmethod
    def detect_outliers_percentile(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """
        Detect outliers using percentile method.
        
        Parameters:
        -----------
        series : pd.Series
            Numeric series to evaluate
        lower : float
            Lower percentile threshold (default 0.01)
        upper : float
            Upper percentile threshold (default 0.99)
            
        Returns:
        --------
        pd.Series
            Boolean mask where True indicates an outlier
        """
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        
        outliers = (series < lower_bound) | (series > upper_bound)
        
        return outliers


class OutlierHandler:
    """Methods for handling detected outliers"""
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
        """
        Remove outliers from DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        outlier_mask : pd.Series
            Boolean mask indicating outliers
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with outliers removed
        """
        return df[~outlier_mask].copy()
    
    @staticmethod
    def winsorize_series(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
        """
        Winsorize series by clipping extreme values to specified percentiles.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
        lower : float
            Lower percentile for clipping (default 0.05)
        upper : float
            Upper percentile for clipping (default 0.95)
            
        Returns:
        --------
        pd.Series
            Winsorized series
        """
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    @staticmethod
    def cap_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
        """
        Cap outliers using IQR method.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
        k : float
            IQR multiplier (default 1.5)
            
        Returns:
        --------
        pd.Series
            Series with outliers capped
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    @staticmethod
    def replace_outliers_median(series: pd.Series, outlier_mask: pd.Series) -> pd.Series:
        """
        Replace outliers with median value.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
        outlier_mask : pd.Series
            Boolean mask indicating outliers
            
        Returns:
        --------
        pd.Series
            Series with outliers replaced by median
        """
        result = series.copy()
        median_value = series[~outlier_mask].median()
        result[outlier_mask] = median_value
        
        return result
    
    @staticmethod
    def replace_outliers_interpolation(series: pd.Series, outlier_mask: pd.Series, method: str = 'linear') -> pd.Series:
        """
        Replace outliers using interpolation.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
        outlier_mask : pd.Series
            Boolean mask indicating outliers
        method : str
            Interpolation method (default 'linear')
            
        Returns:
        --------
        pd.Series
            Series with outliers replaced by interpolated values
        """
        result = series.copy()
        result[outlier_mask] = np.nan
        
        return result.interpolate(method=method)


class SensitivityAnalyzer:
    """Analyze sensitivity of results to outlier treatment"""
    
    @staticmethod
    def compare_summary_statistics(series: pd.Series, treatments: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compare summary statistics across different outlier treatments.
        
        Parameters:
        -----------
        series : pd.Series
            Original series
        treatments : Dict[str, pd.Series]
            Dictionary of treatment name -> treated series
            
        Returns:
        --------
        pd.DataFrame
            Comparison of summary statistics
        """
        results = {}
        
        # Original series
        results['original'] = series.describe()[['count', 'mean', '50%', 'std', 'min', 'max']].rename({'50%': 'median'})
        
        # Treated series
        for name, treated_series in treatments.items():
            results[name] = treated_series.describe()[['count', 'mean', '50%', 'std', 'min', 'max']].rename({'50%': 'median'})
        
        comparison_df = pd.DataFrame(results).T
        
        # Calculate percentage changes from original
        for col in ['mean', 'median', 'std']:
            comparison_df[f'{col}_pct_change'] = ((comparison_df[col] - comparison_df.loc['original', col]) / 
                                                 comparison_df.loc['original', col] * 100)
        
        return comparison_df
    
    @staticmethod
    def compare_regression_results(X: np.ndarray, y: np.ndarray, treatments: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """
        Compare regression results across different outlier treatments.
        
        Parameters:
        -----------
        X : np.ndarray
            Original X data
        y : np.ndarray
            Original y data
        treatments : Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary of treatment name -> (X_treated, y_treated)
            
        Returns:
        --------
        pd.DataFrame
            Comparison of regression results
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        results = {}
        
        # Original data
        model_orig = LinearRegression().fit(X, y)
        y_pred_orig = model_orig.predict(X)
        
        results['original'] = {
            'slope': model_orig.coef_[0] if len(model_orig.coef_) == 1 else model_orig.coef_,
            'intercept': model_orig.intercept_,
            'r2': r2_score(y, y_pred_orig),
            'mae': mean_absolute_error(y, y_pred_orig),
            'mse': mean_squared_error(y, y_pred_orig),
            'n_samples': len(y)
        }
        
        # Treated data
        for name, (X_treated, y_treated) in treatments.items():
            model = LinearRegression().fit(X_treated, y_treated)
            y_pred = model.predict(X_treated)
            
            results[name] = {
                'slope': model.coef_[0] if len(model.coef_) == 1 else model.coef_,
                'intercept': model.intercept_,
                'r2': r2_score(y_treated, y_pred),
                'mae': mean_absolute_error(y_treated, y_pred),
                'mse': mean_squared_error(y_treated, y_pred),
                'n_samples': len(y_treated)
            }
        
        return pd.DataFrame(results).T


class OutlierVisualizer:
    """Visualization tools for outlier analysis"""
    
    @staticmethod
    def plot_outlier_detection(series: pd.Series, outlier_masks: Dict[str, pd.Series], figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive visualization of outlier detection results.
        
        Parameters:
        -----------
        series : pd.Series
            Original data series
        outlier_masks : Dict[str, pd.Series]
            Dictionary of method name -> outlier mask
        figsize : Tuple[int, int]
            Figure size (default (15, 10))
        """
        n_methods = len(outlier_masks)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Outlier Detection Analysis', fontsize=16, fontweight='bold')
        
        # 1. Box plot
        ax1 = axes[0, 0]
        ax1.boxplot(series.dropna(), vert=True)
        ax1.set_title('Box Plot - IQR Method Reference')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram with outliers highlighted
        ax2 = axes[0, 1]
        ax2.hist(series.dropna(), bins=30, alpha=0.7, edgecolor='black', label='All Data')
        
        # Highlight outliers for first method
        if outlier_masks:
            first_method = list(outlier_masks.keys())[0]
            outliers = series[outlier_masks[first_method]]
            if not outliers.empty:
                ax2.hist(outliers, bins=30, alpha=0.8, color='red', edgecolor='darkred', 
                        label=f'Outliers ({first_method})')
        
        ax2.set_title('Distribution with Outliers Highlighted')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot (if series has index that can be used as x)
        ax3 = axes[1, 0]
        x_vals = range(len(series)) if series.index.dtype == 'object' else series.index
        ax3.scatter(x_vals, series, alpha=0.6, label='All Data')
        
        # Highlight outliers
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, (method, mask) in enumerate(outlier_masks.items()):
            if i >= len(colors):
                break
            outliers_x = [x_vals[j] for j in range(len(mask)) if mask.iloc[j]]
            outliers_y = series[mask]
            if not outliers_y.empty:
                ax3.scatter(outliers_x, outliers_y, color=colors[i], s=100, 
                           marker='x', label=f'{method} outliers', linewidth=2)
        
        ax3.set_title('Time Series with Outliers Marked')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Method comparison
        ax4 = axes[1, 1]
        method_counts = {method: mask.sum() for method, mask in outlier_masks.items()}
        
        methods = list(method_counts.keys())
        counts = list(method_counts.values())
        
        bars = ax4.bar(methods, counts, alpha=0.7, edgecolor='black')
        ax4.set_title('Outliers Detected by Method')
        ax4.set_xlabel('Detection Method')
        ax4.set_ylabel('Number of Outliers')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_treatment_comparison(original: pd.Series, treatments: Dict[str, pd.Series], figsize: Tuple[int, int] = (15, 8)):
        """
        Visualize the impact of different outlier treatments.
        
        Parameters:
        -----------
        original : pd.Series
            Original data series
        treatments : Dict[str, pd.Series]
            Dictionary of treatment name -> treated series
        figsize : Tuple[int, int]
            Figure size (default (15, 8))
        """
        n_treatments = len(treatments)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Outlier Treatment Comparison', fontsize=16, fontweight='bold')
        
        # 1. Box plots comparison
        ax1 = axes[0]
        data_to_plot = [original] + list(treatments.values())
        labels = ['Original'] + list(treatments.keys())
        
        ax1.boxplot(data_to_plot, labels=labels)
        ax1.set_title('Distribution Comparison')
        ax1.set_ylabel('Value')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Histograms overlay
        ax2 = axes[1]
        ax2.hist(original, bins=30, alpha=0.5, label='Original', edgecolor='black')
        
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, (name, treated) in enumerate(treatments.items()):
            if i < len(colors):
                ax2.hist(treated, bins=30, alpha=0.5, label=name, 
                        edgecolor='black', color=colors[i])
        
        ax2.set_title('Distribution Overlay')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Summary statistics comparison
        ax3 = axes[2]
        stats_data = {'Original': original}
        stats_data.update(treatments)
        
        means = [data.mean() for data in stats_data.values()]
        stds = [data.std() for data in stats_data.values()]
        labels = list(stats_data.keys())
        
        x_pos = np.arange(len(labels))
        ax3_twin = ax3.twinx()
        
        bars1 = ax3.bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7, color='skyblue')
        bars2 = ax3_twin.bar(x_pos + 0.2, stds, 0.4, label='Std Dev', alpha=0.7, color='lightcoral')
        
        ax3.set_xlabel('Treatment')
        ax3.set_ylabel('Mean', color='skyblue')
        ax3_twin.set_ylabel('Standard Deviation', color='lightcoral')
        ax3.set_title('Statistics Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, means):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4e}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2, stds):
            height = bar.get_height()
            ax3_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Convenience functions for direct import
def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method"""
    return OutlierDetector.detect_outliers_iqr(series, k)

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method"""
    return OutlierDetector.detect_outliers_zscore(series, threshold)

def winsorize_series(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """Winsorize series by clipping to percentiles"""
    return OutlierHandler.winsorize_series(series, lower, upper)

def remove_outliers(df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
    """Remove outliers from DataFrame"""
    return OutlierHandler.remove_outliers(df, outlier_mask) 