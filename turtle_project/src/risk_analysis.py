"""
Financial Risk Analysis and Outlier Detection Module
Turtle Trading Project - Stage 07: Outliers, Risk, and Assumptions
Author: Panwei Hu
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FinancialOutlierDetector:
    """Outlier detection specialized for financial time series data"""
    
    @staticmethod
    def detect_price_outliers_iqr(prices: pd.Series, k: float = 1.5) -> pd.Series:
        """
        Detect price outliers using IQR method on returns.
        More appropriate for financial data than raw prices.
        """
        returns = prices.pct_change().dropna()
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return pd.Series(False, index=prices.index)
        
        lower_fence = q1 - k * iqr
        upper_fence = q3 + k * iqr
        
        # Map back to price index
        return_outliers = (returns < lower_fence) | (returns > upper_fence)
        price_outliers = pd.Series(False, index=prices.index)
        price_outliers.iloc[1:] = return_outliers.values
        
        return price_outliers
    
    @staticmethod
    def detect_volume_outliers(volume: pd.Series, method: str = 'iqr', k: float = 2.0) -> pd.Series:
        """
        Detect volume outliers (typically only high volume is concerning).
        """
        if method == 'iqr':
            q1 = volume.quantile(0.25)
            q3 = volume.quantile(0.75)
            iqr = q3 - q1
            upper_fence = q3 + k * iqr
            return volume > upper_fence
        elif method == 'percentile':
            threshold = volume.quantile(0.95)  # Top 5% volume
            return volume > threshold
        else:
            raise ValueError("Method must be 'iqr' or 'percentile'")
    
    @staticmethod
    def detect_return_outliers_zscore(returns: pd.Series, threshold: float = 3.0, 
                                     rolling_window: Optional[int] = None) -> pd.Series:
        """
        Detect return outliers using Z-score, optionally with rolling statistics.
        """
        if rolling_window:
            # Use rolling statistics for time-varying volatility
            rolling_mean = returns.rolling(window=rolling_window).mean()
            rolling_std = returns.rolling(window=rolling_window).std()
            z_scores = np.abs((returns - rolling_mean) / rolling_std)
        else:
            # Use full sample statistics
            z_scores = np.abs((returns - returns.mean()) / returns.std())
        
        return z_scores > threshold
    
    @staticmethod
    def detect_gap_events(prices: pd.Series, gap_threshold: float = 0.05) -> pd.Series:
        """
        Detect price gaps (overnight jumps) that might indicate data issues or major events.
        """
        returns = prices.pct_change().dropna()
        return np.abs(returns) > gap_threshold
    
    @staticmethod
    def detect_volatility_outliers(returns: pd.Series, window: int = 20, threshold: float = 3.0) -> pd.Series:
        """
        Detect periods of unusually high volatility.
        """
        rolling_vol = returns.rolling(window=window).std()
        vol_zscore = np.abs((rolling_vol - rolling_vol.mean()) / rolling_vol.std())
        return vol_zscore > threshold


class TurtleRiskAnalyzer:
    """Risk analysis specifically for Turtle Trading strategy"""
    
    @staticmethod
    def analyze_signal_quality(df: pd.DataFrame, price_col: str = 'adj_close', 
                              signal_cols: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the quality of trading signals in the presence of outliers.
        """
        if signal_cols is None:
            signal_cols = ['long_entry_20', 'short_entry_20', 'long_exit_10', 'short_exit_10']
        
        analysis = {}
        
        # Detect price outliers
        price_outliers = FinancialOutlierDetector.detect_price_outliers_iqr(df[price_col])
        
        # Analyze signal contamination
        for signal_col in signal_cols:
            if signal_col in df.columns:
                signals = df[signal_col]
                contaminated_signals = signals & price_outliers
                
                analysis[signal_col] = {
                    'total_signals': signals.sum(),
                    'contaminated_signals': contaminated_signals.sum(),
                    'contamination_rate': contaminated_signals.sum() / max(signals.sum(), 1) * 100,
                    'clean_signals': signals.sum() - contaminated_signals.sum()
                }
        
        return analysis
    
    @staticmethod
    def calculate_risk_metrics(returns: pd.Series, confidence_levels: List[float] = [0.05, 0.01]) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for financial returns.
        """
        if len(returns) == 0:
            return {}
        
        metrics = {}
        
        # Basic statistics
        metrics['mean_return'] = returns.mean()
        metrics['volatility'] = returns.std()
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Risk metrics
        for cl in confidence_levels:
            metrics[f'VaR_{int(cl*100)}%'] = returns.quantile(cl)
            tail_returns = returns[returns <= returns.quantile(cl)]
            if len(tail_returns) > 0:
                metrics[f'CVaR_{int(cl*100)}%'] = tail_returns.mean()
        
        # Performance metrics
        if metrics['volatility'] != 0:
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['volatility']
        
        # Drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Extreme event metrics
        metrics['max_loss'] = returns.min()
        metrics['max_gain'] = returns.max()
        metrics['positive_days_pct'] = (returns > 0).mean() * 100
        
        return metrics
    
    @staticmethod
    def stress_test_signals(df: pd.DataFrame, scenarios: Dict[str, Dict] = None) -> pd.DataFrame:
        """
        Stress test trading signals under different market scenarios.
        """
        if scenarios is None:
            scenarios = {
                'market_crash': {'return_shock': -0.10, 'volatility_multiplier': 3.0},
                'flash_crash': {'return_shock': -0.20, 'volatility_multiplier': 5.0},
                'high_volatility': {'return_shock': 0.0, 'volatility_multiplier': 2.0},
                'trending_market': {'return_shock': 0.02, 'volatility_multiplier': 0.5}
            }
        
        results = []
        
        for scenario_name, params in scenarios.items():
            stressed_df = df.copy()
            
            # Apply stress scenario
            if 'return_shock' in params:
                stressed_df['adj_close'] *= (1 + params['return_shock'])
            
            if 'volatility_multiplier' in params:
                returns = stressed_df['adj_close'].pct_change()
                mean_return = returns.mean()
                stressed_returns = mean_return + (returns - mean_return) * params['volatility_multiplier']
                stressed_df['adj_close'] = stressed_df['adj_close'].iloc[0] * (1 + stressed_returns).cumprod()
                stressed_df['adj_close'].iloc[0] = df['adj_close'].iloc[0]  # Keep first price
            
            # Recalculate signals (simplified - assumes we have the technical indicators)
            if 'donchian_high_20' in stressed_df.columns:
                # Recalculate Donchian channels
                for symbol in stressed_df['symbol'].unique() if 'symbol' in stressed_df.columns else [None]:
                    if symbol:
                        mask = stressed_df['symbol'] == symbol
                        prices = stressed_df.loc[mask, 'adj_close']
                    else:
                        prices = stressed_df['adj_close']
                    
                    stressed_df.loc[mask if symbol else slice(None), 'donchian_high_20'] = prices.rolling(20).max()
                    stressed_df.loc[mask if symbol else slice(None), 'donchian_low_20'] = prices.rolling(20).min()
                    stressed_df.loc[mask if symbol else slice(None), 'long_entry_20'] = (
                        prices > stressed_df.loc[mask if symbol else slice(None), 'donchian_high_20'].shift(1)
                    )
            
            # Calculate metrics for this scenario
            returns = stressed_df['adj_close'].pct_change().dropna()
            risk_metrics = TurtleRiskAnalyzer.calculate_risk_metrics(returns)
            risk_metrics['scenario'] = scenario_name
            
            results.append(risk_metrics)
        
        return pd.DataFrame(results)


class FinancialRiskVisualizer:
    """Visualization tools for financial risk analysis"""
    
    @staticmethod
    def plot_price_outliers(df: pd.DataFrame, price_col: str = 'adj_close', 
                           date_col: str = 'date', figsize: Tuple[int, int] = (15, 10)):
        """
        Comprehensive visualization of price outliers in financial time series.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Financial Time Series Outlier Analysis', fontsize=16, fontweight='bold')
        
        # Detect outliers
        price_outliers = FinancialOutlierDetector.detect_price_outliers_iqr(df[price_col])
        returns = df[price_col].pct_change().dropna()
        return_outliers = FinancialOutlierDetector.detect_return_outliers_zscore(returns)
        
        # 1. Price time series with outliers
        ax1 = axes[0, 0]
        ax1.plot(df[date_col], df[price_col], 'b-', alpha=0.7, linewidth=1, label='Price')
        
        outlier_dates = df[price_outliers][date_col]
        outlier_prices = df[price_outliers][price_col]
        if not outlier_dates.empty:
            ax1.scatter(outlier_dates, outlier_prices, color='red', s=50, 
                       marker='o', alpha=0.8, label='Price Outliers')
        
        ax1.set_title('Price Series with Outliers')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns with outliers
        ax2 = axes[0, 1]
        return_dates = df[date_col].iloc[1:]  # Skip first date for returns
        ax2.plot(return_dates, returns, 'g-', alpha=0.7, linewidth=1, label='Returns')
        
        outlier_return_dates = return_dates[return_outliers]
        outlier_returns = returns[return_outliers]
        if not outlier_return_dates.empty:
            ax2.scatter(outlier_return_dates, outlier_returns, color='red', s=50,
                       marker='o', alpha=0.8, label='Return Outliers')
        
        ax2.set_title('Daily Returns with Outliers')
        ax2.set_ylabel('Daily Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Return distribution
        ax3 = axes[1, 0]
        ax3.hist(returns.dropna(), bins=50, alpha=0.7, density=True, edgecolor='black', label='All Returns')
        
        if return_outliers.any():
            ax3.hist(returns[return_outliers], bins=20, alpha=0.8, density=True,
                    color='red', edgecolor='darkred', label='Outlier Returns')
        
        # Overlay normal distribution
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = stats.norm.pdf(x_norm, returns.mean(), returns.std())
        ax3.plot(x_norm, y_norm, 'k--', linewidth=2, label='Normal Distribution')
        
        ax3.set_title('Return Distribution Analysis')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Rolling volatility
        ax4 = axes[1, 1]
        rolling_vol = returns.rolling(window=20).std()
        vol_outliers = FinancialOutlierDetector.detect_volatility_outliers(returns)
        
        ax4.plot(return_dates, rolling_vol, 'purple', linewidth=1.5, label='20-day Rolling Volatility')
        
        vol_outlier_dates = return_dates[vol_outliers.iloc[1:]]  # Align with returns index
        vol_outlier_values = rolling_vol[vol_outliers.iloc[1:]]
        if not vol_outlier_dates.empty:
            ax4.scatter(vol_outlier_dates, vol_outlier_values, color='red', s=50,
                       marker='s', alpha=0.8, label='Volatility Outliers')
        
        ax4.set_title('Rolling Volatility with Outliers')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Volatility')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_risk_metrics_comparison(risk_metrics_dict: Dict[str, Dict], figsize: Tuple[int, int] = (16, 10)):
        """
        Compare risk metrics across different scenarios or treatments.
        """
        risk_df = pd.DataFrame(risk_metrics_dict).T
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Risk Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Key metrics to plot
        metrics_to_plot = [
            ('mean_return', 'Mean Return', axes[0, 0]),
            ('volatility', 'Volatility', axes[0, 1]),
            ('sharpe_ratio', 'Sharpe Ratio', axes[0, 2]),
            ('VaR_5%', 'VaR (5%)', axes[1, 0]),
            ('max_drawdown', 'Max Drawdown', axes[1, 1]),
            ('skewness', 'Skewness', axes[1, 2])
        ]
        
        for metric, title, ax in metrics_to_plot:
            if metric in risk_df.columns:
                risk_df[metric].plot(kind='bar', ax=ax, alpha=0.7, color='steelblue')
                ax.set_title(title)
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()


# Convenience functions for direct import
def detect_price_outliers(prices: pd.Series, method: str = 'iqr', **kwargs) -> pd.Series:
    """Detect outliers in price series"""
    if method == 'iqr':
        return FinancialOutlierDetector.detect_price_outliers_iqr(prices, **kwargs)
    else:
        raise ValueError("Currently only 'iqr' method is supported")

def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate risk metrics for returns"""
    return TurtleRiskAnalyzer.calculate_risk_metrics(returns)

def analyze_signal_quality(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Analyze quality of trading signals"""
    return TurtleRiskAnalyzer.analyze_signal_quality(df, **kwargs) 