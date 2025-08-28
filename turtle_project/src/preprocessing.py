"""
Financial Data Preprocessing Module for Turtle Trading
Project: Turtle Trading Strategy Research
Author: Panwei Hu
Date: 2025-08-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class FinancialDataProcessor:
    """Specialized preprocessing for financial time series data"""
    
    @staticmethod
    def load_turtle_data(data_dir: Path) -> pd.DataFrame:
        """Load the most recent turtle trading dataset"""
        csv_files = list(data_dir.glob('turtle_universe*.csv'))
        
        if not csv_files:
            print("❌ No turtle universe data found. Run 04_data_acquisition.ipynb first.")
            return pd.DataFrame()
        
        # Load the most recent file
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"📊 Loading: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        print(f"✅ Loaded {len(df):,} records for {df['symbol'].nunique()} symbols")
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    
    @staticmethod
    def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality analysis for financial data"""
        if df.empty:
            return {}
            
        analysis = {}
        
        # Basic statistics
        analysis['shape'] = df.shape
        analysis['date_range'] = (df['date'].min(), df['date'].max())
        analysis['symbols'] = sorted(df['symbol'].unique().tolist())
        analysis['trading_days'] = df['date'].nunique()
        
        # Missing data analysis
        missing_by_symbol = df.groupby('symbol')['adj_close'].apply(lambda x: x.isnull().sum())
        analysis['missing_by_symbol'] = missing_by_symbol.to_dict()
        analysis['total_missing'] = df['adj_close'].isnull().sum()
        analysis['missing_percentage'] = (analysis['total_missing'] / len(df)) * 100
        
        # Price data quality
        analysis['negative_prices'] = (df['adj_close'] <= 0).sum()
        analysis['zero_prices'] = (df['adj_close'] == 0).sum()
        
        # Data gaps (missing trading days)
        gaps_by_symbol = {}
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.set_index('date')
            
            # Check for gaps in trading days (more than 3 days)
            date_diffs = symbol_data.index.to_series().diff()
            gaps = date_diffs[date_diffs > pd.Timedelta(days=3)]
            gaps_by_symbol[symbol] = len(gaps)
        
        analysis['data_gaps'] = gaps_by_symbol
        
        return analysis
    
    @staticmethod
    def handle_missing_prices(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Handle missing price data using financial-appropriate methods"""
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        if method == 'forward_fill':
            # Forward fill within each symbol (carry last price forward)
            df_clean['adj_close'] = df_clean.groupby('symbol')['adj_close'].fillna(method='ffill')
            print("✅ Applied forward fill for missing prices")
            
        elif method == 'interpolate':
            # Linear interpolation within each symbol
            df_clean['adj_close'] = df_clean.groupby('symbol')['adj_close'].apply(
                lambda x: x.interpolate(method='linear')
            )
            print("✅ Applied linear interpolation for missing prices")
            
        elif method == 'drop':
            # Drop rows with missing prices
            before_count = len(df_clean)
            df_clean = df_clean.dropna(subset=['adj_close'])
            after_count = len(df_clean)
            print(f"✅ Dropped {before_count - after_count} rows with missing prices")
        
        return df_clean
    
    @staticmethod
    def detect_price_anomalies(df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
        """Detect price anomalies using returns-based Z-score"""
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        # Calculate daily returns for each symbol
        df_clean['returns'] = df_clean.groupby('symbol')['adj_close'].pct_change()
        
        # Calculate Z-scores for returns
        df_clean['return_zscore'] = df_clean.groupby('symbol')['returns'].transform(
            lambda x: np.abs((x - x.mean()) / x.std())
        )
        
        # Flag anomalies
        df_clean['is_anomaly'] = df_clean['return_zscore'] > z_threshold
        
        anomaly_count = df_clean['is_anomaly'].sum()
        print(f"📊 Detected {anomaly_count} price anomalies (Z-score > {z_threshold})")
        
        if anomaly_count > 0:
            anomaly_summary = df_clean[df_clean['is_anomaly']].groupby('symbol').size()
            print("   Anomalies by symbol:")
            for symbol, count in anomaly_summary.items():
                print(f"     {symbol}: {count}")
        
        return df_clean
    
    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for Turtle Trading"""
        if df.empty:
            return df
            
        df_features = df.copy()
        
        print("🔧 Creating technical features for Turtle Trading...")
        
        # Calculate returns first (needed for several indicators)
        df_features['returns'] = df_features.groupby('symbol')['adj_close'].pct_change()
        
        for symbol in df_features['symbol'].unique():
            mask = df_features['symbol'] == symbol
            symbol_data = df_features[mask].copy()
            
            # Sort by date to ensure proper calculation
            symbol_data = symbol_data.sort_values('date')
            prices = symbol_data['adj_close']
            
            # 1. Simple Moving Averages
            symbol_data['sma_20'] = prices.rolling(window=20).mean()
            symbol_data['sma_50'] = prices.rolling(window=50).mean()
            
            # 2. Donchian Channels (core of Turtle Trading)
            symbol_data['donchian_high_20'] = prices.rolling(window=20).max()
            symbol_data['donchian_low_20'] = prices.rolling(window=20).min()
            symbol_data['donchian_mid_20'] = (symbol_data['donchian_high_20'] + symbol_data['donchian_low_20']) / 2
            
            # Long-term Donchian channels
            symbol_data['donchian_high_55'] = prices.rolling(window=55).max()
            symbol_data['donchian_low_55'] = prices.rolling(window=55).min()
            
            # 3. Average True Range (ATR) for position sizing
            # Using price as proxy for OHLC data
            high = prices
            low = prices
            close = prices
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            symbol_data['atr_20'] = true_range.rolling(window=20).mean()
            
            # 4. Price position within Donchian channel (0-1 scale)
            symbol_data['price_position'] = (prices - symbol_data['donchian_low_20']) / (
                symbol_data['donchian_high_20'] - symbol_data['donchian_low_20']
            )
            
            # 5. Turtle Trading signals
            symbol_data['long_entry_20'] = prices > symbol_data['donchian_high_20'].shift(1)
            symbol_data['short_entry_20'] = prices < symbol_data['donchian_low_20'].shift(1)
            symbol_data['long_exit_10'] = prices < symbol_data['donchian_low_20'].rolling(10).min().shift(1)
            symbol_data['short_exit_10'] = prices > symbol_data['donchian_high_20'].rolling(10).max().shift(1)
            
            # 6. Volatility measures
            symbol_data['volatility_20'] = symbol_data['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            # 7. Trend strength indicator
            symbol_data['trend_strength'] = (symbol_data['sma_20'] - symbol_data['sma_50']) / symbol_data['sma_50']
            
            # Update main dataframe
            df_features.loc[mask, symbol_data.columns] = symbol_data
        
        print(f"✅ Created technical features for {df_features['symbol'].nunique()} symbols")
        
        return df_features
    
    @staticmethod
    def align_data_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all symbols have data for the same date range"""
        if df.empty:
            return df
            
        # Find common date range across all symbols
        date_counts = df.groupby('date')['symbol'].count()
        max_symbols = df['symbol'].nunique()
        
        # Keep only dates where all symbols have data
        complete_dates = date_counts[date_counts == max_symbols].index
        
        df_aligned = df[df['date'].isin(complete_dates)].copy()
        
        original_days = df['date'].nunique()
        aligned_days = df_aligned['date'].nunique()
        
        print(f"📅 Date alignment: {original_days} → {aligned_days} trading days")
        print(f"   Removed {original_days - aligned_days} days with incomplete data")
        
        return df_aligned
    
    @staticmethod
    def validate_processed_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of processed financial data"""
        if df.empty:
            return {'valid': False, 'errors': ['Empty dataframe']}
        
        validation_results = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check required columns
        required_cols = ['symbol', 'date', 'adj_close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            validation_results['valid'] = False
        
        # Check for negative or zero prices
        if 'adj_close' in df.columns:
            invalid_prices = (df['adj_close'] <= 0).sum()
            if invalid_prices > 0:
                validation_results['warnings'].append(f"Found {invalid_prices} non-positive prices")
        
        # Check date continuity
        if 'date' in df.columns:
            df_sorted = df.sort_values(['symbol', 'date'])
            for symbol in df['symbol'].unique():
                symbol_data = df_sorted[df_sorted['symbol'] == symbol]
                date_gaps = symbol_data['date'].diff().dt.days
                large_gaps = (date_gaps > 7).sum()  # More than a week
                if large_gaps > 0:
                    validation_results['warnings'].append(f"{symbol}: {large_gaps} large date gaps")
        
        # Check for excessive missing values in technical indicators
        technical_cols = ['donchian_high_20', 'donchian_low_20', 'atr_20']
        for col in technical_cols:
            if col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct > 50:
                    validation_results['warnings'].append(f"{col}: {missing_pct:.1f}% missing values")
        
        return validation_results


class TurtleDataCleaner:
    """Convenience wrapper for common turtle trading data cleaning operations"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processor = FinancialDataProcessor()
    
    def load_and_clean(self, missing_method: str = 'forward_fill') -> pd.DataFrame:
        """Complete data loading and cleaning pipeline"""
        print("🐢 Starting Turtle Data Cleaning Pipeline...")
        
        # Load data
        df = self.processor.load_turtle_data(self.data_dir)
        if df.empty:
            return df
        
        # Analyze quality
        quality = self.processor.analyze_data_quality(df)
        print(f"📊 Data Quality: {quality['total_missing']} missing values ({quality['missing_percentage']:.2f}%)")
        
        # Clean missing values
        df_clean = self.processor.handle_missing_prices(df, method=missing_method)
        
        # Detect anomalies
        df_clean = self.processor.detect_price_anomalies(df_clean)
        
        # Create features
        df_processed = self.processor.create_technical_features(df_clean)
        
        # Validate
        validation = self.processor.validate_processed_data(df_processed)
        if not validation['valid']:
            print(f"❌ Validation errors: {validation['errors']}")
        if validation['warnings']:
            print(f"⚠️  Validation warnings: {validation['warnings']}")
        
        print("✅ Turtle data cleaning pipeline completed!")
        return df_processed


# Convenience functions for direct import
def load_turtle_data(data_dir: Path) -> pd.DataFrame:
    """Load turtle trading data"""
    return FinancialDataProcessor.load_turtle_data(data_dir)

def handle_missing_prices(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """Handle missing price data"""
    return FinancialDataProcessor.handle_missing_prices(df, method)

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators"""
    return FinancialDataProcessor.create_technical_features(df)

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data quality"""
    return FinancialDataProcessor.analyze_data_quality(df) 