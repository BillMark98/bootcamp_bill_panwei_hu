"""
Turtle Trading Feature Engineering Module
Stage 09: Feature Engineering for Turtle Trading Strategy
Author: Panwei Hu
Date: 2025-01-27

This module provides specialized feature engineering capabilities for financial time series
data, specifically designed for the Turtle Trading strategy implementation.
Works with preprocessed turtle data that already contains technical indicators.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import List, Dict, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class TurtleFeatureEngineer:
    """
    Specialized feature engineering for Turtle Trading strategy.
    Designed to work with preprocessed turtle data that already contains
    basic technical indicators and trading signals.
    """
    
    def __init__(self):
        self.fitted_scalers = {}
        self.fitted_pca = {}
        self.feature_metadata = {}
        
    def create_enhanced_price_features(self, df: pd.DataFrame, price_col: str = 'adj_close') -> pd.DataFrame:
        """
        Create additional price-based features that complement existing ones
        """
        df = df.copy()
        
        # Sort by symbol and date
        if 'symbol' in df.columns:
            df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Price momentum features (different from existing returns)
        for period in [5, 15, 30]:
            if 'symbol' in df.columns:
                df[f'price_momentum_{period}d'] = df.groupby('symbol')[price_col].apply(
                    lambda x: x / x.shift(period) - 1
                ).reset_index(0, drop=True)
            else:
                df[f'price_momentum_{period}d'] = df[price_col] / df[price_col].shift(period) - 1
        
        # Price acceleration (second derivative)
        if 'returns' in df.columns:
            if 'symbol' in df.columns:
                df['returns_acceleration'] = df.groupby('symbol')['returns'].diff()
            else:
                df['returns_acceleration'] = df['returns'].diff()
        
        # Price relative to various levels
        if 'sma_20' in df.columns:
            df['price_above_sma20'] = (df[price_col] > df['sma_20']).astype(int)
        if 'sma_50' in df.columns:
            df['price_above_sma50'] = (df[price_col] > df['sma_50']).astype(int)
        
        # Donchian position (if Donchian channels exist)
        if 'donchian_high_20' in df.columns and 'donchian_low_20' in df.columns:
            df['donchian_position_20'] = ((df[price_col] - df['donchian_low_20']) / 
                                         (df['donchian_high_20'] - df['donchian_low_20'] + 1e-8))
        
        if 'donchian_high_55' in df.columns and 'donchian_low_55' in df.columns:
            df['donchian_position_55'] = ((df[price_col] - df['donchian_low_55']) / 
                                         (df['donchian_high_55'] - df['donchian_low_55'] + 1e-8))
        
        return df
    
    def create_enhanced_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced trading signal features building on existing signals
        """
        df = df.copy()
        
        # Signal combinations
        if 'long_entry_20' in df.columns and 'short_entry_20' in df.columns:
            df['any_entry_signal'] = (df['long_entry_20'] | df['short_entry_20']).astype(int)
            df['conflicting_signals'] = (df['long_entry_20'] & df['short_entry_20']).astype(int)
        
        # Signal strength based on ATR and volatility
        if 'atr_20' in df.columns and 'volatility_20' in df.columns:
            df['signal_strength_atr'] = df['atr_20'] / df['volatility_20']
        
        # Signal persistence and streaks
        signal_cols = [col for col in df.columns if 'entry' in col or 'exit' in col]
        for signal_col in signal_cols:
            if 'symbol' in df.columns:
                # Signal streaks
                df[f'{signal_col}_streak'] = df.groupby('symbol')[signal_col].apply(
                    lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
                ).reset_index(0, drop=True)
                
                # Days since last signal
                df[f'{signal_col}_days_since'] = df.groupby('symbol')[signal_col].apply(
                    lambda x: (x == 0).astype(int).groupby(x.ne(0).cumsum()).cumsum()
                ).reset_index(0, drop=True)
            else:
                df[f'{signal_col}_streak'] = df[signal_col] * (
                    df[signal_col].groupby((df[signal_col] != df[signal_col].shift()).cumsum()).cumcount() + 1
                )
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced volatility and risk features
        """
        df = df.copy()
        
        # Volatility ratios and comparisons
        if 'volatility_20' in df.columns and 'atr_20' in df.columns:
            df['vol_atr_ratio'] = df['volatility_20'] / (df['atr_20'] / df['adj_close'] + 1e-8)
        
        # Volatility regime changes
        if 'volatility_20' in df.columns:
            if 'symbol' in df.columns:
                df['vol_regime_change'] = df.groupby('symbol')['volatility_20'].apply(
                    lambda x: (x > x.shift(1) * 1.5).astype(int)  # 50% increase
                ).reset_index(0, drop=True)
            else:
                df['vol_regime_change'] = (df['volatility_20'] > df['volatility_20'].shift(1) * 1.5).astype(int)
        
        # Volatility percentiles
        if 'volatility_20' in df.columns:
            if 'symbol' in df.columns:
                df['vol_percentile_252d'] = df.groupby('symbol')['volatility_20'].rolling(252).rank(pct=True).reset_index(0, drop=True)
            else:
                df['vol_percentile_252d'] = df['volatility_20'].rolling(252).rank(pct=True)
        
        return df
    
    def create_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-asset and market-relative features
        """
        df = df.copy()
        
        if 'symbol' not in df.columns:
            return df
        
        # Market-wide statistics
        if 'returns' in df.columns:
            df['market_return'] = df.groupby('date')['returns'].transform('mean')
            df['market_vol'] = df.groupby('date')['returns'].transform('std')
            df['excess_return'] = df['returns'] - df['market_return']
            df['return_vs_market'] = df['returns'] / (df['market_return'] + 1e-8)
        
        # Asset correlation with market
        if 'returns' in df.columns:
            # for window in [20, 60]:
            #     df[f'market_corr_{window}d'] = df.groupby('symbol')['returns'].rolling(window).corr(
            #         df.groupby('date')['returns'].transform('mean')
            #     ).reset_index(0, drop=True)
            for window in [20, 60]:
                mean_returns = df.groupby('date')['returns'].transform('mean')

                def rolling_corr(x):
                    return x.rolling(window).corr(mean_returns.loc[x.index])

                df[f'market_corr_{window}d'] = df.groupby('symbol')['returns'].transform(rolling_corr)
        
        # Relative performance rankings
        if 'returns' in df.columns:
            for window in [20, 60]:
                rolling_returns = df.groupby('symbol')['returns'].rolling(window).mean().reset_index(0, drop=True)
                df[f'return_rank_{window}d'] = rolling_returns.groupby(df['date']).rank(pct=True)
        
        # Asset category features
        if 'asset_category' in df.columns:
            # Category average performance
            df['category_avg_return'] = df.groupby(['date', 'asset_category'])['returns'].transform('mean')
            df['return_vs_category'] = df['returns'] - df['category_avg_return']
            
            # Category volatility
            df['category_avg_vol'] = df.groupby(['date', 'asset_category'])['volatility_20'].transform('mean')
            df['vol_vs_category'] = df['volatility_20'] - df['category_avg_vol']
        
        return df
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market regime and state features
        """
        df = df.copy()
        
        # Trend regime based on moving averages
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['ma_trend_regime'] = np.where(
                df['sma_20'] > df['sma_50'], 1,  # Uptrend
                np.where(df['sma_20'] < df['sma_50'], -1, 0)  # Downtrend or sideways
            )
        
        # Volatility regime
        if 'volatility_20' in df.columns:
            vol_quantiles = df['volatility_20'].quantile([0.33, 0.67])
            # # df['vol_regime'] = pd.cut(
            #     df['volatility_20'],
            #     bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.67], np.inf],
            #     labels=[0, 1, 2]  # Low, Medium, High
            # ).astype(int)
            df['vol_regime'] = pd.cut(
            df['volatility_20'],
            bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.67], np.inf],
            labels=[0, 1, 2]
        ).astype('Int64')  # Note the capital 'I'

        
        # Breakout regime (based on Donchian position)
        if 'donchian_position_20' in df.columns:
            df['breakout_regime'] = pd.cut(
                df['donchian_position_20'],
                bins=[0, 0.2, 0.8, 1.0],
                labels=[0, 1, 2],  # Low, Middle, High
                include_lowest=True
            ).astype('Int64')
        
        # Combined regime score
        regime_cols = [col for col in df.columns if 'regime' in col and col.endswith(('_regime', 'regime'))]
        if len(regime_cols) >= 2:
            df['combined_regime_score'] = df[regime_cols].sum(axis=1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables
        """
        df = df.copy()
        
        # ATR × Volatility interactions
        if 'atr_20' in df.columns and 'volatility_20' in df.columns:
            df['atr_vol_product'] = df['atr_20'] * df['volatility_20']
            df['atr_vol_ratio'] = df['atr_20'] / (df['volatility_20'] * df['adj_close'] + 1e-8)
        
        # Signal × Volatility interactions
        if 'long_entry_20' in df.columns and 'volatility_20' in df.columns:
            df['entry_signal_vol_interaction'] = df['long_entry_20'] * df['volatility_20']
        
        # Trend × Momentum interactions
        if 'trend_strength' in df.columns and 'price_momentum_5d' in df.columns:
            df['trend_momentum_interaction'] = df['trend_strength'] * df['price_momentum_5d']
        
        # Price position × ATR (position sizing relevance)
        if 'price_position' in df.columns and 'atr_20' in df.columns:
            df['position_atr_interaction'] = df['price_position'] * df['atr_20']
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        """
        df = df.copy()
        
        if 'date' not in df.columns:
            return df
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = ((df['date'].dt.month % 3 == 0) & df['date'].dt.is_month_end).astype(int)
        df['is_year_end'] = ((df['date'].dt.month == 12) & df['date'].dt.is_month_end).astype(int)
        
        # Trading day features
        df['is_monday'] = (df['date'].dt.dayofweek == 0).astype(int)
        df['is_friday'] = (df['date'].dt.dayofweek == 4).astype(int)
        
        return df
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk management and portfolio features
        """
        df = df.copy()
        
        # Sharpe ratio approximation
        if 'returns' in df.columns:
            for window in [20, 60, 252]:
                if 'symbol' in df.columns:
                    rolling_return = df.groupby('symbol')['returns'].rolling(window).mean().reset_index(0, drop=True) * 252
                    rolling_vol = df.groupby('symbol')['returns'].rolling(window).std().reset_index(0, drop=True) * np.sqrt(252)
                else:
                    rolling_return = df['returns'].rolling(window).mean() * 252
                    rolling_vol = df['returns'].rolling(window).std() * np.sqrt(252)
                
                df[f'sharpe_approx_{window}d'] = rolling_return / (rolling_vol + 1e-8)
        
        # Drawdown features
        if 'symbol' in df.columns:
            df['running_max'] = df.groupby('symbol')['adj_close'].expanding().max().reset_index(0, drop=True)
        else:
            df['running_max'] = df['adj_close'].expanding().max()
        
        df['drawdown'] = (df['adj_close'] - df['running_max']) / (df['running_max'] + 1e-8)
        
        # Rolling max drawdown
        for window in [60, 252]:
            if 'symbol' in df.columns:
                df[f'max_dd_{window}d'] = df.groupby('symbol')['drawdown'].rolling(window).min().reset_index(0, drop=True)
            else:
                df[f'max_dd_{window}d'] = df['drawdown'].rolling(window).min()
        
        return df
    
    def create_pca_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create PCA features from technical indicators
        """
        df = df.copy()
        
        # Select technical features for PCA
        technical_features = [col for col in df.columns if any(indicator in col for indicator in 
                            ['sma_', 'donchian_', 'atr_', 'volatility_', 'momentum_', 'trend_'])]
        
        if len(technical_features) >= 3:
            # Prepare data for PCA
            pca_data = df[technical_features].select_dtypes(include=[np.number])
            pca_data = pca_data.fillna(pca_data.mean())
            pca_data = pca_data.replace([np.inf, -np.inf], np.nan).fillna(pca_data.mean())
            
            if pca_data.shape[1] >= 2 and pca_data.shape[0] > pca_data.shape[1]:
                try:
                    n_components = min(3, pca_data.shape[1])
                    pca = PCA(n_components=n_components)
                    pca_features = pca.fit_transform(pca_data)
                    
                    for i in range(n_components):
                        df[f'technical_pca_{i+1}'] = pca_features[:, i]
                    
                    # Store explained variance
                    self.feature_metadata['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
                    self.fitted_pca['technical_pca'] = pca
                    
                except Exception as e:
                    print(f"⚠️ PCA failed: {e}")
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, asset_col: str = 'symbol',
                             price_col: str = 'adj_close') -> pd.DataFrame:
        """
        Apply comprehensive feature engineering pipeline for turtle trading
        """
        print("🔧 Starting Enhanced Turtle Trading Feature Engineering...")
        print(f"   Input dataset: {df.shape}")
        print(f"   Existing columns: {list(df.columns)}")
        
        df_engineered = df.copy()
        original_features = len(df_engineered.columns)
        
        # Apply feature engineering steps
        print("1️⃣ Creating enhanced price features...")
        df_engineered = self.create_enhanced_price_features(df_engineered, price_col)
        
        print("2️⃣ Creating enhanced signal features...")
        df_engineered = self.create_enhanced_signal_features(df_engineered)
        
        print("3️⃣ Creating volatility features...")
        df_engineered = self.create_volatility_features(df_engineered)
        
        print("4️⃣ Creating cross-asset features...")
        df_engineered = self.create_cross_asset_features(df_engineered)
        
        print("5️⃣ Creating regime features...")
        df_engineered = self.create_regime_features(df_engineered)
        
        print("6️⃣ Creating interaction features...")
        df_engineered = self.create_interaction_features(df_engineered)
        
        print("7️⃣ Creating time features...")
        df_engineered = self.create_time_features(df_engineered)
        
        print("8️⃣ Creating risk features...")
        df_engineered = self.create_risk_features(df_engineered)
        
        print("9️⃣ Creating PCA features...")
        df_engineered = self.create_pca_features(df_engineered)
        
        # Store metadata
        final_features = len(df_engineered.columns)
        new_features = final_features - original_features
        
        self.feature_metadata.update({
            'original_features': original_features,
            'final_features': final_features,
            'new_features': new_features,
            'expansion_ratio': new_features / original_features if original_features > 0 else 0,
            'feature_categories': {
                'enhanced_price': len([col for col in df_engineered.columns if 'momentum' in col or 'acceleration' in col or 'above' in col]),
                'enhanced_signals': len([col for col in df_engineered.columns if 'streak' in col or 'days_since' in col or 'strength' in col]),
                'volatility': len([col for col in df_engineered.columns if 'vol_' in col or 'regime_change' in col]),
                'cross_asset': len([col for col in df_engineered.columns if 'market_' in col or 'category_' in col or 'rank_' in col]),
                'regime': len([col for col in df_engineered.columns if 'regime' in col and col != 'vol_regime_change']),
                'interaction': len([col for col in df_engineered.columns if 'interaction' in col]),
                'time': len([col for col in df_engineered.columns if any(x in col for x in ['day_', 'month', 'quarter', 'is_'])]),
                'risk': len([col for col in df_engineered.columns if any(x in col for x in ['sharpe_', 'drawdown', 'max_dd'])]),
                'pca': len([col for col in df_engineered.columns if 'pca_' in col])
            }
        })
        
        print(f"✅ Enhanced feature engineering complete!")
        print(f"   Features: {original_features} → {final_features} (+{new_features})")
        print(f"   Expansion ratio: {new_features/original_features:.1f}x" if original_features > 0 else "   Expansion ratio: N/A")
        
        return df_engineered
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Calculate feature importance based on correlation with target
        """
        if target_col not in df.columns:
            print(f"⚠️ Target column '{target_col}' not found")
            return pd.DataFrame()
        
        correlations = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != target_col and df[col].notna().sum() > 100:
                corr = df[col].corr(df[target_col])
                if not np.isnan(corr):
                    correlations.append({
                        'feature': col,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
        
        importance_df = pd.DataFrame(correlations)
        importance_df = importance_df.sort_values('abs_correlation', ascending=False)
        
        return importance_df


# Convenience functions
def create_turtle_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to create enhanced turtle trading features
    """
    engineer = TurtleFeatureEngineer()
    return engineer.engineer_all_features(df, **kwargs)


if __name__ == "__main__":
    print("🐢 Enhanced Turtle Trading Feature Engineering Module")
    print("=" * 60)
    print("Designed for preprocessed turtle data with existing technical indicators")
    print("Adds complementary features:")
    print("- Enhanced price momentum and acceleration")
    print("- Signal persistence and strength measures") 
    print("- Advanced volatility and regime features")
    print("- Cross-asset and market-relative features")
    print("- Risk management and portfolio features")
    print("- PCA dimensionality reduction")
    print("- Time-based and interaction features") 