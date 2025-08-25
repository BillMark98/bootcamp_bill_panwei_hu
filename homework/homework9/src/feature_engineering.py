"""
Feature Engineering Module
Homework 9 - Stage 09: Feature Engineering
Author: Panwei Hu
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from typing import List, Dict, Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering toolkit based on the 10 categories:
    1. Combining columns
    2. Splitting columns  
    3. Point-wise transformations
    4. Window-based transformations
    5. Categorical binning
    6. Group aggregations
    7. Feature interactions
    8. Temporal/sequential features
    9. External knowledge encoding
    10. Dimensionality reduction
    """
    
    def __init__(self):
        self.fitted_encoders = {}
        self.fitted_scalers = {}
        self.fitted_pca = {}
        
    def combine_features(self, df: pd.DataFrame, combinations: List[Dict]) -> pd.DataFrame:
        """
        Category 1: Combine columns to create new features
        
        Parameters:
        - df: Input DataFrame
        - combinations: List of dicts with 'columns', 'operation', 'name'
        
        Example:
        combinations = [
            {'columns': ['income', 'monthly_spend'], 'operation': 'ratio', 'name': 'spend_income_ratio'},
            {'columns': ['age', 'credit_score'], 'operation': 'sum', 'name': 'age_credit_sum'}
        ]
        """
        df = df.copy()
        
        for combo in combinations:
            cols = combo['columns']
            operation = combo['operation']
            name = combo['name']
            
            if operation == 'sum':
                df[name] = df[cols].sum(axis=1)
            elif operation == 'difference':
                df[name] = df[cols[0]] - df[cols[1]]
            elif operation == 'ratio':
                df[name] = df[cols[0]] / df[cols[1]]
            elif operation == 'product':
                df[name] = df[cols].prod(axis=1)
            elif operation == 'mean':
                df[name] = df[cols].mean(axis=1)
            elif operation == 'max':
                df[name] = df[cols].max(axis=1)
            elif operation == 'min':
                df[name] = df[cols].min(axis=1)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        return df
    
    def split_features(self, df: pd.DataFrame, split_configs: List[Dict]) -> pd.DataFrame:
        """
        Category 2: Split columns into multiple features
        
        Parameters:
        - split_configs: List of dicts with column and split specifications
        
        Example:
        split_configs = [
            {'column': 'date', 'type': 'datetime', 'components': ['year', 'month', 'day']},
            {'column': 'income', 'type': 'numeric', 'splits': [10, 100, 1000]}  # Digits
        ]
        """
        df = df.copy()
        
        for config in split_configs:
            col = config['column']
            split_type = config['type']
            
            if split_type == 'datetime':
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    dt_col = df[col]
                else:
                    dt_col = pd.to_datetime(df[col])
                
                components = config.get('components', ['year', 'month', 'day'])
                
                for component in components:
                    if component == 'year':
                        df[f'{col}_year'] = dt_col.dt.year
                    elif component == 'month':
                        df[f'{col}_month'] = dt_col.dt.month
                    elif component == 'day':
                        df[f'{col}_day'] = dt_col.dt.day
                    elif component == 'dayofweek':
                        df[f'{col}_dayofweek'] = dt_col.dt.dayofweek
                    elif component == 'quarter':
                        df[f'{col}_quarter'] = dt_col.dt.quarter
                    elif component == 'is_weekend':
                        df[f'{col}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
            
            elif split_type == 'numeric':
                # Split numeric values by place value
                splits = config.get('splits', [10, 100])
                for i, divisor in enumerate(splits):
                    df[f'{col}_div_{divisor}'] = df[col] // divisor
                    df[f'{col}_mod_{divisor}'] = df[col] % divisor
        
        return df
    
    def pointwise_transforms(self, df: pd.DataFrame, transforms: List[Dict]) -> pd.DataFrame:
        """
        Category 3: Apply point-wise transformations
        
        Parameters:
        - transforms: List of transformation specifications
        
        Example:
        transforms = [
            {'column': 'income', 'transform': 'log', 'name': 'log_income'},
            {'column': 'age', 'transform': 'square', 'name': 'age_squared'}
        ]
        """
        df = df.copy()
        
        for transform in transforms:
            col = transform['column']
            transform_type = transform['transform']
            name = transform['name']
            
            if transform_type == 'log':
                df[name] = np.log1p(df[col])  # log(1+x) to handle zeros
            elif transform_type == 'sqrt':
                df[name] = np.sqrt(np.abs(df[col]))
            elif transform_type == 'square':
                df[name] = df[col] ** 2
            elif transform_type == 'cube':
                df[name] = df[col] ** 3
            elif transform_type == 'reciprocal':
                df[name] = 1 / (df[col] + 1e-8)  # Avoid division by zero
            elif transform_type == 'abs':
                df[name] = np.abs(df[col])
            elif transform_type == 'sign':
                df[name] = np.sign(df[col])
            elif transform_type == 'normalize':
                df[name] = (df[col] - df[col].mean()) / df[col].std()
            elif transform_type == 'minmax':
                df[name] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                raise ValueError(f"Unknown transform: {transform_type}")
        
        return df
    
    def window_transforms(self, df: pd.DataFrame, window_configs: List[Dict], 
                         sort_column: str = None) -> pd.DataFrame:
        """
        Category 4: Window-based transformations
        
        Parameters:
        - window_configs: List of window transformation specifications
        - sort_column: Column to sort by before applying window functions
        
        Example:
        window_configs = [
            {'column': 'monthly_spend', 'window': 3, 'operation': 'mean', 'name': 'spend_3m_avg'},
            {'column': 'income', 'window': 5, 'operation': 'std', 'name': 'income_5_std'}
        ]
        """
        df = df.copy()
        
        if sort_column:
            df = df.sort_values(sort_column)
        
        for config in window_configs:
            col = config['column']
            window = config['window']
            operation = config['operation']
            name = config['name']
            
            if operation == 'mean':
                df[name] = df[col].rolling(window=window, min_periods=1).mean()
            elif operation == 'sum':
                df[name] = df[col].rolling(window=window, min_periods=1).sum()
            elif operation == 'std':
                df[name] = df[col].rolling(window=window, min_periods=1).std()
            elif operation == 'min':
                df[name] = df[col].rolling(window=window, min_periods=1).min()
            elif operation == 'max':
                df[name] = df[col].rolling(window=window, min_periods=1).max()
            elif operation == 'median':
                df[name] = df[col].rolling(window=window, min_periods=1).median()
            elif operation == 'diff':
                df[name] = df[col].diff(periods=window).fillna(0)
            elif operation == 'pct_change':
                df[name] = df[col].pct_change(periods=window).fillna(0)
            elif operation == 'cumsum':
                df[name] = df[col].cumsum()
            elif operation == 'cumprod':
                df[name] = df[col].cumprod()
            else:
                raise ValueError(f"Unknown window operation: {operation}")
        
        return df
    
    def categorical_binning(self, df: pd.DataFrame, binning_configs: List[Dict]) -> pd.DataFrame:
        """
        Category 5: Split continuous features into categories
        
        Parameters:
        - binning_configs: List of binning specifications
        
        Example:
        binning_configs = [
            {'column': 'age', 'bins': [18, 30, 50, 65, 100], 'labels': ['young', 'adult', 'middle', 'senior']},
            {'column': 'income', 'method': 'quantile', 'n_bins': 5}
        ]
        """
        df = df.copy()
        
        for config in binning_configs:
            col = config['column']
            
            if 'bins' in config:
                # Custom bins
                bins = config['bins']
                labels = config.get('labels', None)
                df[f'{col}_binned'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
                
            elif config.get('method') == 'quantile':
                # Quantile-based binning
                n_bins = config.get('n_bins', 5)
                df[f'{col}_quantile'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                
            elif config.get('method') == 'equal_width':
                # Equal-width binning
                n_bins = config.get('n_bins', 5)
                df[f'{col}_equal_width'] = pd.cut(df[col], bins=n_bins, labels=False)
        
        return df
    
    def group_aggregations(self, df: pd.DataFrame, group_configs: List[Dict]) -> pd.DataFrame:
        """
        Category 6: Aggregate features across groups
        
        Parameters:
        - group_configs: List of grouping and aggregation specifications
        
        Example:
        group_configs = [
            {'groupby': 'region', 'column': 'income', 'operation': 'mean', 'name': 'region_avg_income'},
            {'groupby': ['age_group', 'region'], 'column': 'spend', 'operation': 'median'}
        ]
        """
        df = df.copy()
        
        for config in group_configs:
            groupby_cols = config['groupby']
            if isinstance(groupby_cols, str):
                groupby_cols = [groupby_cols]
            
            col = config['column']
            operation = config['operation']
            name = config.get('name', f"{col}_{operation}_by_{'_'.join(groupby_cols)}")
            
            if operation == 'mean':
                df[name] = df.groupby(groupby_cols)[col].transform('mean')
            elif operation == 'median':
                df[name] = df.groupby(groupby_cols)[col].transform('median')
            elif operation == 'std':
                df[name] = df.groupby(groupby_cols)[col].transform('std')
            elif operation == 'count':
                df[name] = df.groupby(groupby_cols)[col].transform('count')
            elif operation == 'min':
                df[name] = df.groupby(groupby_cols)[col].transform('min')
            elif operation == 'max':
                df[name] = df.groupby(groupby_cols)[col].transform('max')
            elif operation == 'rank':
                df[name] = df.groupby(groupby_cols)[col].rank()
            else:
                raise ValueError(f"Unknown aggregation operation: {operation}")
        
        return df
    
    def feature_interactions(self, df: pd.DataFrame, interaction_configs: List[Dict]) -> pd.DataFrame:
        """
        Category 7: Create interaction features
        
        Parameters:
        - interaction_configs: List of interaction specifications
        
        Example:
        interaction_configs = [
            {'columns': ['age', 'income'], 'type': 'multiply', 'name': 'age_income_interaction'},
            {'columns': ['credit_score', 'monthly_spend'], 'type': 'polynomial', 'degree': 2}
        ]
        """
        df = df.copy()
        
        for config in interaction_configs:
            cols = config['columns']
            interaction_type = config['type']
            
            if interaction_type == 'multiply':
                name = config.get('name', f"{'_'.join(cols)}_interaction")
                df[name] = df[cols].prod(axis=1)
                
            elif interaction_type == 'polynomial':
                degree = config.get('degree', 2)
                include_bias = config.get('include_bias', False)
                
                poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
                poly_features = poly.fit_transform(df[cols])
                feature_names = poly.get_feature_names_out(cols)
                
                # Add polynomial features (excluding original features)
                for i, name in enumerate(feature_names):
                    if name not in cols:  # Skip original features
                        df[f"poly_{name}"] = poly_features[:, i]
        
        return df
    
    def temporal_features(self, df: pd.DataFrame, temporal_configs: List[Dict]) -> pd.DataFrame:
        """
        Category 8: Create temporal/sequential features
        
        Parameters:
        - temporal_configs: List of temporal feature specifications
        
        Example:
        temporal_configs = [
            {'column': 'monthly_spend', 'type': 'lag', 'periods': [1, 3, 6]},
            {'column': 'income', 'type': 'diff', 'periods': 1}
        ]
        """
        df = df.copy()
        
        for config in temporal_configs:
            col = config['column']
            temporal_type = config['type']
            
            if temporal_type == 'lag':
                periods = config.get('periods', [1])
                if isinstance(periods, int):
                    periods = [periods]
                
                for period in periods:
                    df[f'{col}_lag_{period}'] = df[col].shift(period)
            
            elif temporal_type == 'lead':
                periods = config.get('periods', [1])
                if isinstance(periods, int):
                    periods = [periods]
                
                for period in periods:
                    df[f'{col}_lead_{period}'] = df[col].shift(-period)
            
            elif temporal_type == 'diff':
                periods = config.get('periods', 1)
                df[f'{col}_diff_{periods}'] = df[col].diff(periods)
            
            elif temporal_type == 'pct_change':
                periods = config.get('periods', 1)
                df[f'{col}_pct_change_{periods}'] = df[col].pct_change(periods)
        
        return df
    
    def encode_external_knowledge(self, df: pd.DataFrame, encoding_configs: List[Dict]) -> pd.DataFrame:
        """
        Category 9: Encode external knowledge
        
        Parameters:
        - encoding_configs: List of encoding specifications
        
        Example:
        encoding_configs = [
            {'column': 'region', 'type': 'onehot'},
            {'column': 'education', 'type': 'label'},
            {'column': 'region', 'type': 'frequency'}
        ]
        """
        df = df.copy()
        
        for config in encoding_configs:
            col = config['column']
            encoding_type = config['type']
            
            if encoding_type == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
            
            elif encoding_type == 'label':
                # Label encoding
                if col not in self.fitted_encoders:
                    self.fitted_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.fitted_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.fitted_encoders[col].transform(df[col].astype(str))
            
            elif encoding_type == 'frequency':
                # Frequency encoding
                freq_map = df[col].value_counts(normalize=True)
                df[f'{col}_frequency'] = df[col].map(freq_map)
            
            elif encoding_type == 'target':
                # Target encoding (requires target column)
                target_col = config.get('target_column')
                if target_col:
                    target_mean = df.groupby(col)[target_col].mean()
                    df[f'{col}_target_encoded'] = df[col].map(target_mean)
        
        return df
    
    def dimensionality_reduction(self, df: pd.DataFrame, reduction_configs: List[Dict]) -> pd.DataFrame:
        """
        Category 10: Dimensionality reduction
        
        Parameters:
        - reduction_configs: List of reduction specifications
        
        Example:
        reduction_configs = [
            {'columns': ['income', 'age', 'credit_score'], 'method': 'pca', 'n_components': 2},
            {'columns': ['numeric_features'], 'method': 'standardize'}
        ]
        """
        df = df.copy()
        
        for config in reduction_configs:
            cols = config['columns']
            method = config['method']
            
            if method == 'pca':
                n_components = config.get('n_components', 2)
                
                if 'pca' not in self.fitted_pca:
                    self.fitted_pca['pca'] = PCA(n_components=n_components)
                    pca_features = self.fitted_pca['pca'].fit_transform(df[cols])
                else:
                    pca_features = self.fitted_pca['pca'].transform(df[cols])
                
                for i in range(n_components):
                    df[f'pca_component_{i+1}'] = pca_features[:, i]
            
            elif method == 'standardize':
                scaler_name = f"standard_{'_'.join(cols)}"
                if scaler_name not in self.fitted_scalers:
                    self.fitted_scalers[scaler_name] = StandardScaler()
                    scaled_features = self.fitted_scalers[scaler_name].fit_transform(df[cols])
                else:
                    scaled_features = self.fitted_scalers[scaler_name].transform(df[cols])
                
                for i, col in enumerate(cols):
                    df[f'{col}_standardized'] = scaled_features[:, i]
            
            elif method == 'minmax':
                scaler_name = f"minmax_{'_'.join(cols)}"
                if scaler_name not in self.fitted_scalers:
                    self.fitted_scalers[scaler_name] = MinMaxScaler()
                    scaled_features = self.fitted_scalers[scaler_name].fit_transform(df[cols])
                else:
                    scaled_features = self.fitted_scalers[scaler_name].transform(df[cols])
                
                for i, col in enumerate(cols):
                    df[f'{col}_minmax'] = scaled_features[:, i]
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
        """
        Apply all feature engineering steps in sequence
        
        Parameters:
        - df: Input DataFrame
        - feature_config: Dictionary with all feature engineering configurations
        
        Returns:
        - DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # Apply transformations in order
        if 'combine' in feature_config:
            df_engineered = self.combine_features(df_engineered, feature_config['combine'])
        
        if 'split' in feature_config:
            df_engineered = self.split_features(df_engineered, feature_config['split'])
        
        if 'pointwise' in feature_config:
            df_engineered = self.pointwise_transforms(df_engineered, feature_config['pointwise'])
        
        if 'window' in feature_config:
            sort_col = feature_config.get('sort_column', None)
            df_engineered = self.window_transforms(df_engineered, feature_config['window'], sort_col)
        
        if 'binning' in feature_config:
            df_engineered = self.categorical_binning(df_engineered, feature_config['binning'])
        
        if 'grouping' in feature_config:
            df_engineered = self.group_aggregations(df_engineered, feature_config['grouping'])
        
        if 'interactions' in feature_config:
            df_engineered = self.feature_interactions(df_engineered, feature_config['interactions'])
        
        if 'temporal' in feature_config:
            df_engineered = self.temporal_features(df_engineered, feature_config['temporal'])
        
        if 'encoding' in feature_config:
            df_engineered = self.encode_external_knowledge(df_engineered, feature_config['encoding'])
        
        if 'reduction' in feature_config:
            df_engineered = self.dimensionality_reduction(df_engineered, feature_config['reduction'])
        
        return df_engineered


def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create a comprehensive set of features
    for financial/behavioral data
    """
    engineer = FeatureEngineer()
    
    # Define comprehensive feature engineering configuration
    feature_config = {
        'combine': [
            {'columns': ['income', 'monthly_spending'], 'operation': 'ratio', 'name': 'spend_income_ratio'},
            {'columns': ['age', 'credit_score'], 'operation': 'product', 'name': 'age_credit_interaction'},
            {'columns': ['income', 'account_balance'], 'operation': 'sum', 'name': 'total_wealth'}
        ],
        'pointwise': [
            {'column': 'income', 'transform': 'log', 'name': 'log_income'},
            {'column': 'monthly_spending', 'transform': 'sqrt', 'name': 'sqrt_spending'},
            {'column': 'age', 'transform': 'square', 'name': 'age_squared'}
        ],
        'window': [
            {'column': 'monthly_spending', 'window': 3, 'operation': 'mean', 'name': 'spending_3m_avg'},
            {'column': 'income', 'window': 6, 'operation': 'std', 'name': 'income_6m_volatility'},
            {'column': 'account_balance', 'window': 12, 'operation': 'max', 'name': 'balance_12m_peak'}
        ],
        'binning': [
            {'column': 'age', 'bins': [18, 30, 45, 60, 100], 'labels': ['young', 'adult', 'middle', 'senior']},
            {'column': 'income', 'method': 'quantile', 'n_bins': 5}
        ],
        'interactions': [
            {'columns': ['age', 'income'], 'type': 'multiply', 'name': 'age_income_interaction'},
            {'columns': ['credit_score', 'account_balance'], 'type': 'polynomial', 'degree': 2}
        ]
    }
    
    return engineer.engineer_all_features(df, feature_config)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n = 1000
    
    sample_df = pd.DataFrame({
        'customer_id': range(1, n+1),
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'age': np.random.normal(40, 12, n).clip(18, 80).round(0),
        'income': np.random.lognormal(mean=10.8, sigma=0.6, size=n).round(2),
        'credit_score': np.random.normal(720, 80, size=n).clip(300, 850).round(0),
        'account_balance': np.random.exponential(scale=2500, size=n).round(2),
        'monthly_spending': np.random.normal(2000, 600, size=n).clip(0).round(2),
        'region': np.random.choice(['North', 'South', 'East', 'West'], size=n),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n),
        'default_flag': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    })
    
    print("Original DataFrame shape:", sample_df.shape)
    print("\nOriginal columns:", sample_df.columns.tolist())
    
    # Apply comprehensive feature engineering
    engineered_df = create_comprehensive_features(sample_df)
    
    print("\nEngineered DataFrame shape:", engineered_df.shape)
    print(f"Added {engineered_df.shape[1] - sample_df.shape[1]} new features")
    print("\nNew features created:")
    new_features = [col for col in engineered_df.columns if col not in sample_df.columns]
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}") 