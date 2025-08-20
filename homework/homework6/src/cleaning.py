"""
Data Cleaning and Preprocessing Module
Homework 6 - Stage 06: Data Preprocessing
Author: Panwei Hu
Date: 2025-08-20
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Comprehensive data cleaning and preprocessing utilities"""
    
    @staticmethod
    def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns in the DataFrame"""
        missing_info = {}
        
        # Basic missing data statistics
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_info['missing_counts'] = missing_counts.to_dict()
        missing_info['missing_percentages'] = missing_percentages.to_dict()
        missing_info['total_missing'] = missing_counts.sum()
        missing_info['columns_with_missing'] = missing_counts[missing_counts > 0].index.tolist()
        
        # Missing data patterns
        missing_patterns = df.isnull().value_counts()
        missing_info['missing_patterns'] = missing_patterns.head(10).to_dict()
        
        return missing_info
    
    @staticmethod
    def fill_missing_median(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values with median for specified numeric columns"""
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                median_value = df[col].median()
                df_cleaned[col].fillna(median_value, inplace=True)
                print(f"✅ Filled {col} missing values with median: {median_value:.2f}")
            else:
                print(f"⚠️  Column {col} not found or not numeric")
        
        return df_cleaned
    
    @staticmethod
    def fill_missing_mean(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values with mean for specified numeric columns"""
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                mean_value = df[col].mean()
                df_cleaned[col].fillna(mean_value, inplace=True)
                print(f"✅ Filled {col} missing values with mean: {mean_value:.2f}")
            else:
                print(f"⚠️  Column {col} not found or not numeric")
        
        return df_cleaned
    
    @staticmethod
    def fill_missing_mode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Fill missing values with mode for specified columns"""
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df.columns:
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df_cleaned[col].fillna(mode_value[0], inplace=True)
                    print(f"✅ Filled {col} missing values with mode: {mode_value[0]}")
                else:
                    print(f"⚠️  No mode found for column {col}")
            else:
                print(f"⚠️  Column {col} not found")
        
        return df_cleaned
    
    @staticmethod
    def fill_missing_forward_fill(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Forward fill missing values for specified columns"""
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df.columns:
                before_count = df_cleaned[col].isnull().sum()
                df_cleaned[col].fillna(method='ffill', inplace=True)
                after_count = df_cleaned[col].isnull().sum()
                filled_count = before_count - after_count
                print(f"✅ Forward filled {filled_count} missing values in {col}")
            else:
                print(f"⚠️  Column {col} not found")
        
        return df_cleaned
    
    @staticmethod
    def fill_missing_interpolate(df: pd.DataFrame, columns: List[str], method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing values for specified numeric columns"""
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                before_count = df_cleaned[col].isnull().sum()
                df_cleaned[col].interpolate(method=method, inplace=True)
                after_count = df_cleaned[col].isnull().sum()
                filled_count = before_count - after_count
                print(f"✅ Interpolated {filled_count} missing values in {col} using {method} method")
            else:
                print(f"⚠️  Column {col} not found or not numeric")
        
        return df_cleaned
    
    @staticmethod
    def drop_missing(df: pd.DataFrame, threshold: float = 0.5, axis: int = 0) -> pd.DataFrame:
        """Drop rows/columns with missing values above threshold"""
        df_cleaned = df.copy()
        
        if axis == 0:  # Drop rows
            missing_threshold = int(threshold * len(df.columns))
            before_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(thresh=missing_threshold)
            after_rows = len(df_cleaned)
            dropped_rows = before_rows - after_rows
            print(f"✅ Dropped {dropped_rows} rows with >{threshold*100}% missing values")
        else:  # Drop columns
            missing_threshold = int(threshold * len(df))
            before_cols = len(df_cleaned.columns)
            df_cleaned = df_cleaned.dropna(thresh=missing_threshold, axis=1)
            after_cols = len(df_cleaned.columns)
            dropped_cols = before_cols - after_cols
            print(f"✅ Dropped {dropped_cols} columns with >{threshold*100}% missing values")
        
        return df_cleaned
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, columns: List[str], method: str = 'minmax') -> pd.DataFrame:
        """Normalize specified numeric columns using different methods"""
        df_normalized = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                if method == 'minmax':
                    # Min-Max scaling (0-1)
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
                    print(f"✅ Min-Max normalized {col} (range: {min_val:.2f} - {max_val:.2f})")
                
                elif method == 'zscore':
                    # Z-score standardization
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    df_normalized[col] = (df[col] - mean_val) / std_val
                    print(f"✅ Z-score normalized {col} (mean: {mean_val:.2f}, std: {std_val:.2f})")
                
                elif method == 'robust':
                    # Robust scaling using median and IQR
                    median_val = df[col].median()
                    q75, q25 = np.percentile(df[col].dropna(), [75, 25])
                    iqr = q75 - q25
                    df_normalized[col] = (df[col] - median_val) / iqr
                    print(f"✅ Robust normalized {col} (median: {median_val:.2f}, IQR: {iqr:.2f})")
                
                else:
                    print(f"⚠️  Unknown normalization method: {method}")
            else:
                print(f"⚠️  Column {col} not found or not numeric")
        
        return df_normalized
    
    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, columns: List[str], multiplier: float = 1.5) -> Dict[str, pd.Series]:
        """Detect outliers using IQR method"""
        outliers = {}
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers[col] = outlier_mask
                
                outlier_count = outlier_mask.sum()
                print(f"📊 {col}: {outlier_count} outliers detected (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        
        return outliers
    
    @staticmethod
    def detect_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> Dict[str, pd.Series]:
        """Detect outliers using Z-score method"""
        outliers = {}
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
                outliers[col] = outlier_mask
                
                outlier_count = outlier_mask.sum()
                print(f"📊 {col}: {outlier_count} outliers detected (Z-score > {threshold})")
        
        return outliers
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, outlier_masks: Dict[str, pd.Series], method: str = 'any') -> pd.DataFrame:
        """Remove outliers based on detection masks"""
        df_cleaned = df.copy()
        
        if method == 'any':
            # Remove rows that have outliers in ANY column
            combined_mask = pd.Series(False, index=df.index)
            for mask in outlier_masks.values():
                combined_mask |= mask
        elif method == 'all':
            # Remove rows that have outliers in ALL columns
            combined_mask = pd.Series(True, index=df.index)
            for mask in outlier_masks.values():
                combined_mask &= mask
        else:
            raise ValueError("Method must be 'any' or 'all'")
        
        before_rows = len(df_cleaned)
        df_cleaned = df_cleaned[~combined_mask]
        after_rows = len(df_cleaned)
        removed_rows = before_rows - after_rows
        
        print(f"✅ Removed {removed_rows} rows containing outliers ({method} method)")
        
        return df_cleaned
    
    @staticmethod
    def cap_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """Cap outliers at specified bounds instead of removing them"""
        df_capped = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                elif method == 'percentile':
                    lower_bound = df[col].quantile(0.01)
                    upper_bound = df[col].quantile(0.99)
                else:
                    print(f"⚠️  Unknown capping method: {method}")
                    continue
                
                # Count values that will be capped
                lower_capped = (df_capped[col] < lower_bound).sum()
                upper_capped = (df_capped[col] > upper_bound).sum()
                
                # Apply capping
                df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
                
                print(f"✅ Capped {col}: {lower_capped} values at lower bound, {upper_capped} at upper bound")
        
        return df_capped
    
    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical variables using different methods"""
        df_encoded = df.copy()
        
        for col in columns:
            if col in df.columns:
                if method == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                    print(f"✅ One-hot encoded {col} into {len(dummies.columns)} columns")
                
                elif method == 'label':
                    # Label encoding
                    unique_values = df_encoded[col].unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    df_encoded[col] = df_encoded[col].map(label_map)
                    print(f"✅ Label encoded {col} ({len(unique_values)} unique values)")
                
                else:
                    print(f"⚠️  Unknown encoding method: {method}")
            else:
                print(f"⚠️  Column {col} not found")
        
        return df_encoded
    
    @staticmethod
    def create_preprocessing_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Create a comprehensive report comparing original and cleaned data"""
        report = {}
        
        # Basic statistics
        report['original_shape'] = original_df.shape
        report['cleaned_shape'] = cleaned_df.shape
        report['rows_removed'] = original_df.shape[0] - cleaned_df.shape[0]
        report['columns_removed'] = original_df.shape[1] - cleaned_df.shape[1]
        
        # Missing data comparison
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        report['original_missing_values'] = original_missing
        report['cleaned_missing_values'] = cleaned_missing
        report['missing_values_handled'] = original_missing - cleaned_missing
        
        # Data types
        report['original_dtypes'] = original_df.dtypes.value_counts().to_dict()
        report['cleaned_dtypes'] = cleaned_df.dtypes.value_counts().to_dict()
        
        return report


# Convenience functions for direct use
def fill_missing_median(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convenience function for median imputation"""
    return DataCleaner.fill_missing_median(df, columns)

def fill_missing_mean(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convenience function for mean imputation"""
    return DataCleaner.fill_missing_mean(df, columns)

def drop_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Convenience function for dropping missing values"""
    return DataCleaner.drop_missing(df, threshold)

def normalize_data(df: pd.DataFrame, columns: List[str], method: str = 'minmax') -> pd.DataFrame:
    """Convenience function for data normalization"""
    return DataCleaner.normalize_data(df, columns, method)

def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, pd.Series]:
    """Convenience function for outlier detection"""
    if method == 'iqr':
        return DataCleaner.detect_outliers_iqr(df, columns)
    elif method == 'zscore':
        return DataCleaner.detect_outliers_zscore(df, columns)
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'") 