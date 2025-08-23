"""
Exploratory Data Analysis (EDA) Utilities Module
Homework 8 - Stage 08: Exploratory Data Analysis
Author: Panwei Hu
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, normaltest, jarque_bera
from typing import List, Dict, Union, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class UnivariateAnalyzer:
    """Comprehensive univariate analysis tools"""
    
    @staticmethod
    def analyze_numeric_column(series: pd.Series, column_name: str = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a numeric column.
        """
        if column_name is None:
            column_name = series.name or 'Unknown'
        
        # Remove missing values for calculations
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {'error': 'No valid data points'}
        
        analysis = {
            'column_name': column_name,
            'total_count': len(series),
            'valid_count': len(clean_series),
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
        }
        
        # Descriptive statistics
        analysis.update({
            'mean': clean_series.mean(),
            'median': clean_series.median(),
            'mode': clean_series.mode().iloc[0] if not clean_series.mode().empty else np.nan,
            'std': clean_series.std(),
            'variance': clean_series.var(),
            'min': clean_series.min(),
            'max': clean_series.max(),
            'range': clean_series.max() - clean_series.min(),
            'q25': clean_series.quantile(0.25),
            'q75': clean_series.quantile(0.75),
            'iqr': clean_series.quantile(0.75) - clean_series.quantile(0.25),
        })
        
        # Shape statistics
        analysis.update({
            'skewness': skew(clean_series),
            'kurtosis': kurtosis(clean_series),
            'excess_kurtosis': kurtosis(clean_series, fisher=True),
        })
        
        # Normality tests
        if len(clean_series) >= 8:  # Minimum for normality tests
            try:
                shapiro_stat, shapiro_p = stats.shapiro(clean_series[:5000])  # Limit for Shapiro-Wilk
                analysis['shapiro_stat'] = shapiro_stat
                analysis['shapiro_p_value'] = shapiro_p
                analysis['is_normal_shapiro'] = shapiro_p > 0.05
            except:
                analysis['shapiro_stat'] = np.nan
                analysis['shapiro_p_value'] = np.nan
                analysis['is_normal_shapiro'] = False
            
            try:
                jb_stat, jb_p = jarque_bera(clean_series)
                analysis['jarque_bera_stat'] = jb_stat
                analysis['jarque_bera_p_value'] = jb_p
                analysis['is_normal_jb'] = jb_p > 0.05
            except:
                analysis['jarque_bera_stat'] = np.nan
                analysis['jarque_bera_p_value'] = np.nan
                analysis['is_normal_jb'] = False
        
        # Outlier detection (IQR method)
        q1, q3 = analysis['q25'], analysis['q75']
        iqr = analysis['iqr']
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        outliers = clean_series[(clean_series < lower_fence) | (clean_series > upper_fence)]
        analysis.update({
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(clean_series)) * 100,
            'lower_fence': lower_fence,
            'upper_fence': upper_fence,
        })
        
        # Distribution classification
        analysis['distribution_type'] = UnivariateAnalyzer._classify_distribution(analysis)
        
        return analysis
    
    @staticmethod
    def _classify_distribution(analysis: Dict) -> str:
        """Classify the distribution type based on statistical properties"""
        skewness = analysis.get('skewness', 0)
        kurtosis_val = analysis.get('excess_kurtosis', 0)
        
        # Classification rules
        if abs(skewness) < 0.5 and abs(kurtosis_val) < 0.5:
            return 'Approximately Normal'
        elif skewness > 1:
            return 'Highly Right-Skewed'
        elif skewness > 0.5:
            return 'Moderately Right-Skewed'
        elif skewness < -1:
            return 'Highly Left-Skewed'
        elif skewness < -0.5:
            return 'Moderately Left-Skewed'
        elif kurtosis_val > 1:
            return 'Heavy-Tailed (Leptokurtic)'
        elif kurtosis_val < -1:
            return 'Light-Tailed (Platykurtic)'
        else:
            return 'Approximately Symmetric'
    
    @staticmethod
    def analyze_categorical_column(series: pd.Series, column_name: str = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a categorical column.
        """
        if column_name is None:
            column_name = series.name or 'Unknown'
        
        analysis = {
            'column_name': column_name,
            'total_count': len(series),
            'valid_count': series.count(),
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
        }
        
        # Category analysis
        value_counts = series.value_counts()
        analysis.update({
            'unique_count': series.nunique(),
            'most_frequent': value_counts.index[0] if not value_counts.empty else None,
            'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
            'most_frequent_percentage': (value_counts.iloc[0] / series.count() * 100) if not value_counts.empty else 0,
            'least_frequent': value_counts.index[-1] if not value_counts.empty else None,
            'least_frequent_count': value_counts.iloc[-1] if not value_counts.empty else 0,
        })
        
        # Diversity measures
        if len(value_counts) > 0:
            # Shannon entropy (diversity index)
            proportions = value_counts / series.count()
            shannon_entropy = -np.sum(proportions * np.log2(proportions))
            analysis['shannon_entropy'] = shannon_entropy
            analysis['normalized_entropy'] = shannon_entropy / np.log2(len(value_counts)) if len(value_counts) > 1 else 0
        
        # Distribution evenness
        if len(value_counts) > 1:
            # Coefficient of variation for category frequencies
            cv = value_counts.std() / value_counts.mean()
            analysis['frequency_cv'] = cv
            analysis['distribution_evenness'] = 'Even' if cv < 0.5 else 'Uneven'
        
        analysis['value_counts'] = value_counts.to_dict()
        
        return analysis


class BivariateAnalyzer:
    """Bivariate relationship analysis tools"""
    
    @staticmethod
    def analyze_numeric_relationship(x: pd.Series, y: pd.Series, 
                                   x_name: str = None, y_name: str = None) -> Dict[str, Any]:
        """
        Analyze relationship between two numeric variables.
        """
        if x_name is None:
            x_name = x.name or 'X'
        if y_name is None:
            y_name = y.name or 'Y'
        
        # Clean data (remove rows where either variable is missing)
        clean_data = pd.DataFrame({x_name: x, y_name: y}).dropna()
        
        if len(clean_data) < 2:
            return {'error': 'Insufficient valid data points for analysis'}
        
        x_clean = clean_data[x_name]
        y_clean = clean_data[y_name]
        
        analysis = {
            'x_variable': x_name,
            'y_variable': y_name,
            'valid_pairs': len(clean_data),
            'missing_pairs': len(x) - len(clean_data),
        }
        
        # Correlation analysis
        pearson_corr, pearson_p = stats.pearsonr(x_clean, y_clean)
        spearman_corr, spearman_p = stats.spearmanr(x_clean, y_clean)
        
        analysis.update({
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'pearson_significant': pearson_p < 0.05,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'spearman_significant': spearman_p < 0.05,
        })
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        analysis.update({
            'linear_slope': slope,
            'linear_intercept': intercept,
            'r_squared': r_value**2,
            'regression_p_value': p_value,
            'standard_error': std_err,
        })
        
        # Relationship strength classification
        abs_corr = abs(pearson_corr)
        if abs_corr >= 0.8:
            strength = 'Very Strong'
        elif abs_corr >= 0.6:
            strength = 'Strong'
        elif abs_corr >= 0.4:
            strength = 'Moderate'
        elif abs_corr >= 0.2:
            strength = 'Weak'
        else:
            strength = 'Very Weak'
        
        direction = 'Positive' if pearson_corr > 0 else 'Negative'
        analysis['relationship_strength'] = strength
        analysis['relationship_direction'] = direction
        analysis['relationship_description'] = f"{strength} {direction}"
        
        return analysis
    
    @staticmethod
    def analyze_categorical_numeric_relationship(categorical: pd.Series, numeric: pd.Series,
                                               cat_name: str = None, num_name: str = None) -> Dict[str, Any]:
        """
        Analyze relationship between categorical and numeric variables.
        """
        if cat_name is None:
            cat_name = categorical.name or 'Categorical'
        if num_name is None:
            num_name = numeric.name or 'Numeric'
        
        # Clean data
        clean_data = pd.DataFrame({cat_name: categorical, num_name: numeric}).dropna()
        
        if len(clean_data) < 2:
            return {'error': 'Insufficient valid data points for analysis'}
        
        analysis = {
            'categorical_variable': cat_name,
            'numeric_variable': num_name,
            'valid_pairs': len(clean_data),
            'categories': clean_data[cat_name].nunique(),
        }
        
        # Group statistics
        grouped = clean_data.groupby(cat_name)[num_name]
        group_stats = {
            'group_means': grouped.mean().to_dict(),
            'group_medians': grouped.median().to_dict(),
            'group_stds': grouped.std().to_dict(),
            'group_counts': grouped.count().to_dict(),
        }
        analysis.update(group_stats)
        
        # ANOVA test (if more than 2 groups)
        if clean_data[cat_name].nunique() >= 2:
            groups = [group for name, group in grouped]
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                analysis.update({
                    'anova_f_statistic': f_stat,
                    'anova_p_value': p_value,
                    'groups_significantly_different': p_value < 0.05,
                })
            except:
                analysis.update({
                    'anova_f_statistic': np.nan,
                    'anova_p_value': np.nan,
                    'groups_significantly_different': False,
                })
        
        # Effect size (eta-squared)
        if 'anova_f_statistic' in analysis and not np.isnan(analysis['anova_f_statistic']):
            ss_between = sum(len(group) * (group.mean() - clean_data[num_name].mean())**2 
                           for name, group in grouped)
            ss_total = ((clean_data[num_name] - clean_data[num_name].mean())**2).sum()
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            analysis['eta_squared'] = eta_squared
            
            # Effect size interpretation
            if eta_squared >= 0.14:
                effect_size = 'Large'
            elif eta_squared >= 0.06:
                effect_size = 'Medium'
            elif eta_squared >= 0.01:
                effect_size = 'Small'
            else:
                effect_size = 'Negligible'
            analysis['effect_size'] = effect_size
        
        return analysis


class EDAVisualizer:
    """Comprehensive EDA visualization tools"""
    
    @staticmethod
    def plot_numeric_distribution(series: pd.Series, title: str = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Create comprehensive distribution plot for numeric variable.
        """
        if title is None:
            title = f'Distribution Analysis: {series.name}'
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Remove missing values
        clean_data = series.dropna()
        
        # 1. Histogram with KDE
        ax1 = axes[0, 0]
        ax1.hist(clean_data, bins=30, alpha=0.7, density=True, edgecolor='black', color='skyblue')
        
        # Overlay KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(clean_data)
            x_range = np.linspace(clean_data.min(), clean_data.max(), 200)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            ax1.legend()
        except:
            pass
        
        ax1.set_title('Histogram with KDE')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2 = axes[0, 1]
        box_plot = ax2.boxplot(clean_data, vert=True, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
        ax2.set_title('Box Plot (Outliers)')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        ax3 = axes[1, 0]
        stats.probplot(clean_data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (vs Normal)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        analysis = UnivariateAnalyzer.analyze_numeric_column(series)
        
        stats_text = f"""
        Summary Statistics:
        Count: {analysis.get('valid_count', 'N/A'):,}
        Mean: {analysis.get('mean', 0):.3f}
        Median: {analysis.get('median', 0):.3f}
        Std: {analysis.get('std', 0):.3f}
        Skewness: {analysis.get('skewness', 0):.3f}
        Kurtosis: {analysis.get('excess_kurtosis', 0):.3f}
        
        Missing: {analysis.get('missing_count', 0)} ({analysis.get('missing_percentage', 0):.1f}%)
        Outliers: {analysis.get('outlier_count', 0)} ({analysis.get('outlier_percentage', 0):.1f}%)
        
        Distribution: {analysis.get('distribution_type', 'Unknown')}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_categorical_distribution(series: pd.Series, title: str = None, figsize: Tuple[int, int] = (12, 6)):
        """
        Create comprehensive distribution plot for categorical variable.
        """
        if title is None:
            title = f'Categorical Analysis: {series.name}'
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Get value counts
        value_counts = series.value_counts()
        
        # 1. Bar plot
        ax1 = axes[0]
        bars = ax1.bar(range(len(value_counts)), value_counts.values, 
                      color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Frequency Distribution')
        ax1.set_xlabel('Categories')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(value_counts)))
        ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, value_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(value_counts),
                    f'{value}', ha='center', va='bottom')
        
        # 2. Pie chart
        ax2 = axes[1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        wedges, texts, autotexts = ax2.pie(value_counts.values, labels=value_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Proportion Distribution')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, title: str = 'Correlation Matrix', 
                               figsize: Tuple[int, int] = (10, 8)):
        """
        Create correlation heatmap for numeric variables.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric columns found for correlation analysis.")
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    @staticmethod
    def plot_pairwise_relationships(df: pd.DataFrame, hue: str = None, figsize: Tuple[int, int] = (12, 10)):
        """
        Create pairwise relationship plots for numeric variables.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            print("Need at least 2 numeric columns for pairwise analysis.")
            return
        
        # Create pair plot
        if hue and hue in df.columns:
            g = sns.pairplot(df[list(numeric_df.columns) + [hue]], hue=hue, 
                           diag_kind='hist', plot_kws={'alpha': 0.7})
        else:
            g = sns.pairplot(numeric_df, diag_kind='hist', plot_kws={'alpha': 0.7})
        
        g.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=16, fontweight='bold')
        plt.show()


class EDAReporter:
    """Generate comprehensive EDA reports"""
    
    @staticmethod
    def generate_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive dataset summary.
        """
        summary = {
            'dataset_shape': df.shape,
            'total_cells': df.shape[0] * df.shape[1],
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        # Column type analysis
        dtypes = df.dtypes.value_counts()
        summary['column_types'] = dtypes.to_dict()
        
        # Missing data analysis
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        summary.update({
            'total_missing_values': total_missing,
            'missing_percentage': (total_missing / (df.shape[0] * df.shape[1])) * 100,
            'columns_with_missing': (missing_counts > 0).sum(),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
        })
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        summary.update({
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': (duplicate_rows / len(df)) * 100,
        })
        
        # Numeric vs categorical split
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        summary.update({
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
            'numeric_column_names': numeric_cols,
            'categorical_column_names': categorical_cols,
            'datetime_column_names': datetime_cols,
        })
        
        return summary
    
    @staticmethod
    def generate_insights_and_recommendations(df: pd.DataFrame, analyses: Dict) -> List[str]:
        """
        Generate data-driven insights and recommendations.
        """
        insights = []
        
        # Dataset size insights
        if df.shape[0] < 100:
            insights.append("⚠️ Small dataset size may limit statistical power of analyses")
        elif df.shape[0] > 100000:
            insights.append("✅ Large dataset provides good statistical power")
        
        # Missing data insights
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 20:
            insights.append(f"🚨 High missing data rate ({missing_pct:.1f}%) requires attention")
        elif missing_pct > 5:
            insights.append(f"⚠️ Moderate missing data ({missing_pct:.1f}%) - consider imputation strategies")
        
        # Correlation insights
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)
            if high_corr.any().any():
                insights.append("🔍 High correlations detected - consider multicollinearity issues")
        
        # Distribution insights
        for col in numeric_df.columns:
            analysis = analyses.get(col, {})
            skewness = analysis.get('skewness', 0)
            if abs(skewness) > 2:
                insights.append(f"📊 {col} is highly skewed - consider transformation")
            
            outlier_pct = analysis.get('outlier_percentage', 0)
            if outlier_pct > 10:
                insights.append(f"🎯 {col} has high outlier rate ({outlier_pct:.1f}%) - investigate")
        
        # Categorical insights
        categorical_df = df.select_dtypes(include=['object', 'category'])
        for col in categorical_df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                insights.append(f"🔑 {col} has very high cardinality - may be identifier")
            elif unique_ratio < 0.01:
                insights.append(f"📋 {col} has very low diversity - consider usefulness")
        
        return insights


# Convenience functions for direct import
def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive dataset analysis"""
    return EDAReporter.generate_dataset_summary(df)

def plot_distribution(series: pd.Series, **kwargs):
    """Plot distribution based on data type"""
    if pd.api.types.is_numeric_dtype(series):
        EDAVisualizer.plot_numeric_distribution(series, **kwargs)
    else:
        EDAVisualizer.plot_categorical_distribution(series, **kwargs) 