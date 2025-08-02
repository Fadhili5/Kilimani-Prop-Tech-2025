"""
Data preprocessing utilities for heat hotspot prediction.

This module contains functions for loading, cleaning, and preparing
geospatial and temporal data for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def load_weather_data(filepath: str) -> pd.DataFrame:
    """
    Load and parse weather station data.
    
    Args:
        filepath: Path to weather data file
        
    Returns:
        Cleaned DataFrame with weather observations
    """
    # TODO: Implement actual data loading
    # Placeholder structure for weather data
    return pd.DataFrame()


def load_geographic_data(filepath: str) -> pd.DataFrame:
    """
    Load geographic and urban characteristics data.
    
    Args:
        filepath: Path to geographic data file
        
    Returns:
        DataFrame with geographic features
    """
    # TODO: Implement actual geographic data loading
    return pd.DataFrame()


def create_spatial_features(data: pd.DataFrame, lat_col: str = 'latitude', 
                           lon_col: str = 'longitude') -> pd.DataFrame:
    """
    Engineer spatial features from coordinates.
    
    Args:
        data: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        DataFrame with additional spatial features
    """
    # TODO: Implement spatial feature engineering
    # - Distance to city center
    # - Elevation from DEM
    # - Proximity to water bodies
    # - Urban density metrics
    
    result = data.copy()
    
    # Placeholder spatial features
    result['distance_to_center'] = 0.0
    result['elevation'] = 0.0
    result['distance_to_water'] = 0.0
    
    return result


def create_temporal_features(data: pd.DataFrame, time_col: str = 'datetime') -> pd.DataFrame:
    """
    Engineer temporal features from datetime column.
    
    Args:
        data: DataFrame with datetime column
        time_col: Name of datetime column
        
    Returns:
        DataFrame with additional temporal features
    """
    if time_col not in data.columns:
        raise ValueError(f"Column {time_col} not found in data")
    
    result = data.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(result[time_col]):
        result[time_col] = pd.to_datetime(result[time_col])
    
    # Extract basic temporal features
    result['hour'] = result[time_col].dt.hour
    result['day_of_week'] = result[time_col].dt.dayofweek
    result['day_of_year'] = result[time_col].dt.dayofyear
    result['month'] = result[time_col].dt.month
    result['season'] = result[time_col].dt.month.map({
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,   # Spring  
        6: 2, 7: 2, 8: 2,   # Summer
        9: 3, 10: 3, 11: 3  # Fall
    })
    
    # Cyclical encoding for continuous features
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['day_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
    result['day_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    
    # Boolean features
    result['is_weekend'] = result['day_of_week'].isin([5, 6])
    result['is_summer'] = result['season'] == 2
    result['is_daytime'] = result['hour'].between(6, 18)
    
    return result


def create_temperature_features(data: pd.DataFrame, temp_col: str = 'temperature') -> pd.DataFrame:
    """
    Engineer temperature-based features.
    
    Args:
        data: DataFrame with temperature column
        temp_col: Name of temperature column
        
    Returns:
        DataFrame with temperature-derived features
    """
    if temp_col not in data.columns:
        raise ValueError(f"Column {temp_col} not found in data")
    
    result = data.copy()
    
    # Rolling statistics
    result['temp_ma_3h'] = result[temp_col].rolling(window=3).mean()
    result['temp_ma_6h'] = result[temp_col].rolling(window=6).mean()
    result['temp_ma_12h'] = result[temp_col].rolling(window=12).mean()
    result['temp_ma_24h'] = result[temp_col].rolling(window=24).mean()
    
    result['temp_std_24h'] = result[temp_col].rolling(window=24).std()
    result['temp_max_24h'] = result[temp_col].rolling(window=24).max()
    result['temp_min_24h'] = result[temp_col].rolling(window=24).min()
    result['temp_range_24h'] = result['temp_max_24h'] - result['temp_min_24h']
    
    # Temperature trends
    result['temp_trend_3h'] = result[temp_col].diff(3)
    result['temp_trend_6h'] = result[temp_col].diff(6)
    result['temp_slope_3h'] = result[temp_col].rolling(window=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 3 else np.nan
    )
    
    # Temperature anomalies (compared to seasonal average)
    seasonal_avg = result.groupby(['month', 'hour'])[temp_col].transform('mean')
    result['temp_anomaly'] = result[temp_col] - seasonal_avg
    
    return result


def calculate_heat_index(temperature: pd.Series, humidity: pd.Series) -> pd.Series:
    """
    Calculate heat index from temperature and humidity.
    
    Args:
        temperature: Temperature in Celsius
        humidity: Relative humidity in percentage
        
    Returns:
        Heat index values
    """
    # Convert Celsius to Fahrenheit for heat index calculation
    temp_f = temperature * 9/5 + 32
    
    # Simplified heat index formula (Rothfusz equation)
    hi = (0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094)))
    
    # Use more complex formula for higher temperatures
    mask = hi >= 80
    if mask.any():
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = (
            -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783,
            -0.05481717, 0.00122874, 0.00085282, -0.00000199
        )
        
        hi_complex = (c1 + c2*temp_f + c3*humidity + c4*temp_f*humidity +
                     c5*temp_f**2 + c6*humidity**2 + c7*temp_f**2*humidity +
                     c8*temp_f*humidity**2 + c9*temp_f**2*humidity**2)
        
        hi = hi.where(~mask, hi_complex)
    
    # Convert back to Celsius
    return (hi - 32) * 5/9


def define_hotspot_threshold(temperature_data: pd.Series, method: str = 'percentile',
                           threshold: float = 90.0) -> float:
    """
    Define temperature threshold for hotspot classification.
    
    Args:
        temperature_data: Historical temperature observations
        method: Method for threshold calculation ('percentile', 'std', 'absolute')
        threshold: Threshold value (percentile, standard deviations, or absolute temp)
        
    Returns:
        Temperature threshold for hotspot classification
    """
    if method == 'percentile':
        return np.percentile(temperature_data.dropna(), threshold)
    elif method == 'std':
        mean_temp = temperature_data.mean()
        std_temp = temperature_data.std()
        return mean_temp + threshold * std_temp
    elif method == 'absolute':
        return threshold
    else:
        raise ValueError(f"Unknown method: {method}")


def create_hotspot_labels(data: pd.DataFrame, temp_col: str = 'temperature',
                         threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Create binary hotspot labels based on temperature threshold.
    
    Args:
        data: DataFrame with temperature data
        temp_col: Name of temperature column
        threshold: Temperature threshold (calculated if None)
        
    Returns:
        DataFrame with hotspot labels
    """
    result = data.copy()
    
    if threshold is None:
        threshold = define_hotspot_threshold(data[temp_col])
    
    result['is_hotspot'] = (result[temp_col] > threshold).astype(int)
    result['hotspot_threshold'] = threshold
    
    return result


def split_temporal_data(data: pd.DataFrame, time_col: str = 'datetime',
                       train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train/validation/test sets.
    
    Args:
        data: DataFrame with time series data
        time_col: Name of datetime column
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Sort by time
    sorted_data = data.sort_values(time_col)
    n_samples = len(sorted_data)
    
    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_data = sorted_data.iloc[:train_end]
    val_data = sorted_data.iloc[train_end:val_end]
    test_data = sorted_data.iloc[val_end:]
    
    return train_data, val_data, test_data