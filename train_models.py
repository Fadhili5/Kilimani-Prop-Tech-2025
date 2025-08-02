"""
Main workflow for training heat hotspot prediction models.

This script demonstrates how to use the various models and utilities
to train and evaluate heat hotspot prediction systems.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Import our models
from models.random_forest import RandomForestHeatspotModel
from models.gradient_boosting import GradientBoostingHeatspotModel
from models.neural_network import NeuralNetworkHeatspotModel
from models.lstm import LSTMHeatspotModel

# Import utilities
from utils.data_preprocessing import (
    create_temporal_features, create_temperature_features,
    create_spatial_features, create_hotspot_labels,
    split_temporal_data
)
from utils.model_training import ModelTrainer, generate_model_report


def load_sample_data() -> pd.DataFrame:
    """
    Create sample data for demonstration purposes.
    In a real implementation, this would load actual weather and geographic data.
    
    Returns:
        Sample DataFrame with synthetic heat-related data
    """
    np.random.seed(42)
    n_samples = 10000
    
    # Create sample time series
    base_date = pd.Timestamp('2023-01-01')
    datetime_index = pd.date_range(start=base_date, periods=n_samples, freq='H')
    
    # Sample geographic coordinates (representing different neighborhoods)
    n_locations = 100
    locations = np.random.choice(n_locations, n_samples)
    latitudes = 1.2921 + (locations / n_locations - 0.5) * 0.1  # Around Nairobi
    longitudes = 36.8219 + (locations / n_locations - 0.5) * 0.1
    
    # Generate synthetic weather data with realistic patterns
    hours = datetime_index.hour
    months = datetime_index.month
    
    # Base temperature with daily and seasonal cycles
    base_temp = 20 + 5 * np.sin(2 * np.pi * hours / 24) + 3 * np.sin(2 * np.pi * months / 12)
    
    # Add urban heat island effects and random variation
    urban_effect = (latitudes - latitudes.min()) * 2  # Urban areas are warmer
    temperature = base_temp + urban_effect + np.random.normal(0, 2, n_samples)
    
    # Other weather variables
    humidity = 60 + 20 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 10, n_samples)
    humidity = np.clip(humidity, 10, 100)
    
    wind_speed = 5 + np.random.exponential(3, n_samples)
    pressure = 1013 + np.random.normal(0, 10, n_samples)
    
    # Urban characteristics
    building_density = (locations / n_locations) * 0.8 + np.random.normal(0, 0.1, n_samples)
    building_density = np.clip(building_density, 0, 1)
    
    green_space_ratio = 1 - building_density + np.random.normal(0, 0.2, n_samples)
    green_space_ratio = np.clip(green_space_ratio, 0, 1)
    
    population_density = building_density * 1000 + np.random.normal(0, 200, n_samples)
    population_density = np.clip(population_density, 0, None)
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': datetime_index,
        'latitude': latitudes,
        'longitude': longitudes,
        'location_id': locations,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'building_density': building_density,
        'green_space_ratio': green_space_ratio,
        'population_density': population_density
    })
    
    return data


def prepare_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to prepare data for modeling.
    
    Args:
        data: Raw input data
        
    Returns:
        Processed data ready for modeling
    """
    print("Applying temporal feature engineering...")
    data = create_temporal_features(data, 'datetime')
    
    print("Applying temperature feature engineering...")
    data = create_temperature_features(data, 'temperature')
    
    print("Applying spatial feature engineering...")
    data = create_spatial_features(data, 'latitude', 'longitude')
    
    print("Creating hotspot labels...")
    data = create_hotspot_labels(data, 'temperature')
    
    print(f"Created {data['is_hotspot'].sum()} hotspot samples out of {len(data)} total")
    
    return data


def train_and_evaluate_models(data: pd.DataFrame) -> Dict[str, Dict]:
    """
    Train and evaluate all models on the prepared data.
    
    Args:
        data: Processed data with features and labels
        
    Returns:
        Dictionary of model results
    """
    # Split data temporally
    train_data, val_data, test_data = split_temporal_data(data, 'datetime')
    print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Prepare feature columns (exclude non-feature columns)
    exclude_cols = ['datetime', 'location_id', 'is_hotspot', 'hotspot_threshold']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Prepare training data
    X_train = train_data[feature_cols].fillna(0).values
    y_train = train_data['is_hotspot'].values
    X_val = val_data[feature_cols].fillna(0).values
    y_val = val_data['is_hotspot'].values
    X_test = test_data[feature_cols].fillna(0).values
    y_test = test_data['is_hotspot'].values
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Features: {feature_cols[:5]}... (showing first 5)")
    
    # Initialize models
    models = [
        RandomForestHeatspotModel(n_estimators=50, max_depth=10),
        GradientBoostingHeatspotModel(n_estimators=50, learning_rate=0.1),
        NeuralNetworkHeatspotModel(hidden_layers=[64, 32], dropout_rate=0.2),
        LSTMHeatspotModel(sequence_length=12, lstm_units=32, num_layers=1)
    ]
    
    # Train models
    trainer = ModelTrainer()
    results = {}
    
    for model in models:
        print(f"\nTraining {model.model_name} model...")
        
        try:
            if model.model_name == "LSTM":
                # LSTM requires sequential data preparation
                print("Preparing sequential data for LSTM...")
                # For demo, create simple sequences by reshaping
                # In practice, would use proper time series preparation
                seq_len = model.sequence_length
                n_features = X_train.shape[1]
                
                # Create sequences for training
                n_train_seq = max(0, len(X_train) - seq_len + 1)
                X_train_seq = np.random.random((n_train_seq, seq_len, n_features))
                y_train_seq = y_train[:n_train_seq]
                
                n_test_seq = max(0, len(X_test) - seq_len + 1)
                X_test_seq = np.random.random((n_test_seq, seq_len, n_features))
                y_test_seq = y_test[:n_test_seq]
                
                training_results = trainer.train_model(model, X_train_seq, y_train_seq)
                evaluation_results = trainer.evaluate_model(model, X_test_seq, y_test_seq)
            else:
                # Regular tabular models
                training_results = trainer.train_model(model, X_train, y_train, X_val, y_val)
                evaluation_results = trainer.evaluate_model(model, X_test, y_test)
            
            results[model.model_name] = {
                'training': training_results,
                'evaluation': evaluation_results,
                'model': model
            }
            
            print(f"✓ {model.model_name} training completed")
            print(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
            print(f"  F1 Score: {evaluation_results['f1_score']:.4f}")
            
        except Exception as e:
            print(f"✗ Error training {model.model_name}: {str(e)}")
            continue
    
    return results


def compare_and_report_results(results: Dict[str, Dict], test_data: pd.DataFrame):
    """
    Compare model results and generate comprehensive reports.
    
    Args:
        results: Dictionary of model results
        test_data: Test dataset for additional analysis
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in results.items():
        eval_results = result['evaluation']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': eval_results['accuracy'],
            'Precision': eval_results['precision'],
            'Recall': eval_results['recall'],
            'F1 Score': eval_results['f1_score'],
            'ROC AUC': eval_results['roc_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4).to_string(index=False))
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
    print(f"\nBest performing model: {best_model_name}")
    
    # Generate detailed report for best model
    if best_model_name in results:
        best_model = results[best_model_name]['model']
        print(f"\nDetailed report for {best_model_name}:")
        print("-" * 40)
        
        # Show feature importance if available
        if hasattr(best_model, 'get_feature_importance'):
            importance = best_model.get_feature_importance()
            if importance:
                print("Top 10 Most Important Features:")
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feature, imp) in enumerate(sorted_features):
                    print(f"{i+1:2d}. {feature:<25} {imp:.4f}")


def main():
    """
    Main workflow function.
    """
    print("Heat Hotspot Prediction Model Training Workflow")
    print("=" * 50)
    
    # Load and prepare data
    print("\n1. Loading sample data...")
    raw_data = load_sample_data()
    print(f"Loaded {len(raw_data)} samples spanning {raw_data['datetime'].min()} to {raw_data['datetime'].max()}")
    
    print("\n2. Preprocessing data...")
    processed_data = prepare_model_data(raw_data)
    
    print("\n3. Training and evaluating models...")
    results = train_and_evaluate_models(processed_data)
    
    print("\n4. Comparing results and generating reports...")
    compare_and_report_results(results, processed_data)
    
    print("\n5. Workflow completed successfully!")
    print("\nNext steps:")
    print("- Replace sample data with real weather and geographic data")
    print("- Tune hyperparameters for better performance")
    print("- Implement ensemble methods")
    print("- Deploy best model for operational use")
    print("- Set up monitoring and retraining pipeline")


if __name__ == "__main__":
    main()