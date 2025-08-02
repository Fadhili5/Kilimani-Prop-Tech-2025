"""
Model training and evaluation utilities.

This module provides functions for training models, evaluating performance,
and managing the machine learning workflow for heat hotspot prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """
    Class for managing the training and evaluation of heat hotspot prediction models.
    """
    
    def __init__(self):
        self.trained_models = {}
        self.evaluation_results = {}
        
    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Train a single model with optional validation data.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        # Prepare validation set for models that support it
        if X_val is not None and y_val is not None:
            if hasattr(model, 'train') and 'eval_set' in model.train.__code__.co_varnames:
                kwargs['eval_set'] = (X_val, y_val)
        
        # Train the model
        training_results = model.train(X_train, y_train, **kwargs)
        
        # Store trained model
        self.trained_models[model.model_name] = model
        
        return training_results
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not model.is_trained:
            raise ValueError(f"Model {model.model_name} is not trained")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of hotspot class
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # Store evaluation results
        self.evaluation_results[model.model_name] = metrics
        
        return metrics
    
    def compare_models(self, models: List, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: List of trained model instances
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for model in models:
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics['model_name'] = model.model_name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('model_name')
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create visualization comparing model performance.
        
        Args:
            comparison_df: DataFrame from compare_models()
            save_path: Optional path to save the plot
        """
        # TODO: Implement actual plotting with matplotlib/seaborn
        print("Model Performance Comparison:")
        print(comparison_df.round(4))
        
        if save_path:
            print(f"Plot would be saved to: {save_path}")


def calculate_spatial_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            coordinates: np.ndarray) -> Dict[str, float]:
    """
    Calculate spatial-aware evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        coordinates: Array of (lat, lon) coordinates
        
    Returns:
        Dictionary of spatial metrics
    """
    # TODO: Implement spatial evaluation metrics
    # - Spatial autocorrelation of errors
    # - Hotspot clustering accuracy
    # - Geographic distribution of false positives/negatives
    
    metrics = {
        'spatial_accuracy': 0.0,
        'hotspot_cluster_recall': 0.0,
        'spatial_precision': 0.0,
        'geographic_coverage': 0.0
    }
    
    return metrics


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                               save_path: Optional[str] = None):
    """
    Create and display confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # TODO: Implement actual plotting
    print("Confusion Matrix:")
    print(cm)
    
    if save_path:
        print(f"Confusion matrix plot would be saved to: {save_path}")


def plot_feature_importance(model, top_n: int = 20, save_path: Optional[str] = None):
    """
    Plot feature importance for models that support it.
    
    Args:
        model: Trained model with feature importance
        top_n: Number of top features to plot
        save_path: Optional path to save the plot
    """
    importance_dict = model.get_feature_importance()
    
    if importance_dict is None:
        print(f"Model {model.model_name} does not support feature importance")
        return
    
    # Sort and get top features
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"Top {top_n} Most Important Features for {model.model_name}:")
    for i, (feature, importance) in enumerate(sorted_features):
        print(f"{i+1:2d}. {feature:<30} {importance:.4f}")
    
    if save_path:
        print(f"Feature importance plot would be saved to: {save_path}")


def create_prediction_map(predictions: np.ndarray, coordinates: np.ndarray,
                         save_path: Optional[str] = None):
    """
    Create a geographic map of predictions.
    
    Args:
        predictions: Model predictions (0/1 or probabilities)
        coordinates: Array of (lat, lon) coordinates
        save_path: Optional path to save the map
    """
    # TODO: Implement actual mapping with folium or similar
    print(f"Prediction map with {len(predictions)} points would be created")
    print(f"Coordinate range: lat {coordinates[:, 0].min():.3f} to {coordinates[:, 0].max():.3f}")
    print(f"                 lon {coordinates[:, 1].min():.3f} to {coordinates[:, 1].max():.3f}")
    
    if save_path:
        print(f"Map would be saved to: {save_path}")


def cross_validate_temporal(model, X: np.ndarray, y: np.ndarray, 
                          time_series: pd.Series, n_splits: int = 5) -> Dict[str, List[float]]:
    """
    Perform time-series aware cross-validation.
    
    Args:
        model: Model instance to validate
        X: Feature matrix
        y: Target values
        time_series: Time series for temporal splitting
        n_splits: Number of CV splits
        
    Returns:
        Dictionary of cross-validation scores
    """
    # TODO: Implement proper time-series cross-validation
    # Use TimeSeriesSplit or custom temporal splitting
    
    cv_scores = {
        'accuracy': [0.8, 0.82, 0.79, 0.81, 0.83],  # Placeholder
        'f1_score': [0.75, 0.77, 0.73, 0.76, 0.78],
        'roc_auc': [0.85, 0.87, 0.83, 0.86, 0.88]
    }
    
    return cv_scores


def generate_model_report(model, X_test: np.ndarray, y_test: np.ndarray, 
                         coordinates: Optional[np.ndarray] = None) -> str:
    """
    Generate comprehensive evaluation report for a model.
    
    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test targets
        coordinates: Optional coordinates for spatial analysis
        
    Returns:
        Formatted report string
    """
    # Basic evaluation
    trainer = ModelTrainer()
    basic_metrics = trainer.evaluate_model(model, X_test, y_test)
    
    # Spatial metrics if coordinates provided
    y_pred = model.predict(X_test)
    spatial_metrics = {}
    if coordinates is not None:
        spatial_metrics = calculate_spatial_metrics(y_test, y_pred, coordinates)
    
    # Generate report
    report = f"""
Model Evaluation Report: {model.model_name}
{'='*50}

Basic Performance Metrics:
- Accuracy:  {basic_metrics['accuracy']:.4f}
- Precision: {basic_metrics['precision']:.4f}
- Recall:    {basic_metrics['recall']:.4f}
- F1 Score:  {basic_metrics['f1_score']:.4f}
- ROC AUC:   {basic_metrics['roc_auc']:.4f}

Test Set Statistics:
- Total samples: {len(y_test)}
- Positive cases (hotspots): {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)
- Negative cases (normal): {len(y_test) - np.sum(y_test)} ({(1-np.mean(y_test))*100:.1f}%)
"""
    
    if spatial_metrics:
        report += f"""
Spatial Performance Metrics:
- Spatial Accuracy: {spatial_metrics['spatial_accuracy']:.4f}
- Hotspot Cluster Recall: {spatial_metrics['hotspot_cluster_recall']:.4f}
- Spatial Precision: {spatial_metrics['spatial_precision']:.4f}
- Geographic Coverage: {spatial_metrics['geographic_coverage']:.4f}
"""
    
    # Feature importance if available
    if hasattr(model, 'get_feature_importance') and model.get_feature_importance():
        importance_dict = model.get_feature_importance()
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        
        report += f"""
Top 5 Most Important Features:
"""
        for i, (feature, importance) in enumerate(top_features):
            report += f"{i+1}. {feature}: {importance:.4f}\n"
    
    return report