"""
Base Model for Heat Hotspot Prediction

This module defines the abstract base class that all heat hotspot prediction models
should inherit from. It provides a common interface for training, prediction,
and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class BaseHeatspotModel(ABC):
    """
    Abstract base class for heat hotspot prediction models.
    
    This class defines the common interface that all models should implement
    for consistency and interoperability.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base model.
        
        Args:
            model_name: Name identifier for the model
            **kwargs: Model-specific parameters
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.target_column = None
        self.model_params = kwargs
        
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for model training/prediction.
        
        Args:
            data: Raw input data
            
        Returns:
            Processed feature array
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics and information
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted values
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for heat hotspot classification.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted probabilities
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Test feature matrix
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        predictions = self.predict(X)
        
        # Placeholder for evaluation metrics
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0
        }
        
        # TODO: Implement actual metric calculations
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # TODO: Implement model serialization
        pass
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        # TODO: Implement model deserialization
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # To be implemented by models that support feature importance
        return None