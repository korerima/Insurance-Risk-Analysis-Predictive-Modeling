"""
Base model class for insurance risk analysis models.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class BaseModel(ABC, BaseEstimator):
    """Abstract base class for all insurance risk models."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.model_params = kwargs
        self.model: Optional[Any] = None
        self.is_trained = False
        self.feature_names: Optional[list] = None
        self.training_metrics: Dict[str, float] = {}
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create and return the underlying model instance."""
        pass
    
    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to the training data."""
        pass
    
    @abstractmethod
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_name} model")
        
        if self.model is None:
            self.model = self._create_model()
        
        self.feature_names = list(X.columns)
        self._fit_model(X, y)
        self.is_trained = True
        
        logger.info(f"{self.model_name} model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions array
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.feature_names is None:
            raise ValueError("Feature names not set during training")
        
        # Ensure X has the same columns as training data
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")
        
        extra_cols = set(X.columns) - set(self.feature_names)
        if extra_cols:
            logger.warning(f"Extra features will be ignored: {extra_cols}")
            X = X[self.feature_names]
        
        return self._predict_model(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions if the model supports it.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probability predictions array
            
        Raises:
            NotImplementedError: If model doesn't support probability predictions
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(f"{self.model_name} does not support probability predictions")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if the model supports it.
        
        Returns:
            Dictionary mapping feature names to importance scores
            
        Raises:
            NotImplementedError: If model doesn't support feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise NotImplementedError(f"{self.model_name} does not support feature importance")
        
        if self.feature_names is None:
            raise ValueError("Feature names not set during training")
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, file_path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.model_params = model_data['model_params']
            self.feature_names = model_data['feature_names']
            self.training_metrics = model_data['training_metrics']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model."""
        return {
            'model_name': self.model_name,
            'model_params': self.model_params,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'model_type': type(self.model).__name__ if self.model else None
        }
