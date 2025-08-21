"""
Risk prediction models for insurance analysis.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config.settings import Settings, get_settings
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RiskPredictor(BaseModel):
    """Risk prediction model for insurance claims."""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize the risk predictor.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'logistic', 'svm')
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        super().__init__(f"RiskPredictor_{model_type}", **kwargs)
    
    def _create_model(self) -> Any:
        """Create the underlying model instance."""
        settings = get_settings()
        
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', settings.n_estimators),
                max_depth=self.model_params.get('max_depth', settings.max_depth),
                random_state=settings.random_state,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.model_params.get('n_estimators', settings.n_estimators),
                max_depth=self.model_params.get('max_depth', settings.max_depth),
                learning_rate=self.model_params.get('learning_rate', settings.learning_rate),
                random_state=settings.random_state
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                random_state=settings.random_state,
                max_iter=1000,
                **self.model_params
            )
        elif self.model_type == "svm":
            return SVC(
                probability=True,
                random_state=settings.random_state,
                **self.model_params
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        self.training_metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def predict_risk_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the probability of risk (claim occurrence).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probability array for risk occurrence
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(f"{self.model_type} does not support probability predictions")
        
        proba = self.predict_proba(X)
        # Return probability of positive class (risk)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    
    def get_risk_categories(self, X: pd.DataFrame, thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Categorize risk levels based on probability thresholds.
        
        Args:
            X: Feature DataFrame
            thresholds: Dictionary with risk category thresholds
                       Default: {'Low': 0.3, 'Medium': 0.7, 'High': 1.0}
            
        Returns:
            Array of risk categories
        """
        if thresholds is None:
            thresholds = {'Low': 0.3, 'Medium': 0.7, 'High': 1.0}
        
        probabilities = self.predict_risk_probability(X)
        categories = np.full(len(probabilities), 'Very High', dtype=object)
        
        for category, threshold in thresholds.items():
            categories[probabilities <= threshold] = category
        
        return categories
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        summary = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'feature_count': len(self.feature_names) if self.feature_names else 0
        }
        
        if self.is_trained and hasattr(self.model, 'feature_importances_'):
            feature_importance = self.get_feature_importance()
            if feature_importance:
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                summary['top_features'] = top_features
        
        return summary


class EnsembleRiskPredictor(BaseModel):
    """Ensemble model combining multiple risk predictors."""
    
    def __init__(self, models: Optional[Dict[str, RiskPredictor]] = None, **kwargs):
        """
        Initialize the ensemble predictor.
        
        Args:
            models: Dictionary of RiskPredictor models
            **kwargs: Additional parameters
        """
        super().__init__("EnsembleRiskPredictor", **kwargs)
        self.models = models or {}
        self.weights = kwargs.get('weights', None)
    
    def add_model(self, name: str, model: RiskPredictor) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
    
    def _create_model(self) -> Any:
        """Ensemble doesn't have a single underlying model."""
        return None
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all models in the ensemble."""
        for name, model in self.models.items():
            logger.info(f"Training ensemble model: {name}")
            model.fit(X, y)
        
        # Calculate ensemble training metrics
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Simple voting (majority)
        ensemble_pred = np.mean(predictions, axis=0) > 0.5
        
        self.training_metrics = {
            'accuracy': accuracy_score(y, ensemble_pred),
            'precision': precision_score(y, ensemble_pred, average='weighted'),
            'recall': recall_score(y, ensemble_pred, average='weighted'),
            'f1': f1_score(y, ensemble_pred, average='weighted')
        }
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average if weights provided, otherwise simple average
        if self.weights and len(self.weights) == len(predictions):
            weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            weighted_pred = np.mean(predictions, axis=0)
        
        return (weighted_pred > 0.5).astype(int)
    
    def predict_risk_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions."""
        probabilities = []
        for model in self.models.values():
            try:
                prob = model.predict_risk_probability(X)
                probabilities.append(prob)
            except NotImplementedError:
                # If model doesn't support probabilities, use predictions
                prob = model.predict(X).astype(float)
                probabilities.append(prob)
        
        if self.weights and len(self.weights) == len(probabilities):
            return np.average(probabilities, axis=0, weights=self.weights)
        else:
            return np.mean(probabilities, axis=0)
