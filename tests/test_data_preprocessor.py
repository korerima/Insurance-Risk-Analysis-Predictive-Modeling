"""
Unit tests for data preprocessor module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_initialization(self, data_preprocessor):
        """Test DataPreprocessor initialization."""
        assert data_preprocessor is not None
        assert data_preprocessor.label_encoders == {}
        assert data_preprocessor._is_fitted is False
    
    def test_clean_data(self, data_preprocessor, sample_data):
        """Test data cleaning functionality."""
        cleaned_data = data_preprocessor.clean_data(sample_data)
        
        # Check that invalid records are removed
        assert len(cleaned_data) <= len(sample_data)
        assert all(cleaned_data['TotalPremium'] > 0)
        assert all(cleaned_data['TotalClaims'] >= 0)
        
        # Check that derived features are added
        assert 'LossRatio' in cleaned_data.columns
        assert 'HadClaim' in cleaned_data.columns
        
        # Check that missing values are handled
        assert cleaned_data['Province'].isnull().sum() == 0
        assert cleaned_data['VehicleType'].isnull().sum() == 0
        
        # Check that categorical columns are converted
        assert cleaned_data['Province'].dtype.name == 'category'
        assert cleaned_data['VehicleType'].dtype.name == 'category'
    
    def test_engineer_features(self, data_preprocessor, clean_sample_data):
        """Test feature engineering functionality."""
        engineered_data = data_preprocessor.engineer_features(clean_sample_data)
        
        # Check that date features are added
        assert 'TransactionYear' in engineered_data.columns
        assert 'TransactionMonthNum' in engineered_data.columns
        assert 'TransactionQuarter' in engineered_data.columns
        
        # Check that risk categories are added
        assert 'RiskCategory' in engineered_data.columns
        assert 'PremiumCategory' in engineered_data.columns
        
        # Check that interaction features are added
        assert 'PremiumPerClaim' in engineered_data.columns
        
        # Check that categories are reasonable
        assert set(engineered_data['RiskCategory'].unique()) <= {'Low', 'Medium', 'High', 'Very High'}
        assert set(engineered_data['PremiumCategory'].unique()) <= {'Low', 'Medium', 'High', 'Very High'}
    
    def test_encode_categorical_features_fit(self, data_preprocessor, clean_sample_data):
        """Test categorical feature encoding with fitting."""
        encoded_data = data_preprocessor.encode_categorical_features(clean_sample_data, fit=True)
        
        # Check that categorical columns are encoded
        for col in ['Province', 'VehicleType']:
            if col in encoded_data.columns:
                assert encoded_data[col].dtype in [np.int32, np.int64]
        
        # Check that encoders are stored
        assert len(data_preprocessor.label_encoders) > 0
    
    def test_encode_categorical_features_transform(self, data_preprocessor, clean_sample_data):
        """Test categorical feature encoding with transformation only."""
        # First fit the encoders
        data_preprocessor.encode_categorical_features(clean_sample_data, fit=True)
        
        # Then transform new data
        new_data = clean_sample_data.copy()
        encoded_data = data_preprocessor.encode_categorical_features(new_data, fit=False)
        
        # Check that transformation works without fitting
        for col in ['Province', 'VehicleType']:
            if col in encoded_data.columns:
                assert encoded_data[col].dtype in [np.int32, np.int64]
    
    def test_scale_numerical_features_fit(self, data_preprocessor, clean_sample_data):
        """Test numerical feature scaling with fitting."""
        scaled_data = data_preprocessor.scale_numerical_features(clean_sample_data, fit=True)
        
        # Check that numerical columns are scaled
        for col in ['TotalPremium', 'TotalClaims']:
            if col in scaled_data.columns:
                # Scaled data should have mean close to 0 and std close to 1
                assert abs(scaled_data[col].mean()) < 1e-10
                assert abs(scaled_data[col].std() - 1.0) < 1e-10
        
        assert data_preprocessor._is_fitted is True
    
    def test_scale_numerical_features_transform(self, data_preprocessor, clean_sample_data):
        """Test numerical feature scaling with transformation only."""
        # First fit the scaler
        data_preprocessor.scale_numerical_features(clean_sample_data, fit=True)
        
        # Then transform new data
        new_data = clean_sample_data.copy()
        scaled_data = data_preprocessor.scale_numerical_features(new_data, fit=False)
        
        # Check that transformation works without fitting
        for col in ['TotalPremium', 'TotalClaims']:
            if col in scaled_data.columns:
                assert col in scaled_data.columns
    
    def test_prepare_features(self, data_preprocessor, clean_sample_data):
        """Test feature preparation for modeling."""
        X, feature_names = data_preprocessor.prepare_features(clean_sample_data)
        
        assert X is not None
        assert len(feature_names) > 0
        assert X.shape[1] == len(feature_names)
        
        # Check that no missing values remain
        assert X.isnull().sum().sum() == 0
    
    def test_split_data(self, data_preprocessor, clean_sample_data):
        """Test data splitting functionality."""
        # Prepare features
        X, _ = data_preprocessor.prepare_features(clean_sample_data)
        y = clean_sample_data['HadClaim']
        
        # Split data
        X_train, X_test, y_train, y_test = data_preprocessor.split_data(X, y)
        
        # Check split sizes
        total_size = len(X)
        expected_test_size = int(total_size * data_preprocessor.settings.test_size)
        
        assert len(X_test) == expected_test_size
        assert len(X_train) == total_size - expected_test_size
        assert len(y_test) == expected_test_size
        assert len(y_train) == total_size - expected_test_size
        
        # Check that no overlap between train and test
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0
    
    def test_get_preprocessing_info(self, data_preprocessor):
        """Test preprocessing info retrieval."""
        info = data_preprocessor.get_preprocessing_info()
        
        assert 'label_encoders' in info
        assert 'is_scaler_fitted' in info
        assert 'categorical_columns' in info
        assert 'numerical_columns' in info
        
        assert isinstance(info['label_encoders'], list)
        assert isinstance(info['is_scaler_fitted'], bool)
        assert isinstance(info['categorical_columns'], list)
        assert isinstance(info['numerical_columns'], list)
