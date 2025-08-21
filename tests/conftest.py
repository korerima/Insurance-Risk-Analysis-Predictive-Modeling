"""
Pytest configuration and fixtures for testing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from src.config.settings import Settings, get_settings
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator
from src.models.risk_predictor import RiskPredictor


@pytest.fixture
def settings() -> Settings:
    """Provide test settings."""
    return get_settings()


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    
    n_samples = 1000
    
    data = {
        'PolicyID': range(1, n_samples + 1),
        'TransactionMonth': pd.date_range('2020-01-01', periods=n_samples, freq='M'),
        'TotalPremium': np.random.lognormal(7, 1, n_samples),
        'TotalClaims': np.random.exponential(1000, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Province': np.random.choice(['Gauteng', 'Western Cape', 'KZN'], n_samples),
        'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback'], n_samples),
        'Title': np.random.choice(['Mr', 'Mrs', 'Ms'], n_samples),
        'Language': np.random.choice(['English', 'Afrikaans'], n_samples),
        'LegalType': np.random.choice(['Individual', 'Company'], n_samples),
        'Citizenship': np.random.choice(['South African', 'Other'], n_samples),
        'IsVATRegistered': np.random.choice([True, False], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'Province'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'VehicleType'] = np.nan
    
    # Add some invalid records
    df.loc[np.random.choice(df.index, 10), 'TotalPremium'] = -100
    
    return df


@pytest.fixture
def clean_sample_data(sample_data) -> pd.DataFrame:
    """Create cleaned sample data for testing."""
    preprocessor = DataPreprocessor()
    return preprocessor.clean_data(sample_data)


@pytest.fixture
def data_loader() -> DataLoader:
    """Provide data loader instance."""
    return DataLoader()


@pytest.fixture
def data_preprocessor() -> DataPreprocessor:
    """Provide data preprocessor instance."""
    return DataPreprocessor()


@pytest.fixture
def data_validator() -> DataValidator:
    """Provide data validator instance."""
    return DataValidator()


@pytest.fixture
def risk_predictor() -> RiskPredictor:
    """Provide risk predictor instance."""
    return RiskPredictor(model_type="random_forest", n_estimators=10)


@pytest.fixture
def temp_data_file(tmp_path) -> Path:
    """Create a temporary data file for testing."""
    return tmp_path / "test_data.csv"


@pytest.fixture
def temp_model_file(tmp_path) -> Path:
    """Create a temporary model file for testing."""
    return tmp_path / "test_model.pkl"


@pytest.fixture
def mock_settings() -> Dict[str, Any]:
    """Provide mock settings for testing."""
    return {
        'data_dir': Path('test_data'),
        'models_dir': Path('test_models'),
        'logs_dir': Path('test_logs'),
        'data_file': 'test.csv',
        'delimiter': ',',
        'random_state': 42,
        'test_size': 0.2,
        'categorical_columns': ['Gender', 'Province', 'VehicleType'],
        'numerical_columns': ['TotalPremium', 'TotalClaims'],
        'log_level': 'DEBUG'
    }
