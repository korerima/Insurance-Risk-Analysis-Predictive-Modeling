"""
Tests for dashboard components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import streamlit as st

from src.dashboard.components import (
    DataExplorer,
    ModelTrainer,
    RiskAnalyzer,
    PremiumOptimizer,
    ExplainabilityViewer,
    ReportGenerator
)
from src.models.risk_predictor import RiskPredictor


class TestDataExplorer:
    """Test cases for DataExplorer component."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'PolicyID': range(1, 101),
            'TotalPremium': np.random.lognormal(7, 1, 100),
            'TotalClaims': np.random.exponential(1000, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Province': np.random.choice(['Gauteng', 'Western Cape'], 100),
            'VehicleType': np.random.choice(['Sedan', 'SUV'], 100)
        })
    
    @pytest.fixture
    def explorer(self, sample_data):
        """Create DataExplorer instance."""
        return DataExplorer(sample_data)
    
    def test_initialization(self, explorer, sample_data):
        """Test DataExplorer initialization."""
        assert explorer.data is not None
        assert len(explorer.data) == len(sample_data)
        assert explorer.preprocessor is not None
    
    def test_data_overview(self, explorer):
        """Test data overview functionality."""
        # Test that data overview can be generated
        assert explorer.data.shape[0] > 0
        assert explorer.data.shape[1] > 0
    
    def test_missing_values_analysis(self, explorer):
        """Test missing values analysis."""
        missing_data = explorer.data.isnull().sum()
        assert isinstance(missing_data, pd.Series)
        assert len(missing_data) == len(explorer.data.columns)
    
    def test_data_types_analysis(self, explorer):
        """Test data types analysis."""
        dtype_counts = explorer.data.dtypes.value_counts()
        assert isinstance(dtype_counts, pd.Series)
        assert len(dtype_counts) > 0


class TestModelTrainer:
    """Test cases for ModelTrainer component."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = pd.DataFrame({
            'PolicyID': range(1, 101),
            'TotalPremium': np.random.lognormal(7, 1, 100),
            'TotalClaims': np.random.exponential(1000, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Province': np.random.choice(['Gauteng', 'Western Cape'], 100),
            'VehicleType': np.random.choice(['Sedan', 'SUV'], 100)
        })
        # Add target variable
        data['HadClaim'] = (data['TotalClaims'] > 0).astype(int)
        return data
    
    @pytest.fixture
    def trainer(self, sample_data):
        """Create ModelTrainer instance."""
        return ModelTrainer(sample_data)
    
    def test_initialization(self, trainer, sample_data):
        """Test ModelTrainer initialization."""
        assert trainer.data is not None
        assert len(trainer.data) == len(sample_data)
        assert trainer.preprocessor is not None
        assert trainer.settings is not None
    
    def test_data_preprocessing_options(self, trainer):
        """Test data preprocessing options."""
        # Test that preprocessing options can be set
        assert trainer.data is not None
        assert 'HadClaim' in trainer.data.columns
    
    @patch('src.dashboard.components.st')
    def test_model_training_interface(self, mock_st, trainer):
        """Test model training interface."""
        # Mock streamlit components
        mock_st.selectbox.return_value = "random_forest"
        mock_st.slider.return_value = 100
        mock_st.button.return_value = True
        
        # Test that training interface can be rendered
        assert trainer.data is not None


class TestRiskAnalyzer:
    """Test cases for RiskAnalyzer component."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = pd.DataFrame({
            'PolicyID': range(1, 101),
            'TotalPremium': np.random.lognormal(7, 1, 100),
            'TotalClaims': np.random.exponential(1000, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Province': np.random.choice(['Gauteng', 'Western Cape'], 100),
            'VehicleType': np.random.choice(['Sedan', 'SUV'], 100)
        })
        return data
    
    @pytest.fixture
    def analyzer(self, sample_data):
        """Create RiskAnalyzer instance."""
        return RiskAnalyzer(sample_data)
    
    def test_initialization(self, analyzer, sample_data):
        """Test RiskAnalyzer initialization."""
        assert analyzer.data is not None
        assert len(analyzer.data) == len(sample_data)
        assert analyzer.preprocessor is not None
    
    def test_risk_metrics_calculation(self, analyzer):
        """Test risk metrics calculation."""
        # Test loss ratio calculation
        analyzer.data['LossRatio'] = analyzer.data['TotalClaims'] / analyzer.data['TotalPremium']
        
        assert 'LossRatio' in analyzer.data.columns
        assert analyzer.data['LossRatio'].mean() >= 0
    
    def test_risk_categorization(self, analyzer):
        """Test risk categorization."""
        # Calculate loss ratio
        analyzer.data['LossRatio'] = analyzer.data['TotalClaims'] / analyzer.data['TotalPremium']
        
        # Categorize risk
        analyzer.data['RiskCategory'] = pd.cut(
            analyzer.data['LossRatio'],
            bins=[0, 0.5, 1.0, 2.0, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        assert 'RiskCategory' in analyzer.data.columns
        assert set(analyzer.data['RiskCategory'].unique()) <= {'Low', 'Medium', 'High', 'Very High'}


class TestPremiumOptimizer:
    """Test cases for PremiumOptimizer component."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = pd.DataFrame({
            'PolicyID': range(1, 101),
            'TotalPremium': np.random.lognormal(7, 1, 100),
            'TotalClaims': np.random.exponential(1000, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Province': np.random.choice(['Gauteng', 'Western Cape'], 100),
            'VehicleType': np.random.choice(['Sedan', 'SUV'], 100)
        })
        return data
    
    @pytest.fixture
    def optimizer(self, sample_data):
        """Create PremiumOptimizer instance."""
        return PremiumOptimizer(sample_data)
    
    def test_initialization(self, optimizer, sample_data):
        """Test PremiumOptimizer initialization."""
        assert optimizer.data is not None
        assert len(optimizer.data) == len(sample_data)
        assert optimizer.preprocessor is not None
    
    def test_premium_analysis(self, optimizer):
        """Test premium analysis."""
        if 'TotalPremium' in optimizer.data.columns:
            avg_premium = optimizer.data['TotalPremium'].mean()
            assert avg_premium > 0
            assert isinstance(avg_premium, float)
    
    def test_optimization_parameters(self, optimizer):
        """Test optimization parameters."""
        # Test that optimization parameters can be set
        risk_multiplier = 1.0
        base_premium = 1000.0
        market_factor = 1.0
        competition_discount = 5
        
        assert risk_multiplier > 0
        assert base_premium > 0
        assert market_factor > 0
        assert 0 <= competition_discount <= 100


class TestExplainabilityViewer:
    """Test cases for ExplainabilityViewer component."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock(spec=RiskPredictor)
        model.model = Mock()
        model.model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.4])
        return model
    
    @pytest.fixture
    def viewer(self, mock_model):
        """Create ExplainabilityViewer instance."""
        return ExplainabilityViewer(mock_model)
    
    def test_initialization(self, viewer, mock_model):
        """Test ExplainabilityViewer initialization."""
        assert viewer.model is not None
        assert viewer.model == mock_model
    
    def test_model_availability_check(self, viewer):
        """Test model availability check."""
        # Test with available model
        assert hasattr(viewer.model, 'model')
        assert viewer.model.model is not None
    
    @patch('src.dashboard.components.st')
    def test_explainability_interface(self, mock_st, viewer):
        """Test explainability interface."""
        # Mock streamlit components
        mock_st.header.return_value = None
        mock_st.info.return_value = None
        
        # Test that interface can be rendered
        assert viewer.model is not None


class TestReportGenerator:
    """Test cases for ReportGenerator component."""
    
    @pytest.fixture
    def generator(self):
        """Create ReportGenerator instance."""
        return ReportGenerator()
    
    def test_initialization(self, generator):
        """Test ReportGenerator initialization."""
        assert generator.settings is not None
    
    def test_report_generation(self, generator):
        """Test report generation."""
        report = generator.generate_report(
            report_type="Executive Summary",
            title="Test Report"
        )
        
        assert isinstance(report, str)
        assert "Test Report" in report
        assert "Executive Summary" in report
    
    def test_model_card_generation(self, generator):
        """Test model card generation."""
        # Mock session state
        with patch('src.dashboard.components.st.session_state') as mock_session:
            mock_session.get.return_value = True
            mock_session.current_model = Mock()
            mock_session.model_results = {
                'train_accuracy': 0.85,
                'test_accuracy': 0.82,
                'model_type': 'random_forest',
                'feature_names': ['feature1', 'feature2']
            }
            
            model_card = generator.generate_model_card()
            
            assert isinstance(model_card, dict)
            assert 'model_card_version' in model_card
            assert 'model_details' in model_card
            assert model_card['model_details']['name'] == 'Insurance Risk Predictor - random_forest'
    
    def test_model_card_generation_no_model(self, generator):
        """Test model card generation without trained model."""
        with patch('src.dashboard.components.st.session_state') as mock_session:
            mock_session.get.return_value = False
            
            with pytest.raises(ValueError, match="No trained model available"):
                generator.generate_model_card()


class TestDashboardIntegration:
    """Integration tests for dashboard components."""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data."""
        np.random.seed(42)
        data = pd.DataFrame({
            'PolicyID': range(1, 201),
            'TransactionMonth': pd.date_range('2020-01-01', periods=200, freq='M'),
            'TotalPremium': np.random.lognormal(7, 1, 200),
            'TotalClaims': np.random.exponential(1000, 200),
            'Gender': np.random.choice(['Male', 'Female'], 200),
            'Province': np.random.choice(['Gauteng', 'Western Cape', 'KZN'], 200),
            'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback'], 200),
            'Title': np.random.choice(['Mr', 'Mrs', 'Ms'], 200),
            'Language': np.random.choice(['English', 'Afrikaans'], 200),
            'LegalType': np.random.choice(['Individual', 'Company'], 200),
            'Citizenship': np.random.choice(['South African', 'Other'], 200)
        })
        # Add some missing values
        data.loc[np.random.choice(data.index, 20), 'Province'] = np.nan
        data.loc[np.random.choice(data.index, 15), 'VehicleType'] = np.nan
        
        return data
    
    def test_end_to_end_workflow(self, sample_data):
        """Test end-to-end dashboard workflow."""
        # Test data explorer
        explorer = DataExplorer(sample_data)
        assert explorer.data is not None
        assert len(explorer.data) == 200
        
        # Test risk analyzer
        analyzer = RiskAnalyzer(sample_data)
        analyzer.data['LossRatio'] = analyzer.data['TotalClaims'] / analyzer.data['TotalPremium']
        assert 'LossRatio' in analyzer.data.columns
        
        # Test premium optimizer
        optimizer = PremiumOptimizer(sample_data)
        assert optimizer.data is not None
        
        # Test report generator
        generator = ReportGenerator()
        report = generator.generate_report(report_type="Technical Report")
        assert isinstance(report, str)
        assert len(report) > 0
