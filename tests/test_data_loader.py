"""
Unit tests for data loader module.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_initialization(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader is not None
        assert data_loader._data is None
    
    def test_load_data_success(self, data_loader, sample_data, temp_data_file):
        """Test successful data loading."""
        # Save sample data to temp file
        sample_data.to_csv(temp_data_file, index=False)
        
        # Load data
        loaded_data = data_loader.load_data(temp_data_file)
        
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
        assert data_loader._data is not None
    
    def test_load_data_file_not_found(self, data_loader):
        """Test data loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_data("non_existent_file.csv")
    
    def test_load_data_empty_file(self, data_loader, temp_data_file):
        """Test data loading with empty file."""
        # Create empty file
        temp_data_file.write_text("")
        
        with pytest.raises(ValueError, match="Data file is empty"):
            data_loader.load_data(temp_data_file)
    
    def test_save_data_success(self, data_loader, sample_data, temp_data_file):
        """Test successful data saving."""
        data_loader.save_data(sample_data, temp_data_file)
        
        assert temp_data_file.exists()
        
        # Verify saved data
        saved_data = pd.read_csv(temp_data_file)
        assert len(saved_data) == len(sample_data)
        assert list(saved_data.columns) == list(sample_data.columns)
    
    def test_get_data_info_no_data(self, data_loader):
        """Test get_data_info with no loaded data."""
        info = data_loader.get_data_info()
        assert info["error"] == "No data loaded"
    
    def test_get_data_info_with_data(self, data_loader, sample_data, temp_data_file):
        """Test get_data_info with loaded data."""
        # Save and load data
        sample_data.to_csv(temp_data_file, index=False)
        data_loader.load_data(temp_data_file)
        
        info = data_loader.get_data_info()
        
        assert info["shape"] == sample_data.shape
        assert info["columns"] == list(sample_data.columns)
        assert info["total_rows"] == len(sample_data)
        assert info["total_columns"] == len(sample_data.columns)
    
    def test_get_data(self, data_loader, sample_data, temp_data_file):
        """Test get_data method."""
        # Initially no data
        assert data_loader.get_data() is None
        
        # Load data
        sample_data.to_csv(temp_data_file, index=False)
        data_loader.load_data(temp_data_file)
        
        # Now should have data
        loaded_data = data_loader.get_data()
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
