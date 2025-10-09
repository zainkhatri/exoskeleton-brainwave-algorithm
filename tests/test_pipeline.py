"""
Tests for the brainwave analysis pipeline.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import BrainwaveAnalysisPipeline
from data.loader import DataLoader, create_sample_dataset
from preprocessing.filters import SignalProcessor
from ml.models import GazeDirectionClassifier
from visualization.plots import BrainwaveVisualizer
from utils.config import Config


class TestConfig:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = Config()
        assert config.get('data.sample_rate') == 256
        assert config.get('ml.model_type') == 'random_forest'
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        # Should not raise an exception
        assert config.get('data.sample_rate') > 0


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_sample_dataset_creation(self):
        """Test synthetic dataset creation."""
        raw = create_sample_dataset(n_channels=32, duration=10.0)
        assert raw.info['nchan'] == 32
        assert raw.times[-1] == 10.0
    
    def test_data_validation(self):
        """Test data validation."""
        loader = DataLoader()
        raw = create_sample_dataset(n_channels=16, duration=5.0)
        assert loader.validate_data(raw) == True


class TestSignalProcessor:
    """Test signal processing functionality."""
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        processor = SignalProcessor()
        data = np.random.randn(1000)
        filtered = processor.apply_bandpass_filter(data)
        assert filtered.shape == data.shape
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        processor = SignalProcessor()
        data = np.random.randn(10, 1000)  # 10 channels, 1000 samples
        features = processor.extract_all_features(data)
        assert len(features) > 0
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        processor = SignalProcessor()
        data = np.random.randn(5, 500)
        result = processor.preprocess_pipeline(data)
        
        assert 'original_data' in result
        assert 'filtered_data' in result
        assert 'cleaned_data' in result
        assert 'features' in result


class TestGazeDirectionClassifier:
    """Test machine learning models."""
    
    def test_model_creation(self):
        """Test model creation."""
        classifier = GazeDirectionClassifier()
        assert classifier.model_type == 'random_forest'
        assert not classifier.is_trained
    
    def test_model_training(self):
        """Test model training."""
        classifier = GazeDirectionClassifier()
        
        # Create synthetic data
        X = np.random.randn(100, 20)  # 100 samples, 20 features
        y = np.random.randint(0, 2, 100)  # Binary labels
        
        results = classifier.train(X, y)
        assert classifier.is_trained
        assert 'test_accuracy' in results
        assert 'cv_mean' in results
    
    def test_model_prediction(self):
        """Test model prediction."""
        classifier = GazeDirectionClassifier()
        
        # Train model
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        classifier.train(X, y)
        
        # Make prediction
        X_new = np.random.randn(5, 20)
        predictions = classifier.predict(X_new)
        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)


class TestBrainwaveVisualizer:
    """Test visualization functionality."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = BrainwaveVisualizer()
        assert visualizer.save_plots == True
        assert visualizer.plot_format == 'png'
    
    def test_confusion_matrix_plot(self):
        """Test confusion matrix plotting."""
        visualizer = BrainwaveVisualizer()
        cm = np.array([[45, 5], [8, 42]])
        
        # Should not raise an exception
        fig = visualizer.plot_confusion_matrix(cm, show_plots=False)
        assert fig is not None


class TestBrainwaveAnalysisPipeline:
    """Test the main analysis pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = BrainwaveAnalysisPipeline()
        assert pipeline.config is not None
        assert not pipeline.is_trained
    
    def test_data_loading(self):
        """Test data loading."""
        pipeline = BrainwaveAnalysisPipeline()
        data_info = pipeline.load_data(use_sample_data=True)
        
        assert 'eeg_data' in data_info
        assert 'eog_data' in data_info
        assert 'eeg_info' in data_info
        assert 'eog_info' in data_info
    
    def test_preprocessing(self):
        """Test data preprocessing."""
        pipeline = BrainwaveAnalysisPipeline()
        data_info = pipeline.load_data(use_sample_data=True)
        preprocessing_results = pipeline.preprocess_data(data_info)
        
        assert 'combined_features' in preprocessing_results
        assert 'labels' in preprocessing_results
        assert 'n_epochs' in preprocessing_results
    
    def test_full_pipeline(self):
        """Test complete pipeline."""
        pipeline = BrainwaveAnalysisPipeline()
        results = pipeline.run_full_pipeline(
            use_sample_data=True,
            hyperparameter_tuning=False,
            create_plots=False  # Disable plots for testing
        )
        
        assert 'data_info' in results
        assert 'preprocessing_results' in results
        assert 'training_results' in results
        assert 'evaluation_results' in results
        assert 'final_results' in results
        
        # Check that model is trained
        assert pipeline.is_trained


if __name__ == "__main__":
    pytest.main([__file__])
