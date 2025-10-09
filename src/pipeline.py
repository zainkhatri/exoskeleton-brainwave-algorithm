"""
Main pipeline for brainwave analysis.
Orchestrates data loading, preprocessing, ML training, and visualization.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
import mne

from utils.config import Config
from data.loader import DataLoader, create_sample_dataset
from preprocessing.filters import SignalProcessor
from ml.models import GazeDirectionClassifier, EnsembleClassifier
from visualization.plots import BrainwaveVisualizer

logger = logging.getLogger(__name__)


class BrainwaveAnalysisPipeline:
    """Main pipeline for brainwave analysis and gaze direction prediction."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)
        self.config.create_directories()
        
        # Initialize components
        self.data_loader = DataLoader(self.config._config)
        self.signal_processor = SignalProcessor(self.config._config)
        self.classifier = GazeDirectionClassifier(self.config._config)
        self.visualizer = BrainwaveVisualizer(self.config._config)
        
        # Results storage
        self.results = {}
        self.is_trained = False
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Brainwave analysis pipeline initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_config.get('file', 'brainwave_analysis.log')),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self, eeg_file: Optional[str] = None, 
                  eog_file: Optional[str] = None,
                  use_sample_data: bool = False) -> Dict[str, Any]:
        """
        Load EEG and EOG data.
        
        Args:
            eeg_file: Path to EEG data file
            eog_file: Path to EOG data file
            use_sample_data: Whether to use sample data for testing
            
        Returns:
            Dictionary with loaded data
        """
        data_info = {}
        
        if use_sample_data:
            logger.info("Loading sample data for demonstration")
            # Load sample data
            sample_data = self.data_loader.load_sample_data()
            data_info['sample_data'] = sample_data
            
            # Create synthetic data for training
            eeg_data = create_sample_dataset(n_channels=64, duration=60.0)
            eog_data = create_sample_dataset(n_channels=2, duration=60.0)
            
            data_info['eeg_data'] = eeg_data
            data_info['eog_data'] = eog_data
            
        else:
            if not eeg_file or not eog_file:
                raise ValueError("Both EEG and EOG file paths must be provided")
            
            logger.info(f"Loading real data: EEG={eeg_file}, EOG={eog_file}")
            eeg_data = self.data_loader.load_raw_data(eeg_file)
            eog_data = self.data_loader.load_raw_data(eog_file)
            
            data_info['eeg_data'] = eeg_data
            data_info['eog_data'] = eog_data
        
        # Validate data
        for data_type, data in [('EEG', data_info['eeg_data']), ('EOG', data_info['eog_data'])]:
            if not self.data_loader.validate_data(data):
                logger.warning(f"{data_type} data validation failed")
        
        # Get channel information
        data_info['eeg_info'] = self.data_loader.get_channel_info(data_info['eeg_data'])
        data_info['eog_info'] = self.data_loader.get_channel_info(data_info['eog_data'])
        
        self.data_info = data_info
        logger.info("Data loading completed")
        
        return data_info
    
    def preprocess_data(self, data_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Preprocess the loaded data.
        
        Args:
            data_info: Data information dictionary (uses loaded data if None)
            
        Returns:
            Dictionary with preprocessed data and features
        """
        if data_info is None:
            data_info = self.data_info
        
        logger.info("Starting data preprocessing")
        
        # Extract data arrays
        eeg_data = data_info['eeg_data'].get_data()
        eog_data = data_info['eog_data'].get_data()
        
        # Create epochs
        epoch_duration = self.config.get('data.epoch_duration', 1.0)
        overlap = self.config.get('data.overlap', 0.5)
        
        eeg_epochs = mne.make_fixed_length_epochs(
            data_info['eeg_data'], 
            duration=epoch_duration,
            overlap=overlap,
            preload=True
        )
        eog_epochs = mne.make_fixed_length_epochs(
            data_info['eog_data'],
            duration=epoch_duration,
            overlap=overlap,
            preload=True
        )
        
        # Preprocess each epoch
        eeg_features_list = []
        eog_features_list = []
        
        # Get all epoch data at once
        eeg_epochs_data = eeg_epochs.get_data()  # Shape: (n_epochs, n_channels, n_samples)
        eog_epochs_data = eog_epochs.get_data()  # Shape: (n_epochs, n_channels, n_samples)
        
        for i in range(eeg_epochs_data.shape[0]):
            # Get epoch data for this epoch
            eeg_epoch = eeg_epochs_data[i]  # Shape: (n_channels, n_samples)
            eog_epoch = eog_epochs_data[i]  # Shape: (n_channels, n_samples)
            
            # Preprocess EEG epoch
            eeg_result = self.signal_processor.preprocess_pipeline(
                eeg_epoch, extract_features=True
            )
            eeg_features_list.append(eeg_result['features'])
            
            # Preprocess EOG epoch
            eog_result = self.signal_processor.preprocess_pipeline(
                eog_epoch, extract_features=True
            )
            eog_features_list.append(eog_result['features'])
        
        # Combine features
        eeg_features = np.array(eeg_features_list)
        eog_features = np.array(eog_features_list)
        combined_features = np.hstack((eeg_features, eog_features))
        
        # Create labels (for demonstration - replace with real labels)
        n_epochs = len(eeg_epochs)
        labels = np.random.randint(0, 2, size=n_epochs)  # 0 = left, 1 = right
        
        preprocessing_results = {
            'eeg_features': eeg_features,
            'eog_features': eog_features,
            'combined_features': combined_features,
            'labels': labels,
            'eeg_epochs': eeg_epochs,
            'eog_epochs': eog_epochs,
            'n_epochs': n_epochs,
            'n_features': combined_features.shape[1]
        }
        
        self.preprocessing_results = preprocessing_results
        logger.info(f"Preprocessing completed: {n_epochs} epochs, {combined_features.shape[1]} features")
        
        return preprocessing_results
    
    def train_model(self, preprocessing_results: Optional[Dict[str, Any]] = None,
                   hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train the gaze direction classifier.
        
        Args:
            preprocessing_results: Preprocessing results (uses processed data if None)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Training results dictionary
        """
        if preprocessing_results is None:
            preprocessing_results = self.preprocessing_results
        
        logger.info("Starting model training")
        
        # Get features and labels
        X = preprocessing_results['combined_features']
        y = preprocessing_results['labels']
        
        # Create feature names
        feature_names = []
        n_eeg_features = preprocessing_results['eeg_features'].shape[1]
        n_eog_features = preprocessing_results['eog_features'].shape[1]
        
        # EEG feature names
        for i in range(n_eeg_features):
            feature_names.append(f'EEG_feature_{i}')
        
        # EOG feature names
        for i in range(n_eog_features):
            feature_names.append(f'EOG_feature_{i}')
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning")
            tuning_results = self.classifier.hyperparameter_tuning(X, y)
            logger.info(f"Best parameters: {tuning_results['best_params']}")
        
        # Train model
        training_results = self.classifier.train(X, y, feature_names)
        
        # Store results
        self.results.update(training_results)
        self.results['feature_names'] = feature_names
        self.results['model_type'] = self.classifier.model_type
        self.is_trained = True
        
        logger.info("Model training completed")
        
        return training_results
    
    def evaluate_model(self, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            test_data: Test data dictionary (uses training data if None)
            
        Returns:
            Evaluation results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Starting model evaluation")
        
        if test_data is None:
            # Use training data for evaluation
            X = self.preprocessing_results['combined_features']
            y = self.preprocessing_results['labels']
        else:
            X = test_data['features']
            y = test_data['labels']
        
        # Evaluate model
        evaluation_results = self.classifier.evaluate_model(X, y)
        
        # Get feature importance
        feature_importance = self.classifier.get_feature_importance()
        if feature_importance is not None:
            evaluation_results['feature_importance'] = feature_importance
        
        # Store results
        self.results.update(evaluation_results)
        
        logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def predict(self, eeg_data: np.ndarray, eog_data: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            eeg_data: New EEG data
            eog_data: New EOG data
            
        Returns:
            Prediction results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making predictions on new data")
        
        # Preprocess new data
        eeg_result = self.signal_processor.preprocess_pipeline(
            eeg_data, extract_features=True
        )
        eog_result = self.signal_processor.preprocess_pipeline(
            eog_data, extract_features=True
        )
        
        # Combine features
        combined_features = np.hstack((
            eeg_result['features'].reshape(1, -1),
            eog_result['features'].reshape(1, -1)
        ))
        
        # Make prediction
        prediction = self.classifier.predict(combined_features)
        probabilities = self.classifier.predict_proba(combined_features)
        
        prediction_results = {
            'prediction': prediction[0],
            'probabilities': probabilities[0],
            'gaze_direction': 'Left' if prediction[0] == 0 else 'Right',
            'confidence': np.max(probabilities[0])
        }
        
        logger.info(f"Prediction: {prediction_results['gaze_direction']} (confidence: {prediction_results['confidence']:.3f})")
        
        return prediction_results
    
    def create_visualizations(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive visualizations.
        
        Args:
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary with created plots
        """
        logger.info("Creating visualizations")
        
        plots = {}
        
        # Plot raw data
        if hasattr(self, 'data_info'):
            plots['raw_eeg'] = self.visualizer.plot_raw_data(
                self.data_info['eeg_data'],
                title="Raw EEG Data",
                save_name="raw_eeg_data" if save_plots else None
            )
            
            plots['raw_eog'] = self.visualizer.plot_raw_data(
                self.data_info['eog_data'],
                title="Raw EOG Data",
                save_name="raw_eog_data" if save_plots else None
            )
        
        # Plot epochs
        if hasattr(self, 'preprocessing_results'):
            plots['eeg_epochs'] = self.visualizer.plot_epochs(
                self.preprocessing_results['eeg_epochs'],
                title="EEG Epochs",
                save_name="eeg_epochs" if save_plots else None
            )
        
        # Plot model performance
        if self.results:
            if 'confusion_matrix' in self.results:
                plots['confusion_matrix'] = self.visualizer.plot_confusion_matrix(
                    self.results['confusion_matrix'],
                    class_names=['Left', 'Right'],
                    title="Confusion Matrix",
                    save_name="confusion_matrix" if save_plots else None
                )
            
            if 'feature_importance' in self.results:
                plots['feature_importance'] = self.visualizer.plot_feature_importance(
                    self.results['feature_importance'],
                    feature_names=self.results.get('feature_names'),
                    title="Feature Importance",
                    save_name="feature_importance" if save_plots else None
                )
        
        # Create summary report
        if self.results:
            plots['summary_report'] = self.visualizer.create_summary_report(
                self.results,
                save_name="analysis_summary" if save_plots else None
            )
        
        logger.info(f"Created {len(plots)} visualizations")
        
        return plots
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.classifier.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the model file
        """
        self.classifier.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def run_full_pipeline(self, eeg_file: Optional[str] = None,
                         eog_file: Optional[str] = None,
                         use_sample_data: bool = True,
                         hyperparameter_tuning: bool = False,
                         create_plots: bool = True) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            eeg_file: Path to EEG data file
            eog_file: Path to EOG data file
            use_sample_data: Whether to use sample data
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            create_plots: Whether to create visualizations
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting full analysis pipeline")
        
        # Step 1: Load data
        data_info = self.load_data(eeg_file, eog_file, use_sample_data)
        
        # Step 2: Preprocess data
        preprocessing_results = self.preprocess_data(data_info)
        
        # Step 3: Train model
        training_results = self.train_model(preprocessing_results, hyperparameter_tuning)
        
        # Step 4: Evaluate model
        evaluation_results = self.evaluate_model()
        
        # Step 5: Create visualizations
        if create_plots:
            plots = self.create_visualizations()
            self.results['plots'] = plots
        
        # Combine all results
        pipeline_results = {
            'data_info': data_info,
            'preprocessing_results': preprocessing_results,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'final_results': self.results
        }
        
        logger.info("Full analysis pipeline completed")
        
        return pipeline_results


def main():
    """Main function for running the analysis pipeline."""
    # Initialize pipeline
    pipeline = BrainwaveAnalysisPipeline()
    
    # Run full pipeline with sample data
    results = pipeline.run_full_pipeline(
        use_sample_data=True,
        hyperparameter_tuning=False,
        create_plots=True
    )
    
    # Print summary
    print("\n" + "="*50)
    print("BRAINWAVE ANALYSIS SUMMARY")
    print("="*50)
    print(f"Model Type: {results['final_results'].get('model_type', 'Unknown')}")
    print(f"Test Accuracy: {results['final_results'].get('accuracy', 0):.3f}")
    print(f"Cross-validation Mean: {results['final_results'].get('cv_mean', 0):.3f} ± {results['final_results'].get('cv_std', 0):.3f}")
    print(f"Number of Features: {results['final_results'].get('n_features', 0)}")
    print(f"Number of Samples: {results['final_results'].get('n_samples', 0)}")
    print("="*50)


if __name__ == "__main__":
    main()
