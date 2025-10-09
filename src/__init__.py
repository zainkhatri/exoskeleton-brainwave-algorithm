"""
Brainwave Analysis Package
A comprehensive toolkit for EEG/EOG data analysis and gaze direction prediction.
"""

from .pipeline import BrainwaveAnalysisPipeline
from .data.loader import DataLoader, create_sample_dataset
from .preprocessing.filters import SignalProcessor
from .ml.models import GazeDirectionClassifier, EnsembleClassifier
from .visualization.plots import BrainwaveVisualizer
from .utils.config import Config

__version__ = "1.0.0"
__author__ = "Brainwave Analysis Team"

__all__ = [
    'BrainwaveAnalysisPipeline',
    'DataLoader',
    'create_sample_dataset',
    'SignalProcessor',
    'GazeDirectionClassifier',
    'EnsembleClassifier',
    'BrainwaveVisualizer',
    'Config'
]
