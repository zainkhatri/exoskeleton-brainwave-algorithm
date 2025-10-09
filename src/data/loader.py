"""
Data loading utilities for EEG and EOG data.
Supports multiple file formats and provides robust error handling.
"""

import os
import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of EEG and EOG data from various file formats."""
    
    SUPPORTED_FORMATS = {
        '.fif': 'FIF',
        '.edf': 'EDF',
        '.bdf': 'BDF',
        '.set': 'EEGLAB',
        '.vhdr': 'BrainVision',
        '.eeg': 'BrainVision',
        '.vmrk': 'BrainVision'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 256)
        
    def detect_file_format(self, file_path: str) -> str:
        """
        Detect file format from extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File format string
        """
        ext = Path(file_path).suffix.lower()
        if ext in self.SUPPORTED_FORMATS:
            return self.SUPPORTED_FORMATS[ext]
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(self.SUPPORTED_FORMATS.keys())}")
    
    def load_raw_data(self, file_path: str, preload: bool = True) -> mne.io.Raw:
        """
        Load raw EEG/EOG data from file.
        
        Args:
            file_path: Path to the data file
            preload: Whether to preload data into memory
            
        Returns:
            MNE Raw object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            file_format = self.detect_file_format(str(file_path))
            logger.info(f"Loading {file_format} file: {file_path}")
            
            # Load based on format
            if file_format == 'FIF':
                raw = mne.io.read_raw_fif(file_path, preload=preload, verbose=False)
            elif file_format == 'EDF':
                raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=False)
            elif file_format == 'BDF':
                raw = mne.io.read_raw_bdf(file_path, preload=preload, verbose=False)
            elif file_format == 'EEGLAB':
                raw = mne.io.read_raw_eeglab(file_path, preload=preload, verbose=False)
            elif file_format == 'BrainVision':
                raw = mne.io.read_raw_brainvision(file_path, preload=preload, verbose=False)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            logger.info(f"Successfully loaded data: {raw.info['nchan']} channels, {raw.times[-1]:.2f}s duration")
            return raw
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def load_sample_data(self, dataset: str = 'sample') -> mne.EvokedArray:
        """
        Load sample data for testing and demonstration.
        
        Args:
            dataset: Dataset name ('sample' for MNE sample data)
            
        Returns:
            Sample evoked data
        """
        try:
            if dataset == 'sample':
                sample_data_folder = mne.datasets.sample.data_path()
                sample_data_file = os.path.join(
                    sample_data_folder, "MEG", "sample", "sample_audvis-ave.fif"
                )
                evoked_data = mne.read_evokeds(
                    sample_data_file, baseline=(None, 0), proj=True, verbose=False
                )
                logger.info("Loaded MNE sample data")
                return evoked_data
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
                
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            raise
    
    def load_epochs_data(self, raw_data: mne.io.Raw, events: Optional[np.ndarray] = None, 
                        event_id: Optional[Dict[str, int]] = None,
                        tmin: float = -0.2, tmax: float = 0.5) -> mne.Epochs:
        """
        Load and create epochs from raw data.
        
        Args:
            raw_data: MNE Raw object
            events: Events array (if None, will try to find events in file)
            event_id: Event ID mapping
            tmin: Start time before event (seconds)
            tmax: End time after event (seconds)
            
        Returns:
            MNE Epochs object
        """
        raw = raw_data
        
        # Find events if not provided
        if events is None:
            try:
                events = mne.find_events(raw, verbose=False)
                logger.info(f"Found {len(events)} events")
            except ValueError:
                logger.warning("No events found, creating fixed-length epochs")
                return mne.make_fixed_length_epochs(
                    raw, duration=self.config.get('epoch_duration', 1.0), 
                    overlap=self.config.get('overlap', 0.5), preload=True
                )
        
        # Create epochs
        epochs = mne.Epochs(
            raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
            baseline=(None, 0), preload=True, verbose=False
        )
        
        logger.info(f"Created {len(epochs)} epochs")
        return epochs
    
    def load_multiple_files(self, file_paths: List[str]) -> List[mne.io.Raw]:
        """
        Load multiple data files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of MNE Raw objects
        """
        raw_data_list = []
        
        for file_path in file_paths:
            try:
                raw = self.load_raw_data(file_path)
                raw_data_list.append(raw)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(raw_data_list)}/{len(file_paths)} files")
        return raw_data_list
    
    def get_channel_info(self, raw: mne.io.Raw) -> Dict[str, Any]:
        """
        Extract channel information from raw data.
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Dictionary with channel information
        """
        info = raw.info
        ch_types = [info['chs'][i]['kind'] for i in range(info['nchan'])]
        
        channel_info = {
            'n_channels': info['nchan'],
            'sample_rate': info['sfreq'],
            'duration': raw.times[-1],
            'channel_names': info['ch_names'],
            'channel_types': ch_types,
            'eeg_channels': mne.pick_types(info, eeg=True),
            'eog_channels': mne.pick_types(info, eog=True),
            'meg_channels': mne.pick_types(info, meg=True)
        }
        
        return channel_info
    
    def validate_data(self, raw: mne.io.Raw) -> bool:
        """
        Validate loaded data for common issues.
        
        Args:
            raw: MNE Raw object
            
        Returns:
            True if data is valid, False otherwise
        """
        issues = []
        
        # Check for NaN values
        if np.isnan(raw.get_data()).any():
            issues.append("Data contains NaN values")
        
        # Check for infinite values
        if np.isinf(raw.get_data()).any():
            issues.append("Data contains infinite values")
        
        # Check sample rate
        if raw.info['sfreq'] <= 0:
            issues.append("Invalid sample rate")
        
        # Check duration
        if raw.times[-1] <= 0:
            issues.append("Invalid data duration")
        
        # Check for channels
        if raw.info['nchan'] == 0:
            issues.append("No channels found")
        
        if issues:
            logger.warning(f"Data validation issues: {issues}")
            return False
        
        logger.info("Data validation passed")
        return True


def create_sample_dataset(n_channels: int = 64, duration: float = 60.0, 
                         sample_rate: int = 256) -> mne.io.Raw:
    """
    Create synthetic EEG data for testing.
    
    Args:
        n_channels: Number of channels
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Synthetic MNE Raw object
    """
    # Create channel names
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    
    # Create info object
    info = mne.create_info(ch_names=ch_names, sfreq=sample_rate, ch_types=ch_types)
    
    # Generate synthetic data (mix of different frequency components)
    n_samples = int(duration * sample_rate)
    data = np.zeros((n_channels, n_samples))
    
    for i in range(n_channels):
        # Alpha waves (8-13 Hz)
        alpha = np.sin(2 * np.pi * 10 * np.linspace(0, duration, n_samples))
        # Beta waves (13-30 Hz)
        beta = 0.5 * np.sin(2 * np.pi * 20 * np.linspace(0, duration, n_samples))
        # Theta waves (4-8 Hz)
        theta = 0.3 * np.sin(2 * np.pi * 6 * np.linspace(0, duration, n_samples))
        # Noise
        noise = 0.1 * np.random.randn(n_samples)
        
        data[i] = alpha + beta + theta + noise
    
    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    
    logger.info(f"Created synthetic dataset: {n_channels} channels, {duration}s duration")
    return raw
