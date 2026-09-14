"""
Signal preprocessing and filtering utilities for EEG/EOG data.
Includes artifact removal, filtering, and feature extraction.
"""

import numpy as np
import mne
from scipy.signal import butter, lfilter, filtfilt, welch
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Handles signal preprocessing for EEG/EOG data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize signal processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 256)
        self.filter_config = self.config.get('filtering', {})
        
    def apply_bandpass_filter(self, data: np.ndarray, 
                            low_cutoff: Optional[float] = None,
                            high_cutoff: Optional[float] = None,
                            filter_order: Optional[int] = None,
                            sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to data.
        
        Args:
            data: Input signal data
            low_cutoff: Low cutoff frequency (Hz)
            high_cutoff: High cutoff frequency (Hz)
            filter_order: Filter order
            sample_rate: Sampling rate (Hz)
            
        Returns:
            Filtered data
        """
        # Use config defaults if not provided
        low_cutoff = low_cutoff or self.filter_config.get('low_cutoff', 1.0)
        high_cutoff = high_cutoff or self.filter_config.get('high_cutoff', 50.0)
        filter_order = filter_order or self.filter_config.get('filter_order', 5)
        sample_rate = sample_rate or self.sample_rate
        
        # Validate parameters
        if low_cutoff >= high_cutoff:
            raise ValueError("Low cutoff must be less than high cutoff")
        if high_cutoff >= sample_rate / 2:
            raise ValueError("High cutoff must be less than Nyquist frequency")
        
        # Calculate normalized frequencies
        nyquist = 0.5 * sample_rate
        low_norm = low_cutoff / nyquist
        high_norm = high_cutoff / nyquist
        
        # Design filter
        b, a = butter(filter_order, [low_norm, high_norm], btype='band')
        
        # Apply filter (use filtfilt for zero-phase filtering)
        if data.ndim == 1:
            filtered_data = filtfilt(b, a, data)
        else:
            filtered_data = np.array([filtfilt(b, a, channel) for channel in data])
        
        logger.debug(f"Applied bandpass filter: {low_cutoff}-{high_cutoff} Hz")
        return filtered_data
    
    def apply_notch_filter(self, data: np.ndarray, 
                          notch_freq: Optional[float] = None,
                          sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Apply notch filter to remove power line noise.
        
        Args:
            data: Input signal data
            notch_freq: Notch frequency (Hz)
            sample_rate: Sampling rate (Hz)
            
        Returns:
            Filtered data
        """
        notch_freq = notch_freq or self.filter_config.get('notch_freq', 60.0)
        sample_rate = sample_rate or self.sample_rate
        
        # Use MNE's notch filter
        if data.ndim == 1:
            # For 1D data, create a temporary Raw object
            info = mne.create_info(['ch1'], sample_rate, 'eeg')
            raw = mne.io.RawArray(data.reshape(1, -1), info, verbose=False)
            raw.notch_filter(notch_freq, verbose=False)
            filtered_data = raw.get_data()[0]
        else:
            # For 2D data
            info = mne.create_info([f'ch{i}' for i in range(data.shape[0])], 
                                 sample_rate, 'eeg')
            raw = mne.io.RawArray(data, info, verbose=False)
            raw.notch_filter(notch_freq, verbose=False)
            filtered_data = raw.get_data()
        
        logger.debug(f"Applied notch filter at {notch_freq} Hz")
        return filtered_data
    
    def remove_artifacts(self, data: np.ndarray, 
                        method: str = 'statistical',
                        threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove artifacts from data.
        
        Args:
            data: Input signal data
            method: Artifact removal method ('statistical', 'amplitude')
            threshold: Threshold for artifact detection
            
        Returns:
            Tuple of (cleaned_data, artifact_mask)
        """
        if method == 'statistical':
            return self._remove_artifacts_statistical(data, threshold)
        elif method == 'amplitude':
            return self._remove_artifacts_amplitude(data, threshold)
        else:
            raise ValueError(f"Unknown artifact removal method: {method}")
    
    def _remove_artifacts_statistical(self, data: np.ndarray, 
                                    threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Remove artifacts using statistical methods."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        artifact_mask = np.zeros(data.shape[1], dtype=bool)
        
        for i in range(data.shape[0]):
            channel_data = data[i]
            
            # Z-score based artifact detection
            z_scores = np.abs((channel_data - np.mean(channel_data)) / np.std(channel_data))
            channel_artifacts = z_scores > threshold
            artifact_mask |= channel_artifacts
        
        # Remove artifacts
        cleaned_data = data.copy()
        cleaned_data[:, artifact_mask] = np.nan
        
        if squeeze_output:
            cleaned_data = cleaned_data[0]
        
        logger.debug(f"Removed {np.sum(artifact_mask)} artifact samples")
        return cleaned_data, artifact_mask
    
    def _remove_artifacts_amplitude(self, data: np.ndarray, 
                                  threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Remove artifacts using amplitude thresholding."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Calculate amplitude threshold
        amplitude_threshold = threshold * np.std(data)
        artifact_mask = np.abs(data) > amplitude_threshold
        artifact_mask = np.any(artifact_mask, axis=0)
        
        # Remove artifacts
        cleaned_data = data.copy()
        cleaned_data[:, artifact_mask] = np.nan
        
        if squeeze_output:
            cleaned_data = cleaned_data[0]
        
        logger.debug(f"Removed {np.sum(artifact_mask)} artifact samples")
        return cleaned_data, artifact_mask
    
    def extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from data.
        
        Args:
            data: Input signal data
            
        Returns:
            Array of statistical features
        """
        features = []
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        for channel in data:
            # Remove NaN values
            channel_data = channel[~np.isnan(channel)]
            
            if len(channel_data) == 0:
                # If all data is NaN, return zeros
                features.extend([0, 0, 0, 0])
                continue
            
            # Statistical features
            mean_val = np.mean(channel_data)
            var_val = np.var(channel_data)
            skew_val = skew(channel_data)
            kurt_val = kurtosis(channel_data)
            
            features.extend([mean_val, var_val, skew_val, kurt_val])
        
        features = np.array(features)
        
        if squeeze_output:
            features = features.reshape(1, -1)
        
        return features
    
    def extract_spectral_features(self, data: np.ndarray, 
                                sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Extract spectral features from data.
        
        Args:
            data: Input signal data
            sample_rate: Sampling rate (Hz)
            
        Returns:
            Array of spectral features
        """
        sample_rate = sample_rate or self.sample_rate
        features = []
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Define frequency bands
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        for channel in data:
            # Remove NaN values
            channel_data = channel[~np.isnan(channel)]
            
            if len(channel_data) == 0:
                # If all data is NaN, return zeros
                features.extend([0] * len(freq_bands))
                continue
            
            # Calculate power spectral density
            freqs, psd = welch(channel_data, fs=sample_rate, nperseg=min(256, len(channel_data)))
            
            # Calculate power in each frequency band
            band_powers = []
            for band_name, (low_freq, high_freq) in freq_bands.items():
                # Find frequency indices
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapezoid(psd[freq_mask], freqs[freq_mask])
                band_powers.append(band_power)
            
            features.extend(band_powers)
        
        features = np.array(features)
        
        if squeeze_output:
            features = features.reshape(1, -1)
        
        return features
    
    def extract_temporal_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract temporal features from data.
        
        Args:
            data: Input signal data
            
        Returns:
            Array of temporal features
        """
        features = []
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        for channel in data:
            # Remove NaN values
            channel_data = channel[~np.isnan(channel)]
            
            if len(channel_data) == 0:
                # If all data is NaN, return zeros
                features.extend([0, 0, 0])
                continue
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
            zcr = zero_crossings / len(channel_data)
            
            # Peak count
            peaks = np.sum((channel_data[1:-1] > channel_data[:-2]) & 
                          (channel_data[1:-1] > channel_data[2:]))
            peak_count = peaks / len(channel_data)
            
            # RMS (Root Mean Square)
            rms = np.sqrt(np.mean(channel_data**2))
            
            features.extend([zcr, peak_count, rms])
        
        features = np.array(features)
        
        if squeeze_output:
            features = features.reshape(1, -1)
        
        return features
    
    def extract_all_features(self, data: np.ndarray, 
                           sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Extract all available features from data.
        
        Args:
            data: Input signal data
            sample_rate: Sampling rate (Hz)
            
        Returns:
            Array of all features
        """
        sample_rate = sample_rate or self.sample_rate
        
        # Extract different types of features
        statistical_features = self.extract_statistical_features(data)
        spectral_features = self.extract_spectral_features(data, sample_rate)
        temporal_features = self.extract_temporal_features(data)
        
        # Combine all features
        all_features = np.concatenate([
            statistical_features.flatten(),
            spectral_features.flatten(),
            temporal_features.flatten()
        ])
        
        return all_features
    
    def preprocess_pipeline(self, data: np.ndarray, 
                          apply_filters: bool = True,
                          remove_artifacts: bool = True,
                          extract_features: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Input signal data
            apply_filters: Whether to apply filters
            remove_artifacts: Whether to remove artifacts
            extract_features: Whether to extract features
            
        Returns:
            Dictionary with processed data and features
        """
        result = {'original_data': data.copy()}
        
        # Apply filters
        if apply_filters:
            # Bandpass filter
            filtered_data = self.apply_bandpass_filter(data)
            # Notch filter
            filtered_data = self.apply_notch_filter(filtered_data)
            result['filtered_data'] = filtered_data
        else:
            result['filtered_data'] = data.copy()
        
        # Remove artifacts
        if remove_artifacts:
            cleaned_data, artifact_mask = self.remove_artifacts(result['filtered_data'])
            result['cleaned_data'] = cleaned_data
            result['artifact_mask'] = artifact_mask
        else:
            result['cleaned_data'] = result['filtered_data']
            result['artifact_mask'] = np.zeros(data.shape[-1], dtype=bool)
        
        # Extract features
        if extract_features:
            features = self.extract_all_features(result['cleaned_data'])
            result['features'] = features
        
        return result
