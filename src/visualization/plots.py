"""
Visualization utilities for EEG/EOG data and analysis results.
Provides comprehensive plotting capabilities for brainwave analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.viz import plot_topomap, plot_compare_evokeds
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BrainwaveVisualizer:
    """Handles visualization of EEG/EOG data and analysis results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.viz_config = self.config.get('visualization', {})
        self.save_plots = self.viz_config.get('save_plots', True)
        self.plot_format = self.viz_config.get('plot_format', 'png')
        self.dpi = self.viz_config.get('dpi', 300)
        self.show_plots = self.viz_config.get('show_plots', False)
        
        # Create plots directory if saving
        if self.save_plots:
            plots_dir = Path(self.config.get('paths', {}).get('plots_dir', 'plots/'))
            plots_dir.mkdir(parents=True, exist_ok=True)
            self.plots_dir = plots_dir
    
    def plot_raw_data(self, raw: mne.io.Raw, 
                     channels: Optional[List[str]] = None,
                     duration: Optional[float] = None,
                     title: str = "Raw EEG Data",
                     save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot raw EEG/EOG data.
        
        Args:
            raw: MNE Raw object
            channels: Specific channels to plot
            duration: Duration to plot (seconds)
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        # Select channels
        if channels:
            picks = mne.pick_channels(raw.ch_names, channels)
        else:
            picks = mne.pick_types(raw.info, eeg=True, eog=True)

        # raw.plot() opens its own figure, not the one we created above
        if duration:
            fig = raw.plot(duration=duration, picks=picks, show=False, title=title)
        else:
            fig = raw.plot(picks=picks, show=False, title=title)

        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_epochs(self, epochs: mne.Epochs,
                   condition: Optional[str] = None,
                   title: str = "EEG Epochs",
                   save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot EEG epochs.
        
        Args:
            epochs: MNE Epochs object
            condition: Specific condition to plot
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        # epochs.plot() opens its own figure, not one we create beforehand
        if condition:
            fig = epochs[condition].plot(show=False, title=f"{title} - {condition}")
        else:
            fig = epochs.plot(show=False, title=title)

        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_evoked_comparison(self, evoked_list: List[mne.Evoked],
                             conditions: List[str],
                             title: str = "Evoked Response Comparison",
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of evoked responses.
        
        Args:
            evoked_list: List of MNE Evoked objects
            conditions: List of condition names
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create condition mapping
        condition_map = dict(zip(conditions, evoked_list))
        
        # Plot comparison
        plot_compare_evokeds(condition_map, picks='eeg', show=False, title=title)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_topography(self, data: np.ndarray,
                       info: mne.Info,
                       times: Optional[np.ndarray] = None,
                       title: str = "Topographic Map",
                       save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot topographic maps.
        
        Args:
            data: Data array (channels x time)
            info: MNE Info object
            times: Time points to plot
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if times is None:
            times = np.linspace(0, data.shape[1] - 1, 5)
        
        # Plot topomap
        plot_topomap(data, info, times=times, show=False, title=title)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_power_spectrum(self, data: np.ndarray,
                           sample_rate: float,
                           channels: Optional[List[str]] = None,
                           title: str = "Power Spectrum",
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot power spectrum of data.
        
        Args:
            data: Data array (channels x time)
            sample_rate: Sampling rate (Hz)
            channels: Channel names
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Calculate power spectral density
        freqs, psd = mne.time_frequency.psd_array_multitaper(
            data, sfreq=sample_rate, fmin=1, fmax=50
        )
        
        # Plot for each channel (up to 4)
        n_channels = min(data.shape[0], 4)
        for i in range(n_channels):
            axes[i].semilogy(freqs, psd[i])
            axes[i].set_xlabel('Frequency (Hz)')
            axes[i].set_ylabel('Power (V²/Hz)')
            if channels:
                axes[i].set_title(f'Channel {channels[i]}')
            else:
                axes[i].set_title(f'Channel {i}')
            axes[i].grid(True)
        
        # Hide unused subplots
        for i in range(n_channels, 4):
            axes[i].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_feature_importance(self, feature_importance: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               top_n: int = 20,
                               title: str = "Feature Importance",
                               save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Feature importance values
            feature_names: Names of features
            top_n: Number of top features to show
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-top_n:]
        top_importance = feature_importance[top_indices]
        
        if feature_names:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f'Feature {i}' for i in top_indices]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_importance)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                             class_names: List[str] = None,
                             title: str = "Confusion Matrix",
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Names of classes
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(confusion_matrix.shape[0])]
        
        # Plot heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_training_history(self, history: Dict[str, List[float]],
                             title: str = "Training History",
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot training history (for neural networks).
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Model Loss')
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot accuracy
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history:
                axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_gaze_direction_comparison(self, left_data: np.ndarray,
                                     right_data: np.ndarray,
                                     sample_rate: float,
                                     title: str = "Gaze Direction Comparison",
                                     save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between left and right gaze directions.
        
        Args:
            left_data: Data for left gaze
            right_data: Data for right gaze
            sample_rate: Sampling rate (Hz)
            title: Plot title
            save_name: Name for saving plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time axis
        time_left = np.arange(left_data.shape[1]) / sample_rate
        time_right = np.arange(right_data.shape[1]) / sample_rate
        
        # Plot average signals
        axes[0, 0].plot(time_left, np.mean(left_data, axis=0), label='Left Gaze', alpha=0.7)
        axes[0, 0].plot(time_right, np.mean(right_data, axis=0), label='Right Gaze', alpha=0.7)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Average Signal')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot power spectrum comparison
        freqs_left, psd_left = mne.time_frequency.psd_array_multitaper(
            left_data, sfreq=sample_rate, fmin=1, fmax=50
        )
        freqs_right, psd_right = mne.time_frequency.psd_array_multitaper(
            right_data, sfreq=sample_rate, fmin=1, fmax=50
        )
        
        axes[0, 1].semilogy(freqs_left, np.mean(psd_left, axis=0), label='Left Gaze')
        axes[0, 1].semilogy(freqs_right, np.mean(psd_right, axis=0), label='Right Gaze')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power (V²/Hz)')
        axes[0, 1].set_title('Power Spectrum')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot difference
        diff_signal = np.mean(right_data, axis=0) - np.mean(left_data, axis=0)
        axes[1, 0].plot(time_left, diff_signal)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude Difference')
        axes[1, 0].set_title('Right - Left Difference')
        axes[1, 0].grid(True)
        
        # Plot frequency band power comparison
        freq_bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 
                     'Beta': (13, 30), 'Gamma': (30, 50)}
        
        band_powers_left = []
        band_powers_right = []
        band_names = []
        
        for band_name, (low, high) in freq_bands.items():
            # Find frequency indices
            freq_mask = (freqs_left >= low) & (freqs_left <= high)
            power_left = np.mean(np.trapz(psd_left[:, freq_mask], freqs_left[freq_mask], axis=1))
            power_right = np.mean(np.trapz(psd_right[:, freq_mask], freqs_right[freq_mask], axis=1))
            
            band_powers_left.append(power_left)
            band_powers_right.append(power_right)
            band_names.append(band_name)
        
        x = np.arange(len(band_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, band_powers_left, width, label='Left Gaze', alpha=0.7)
        axes[1, 1].bar(x + width/2, band_powers_right, width, label='Right Gaze', alpha=0.7)
        axes[1, 1].set_xlabel('Frequency Band')
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].set_title('Band Power Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(band_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if self.save_plots and save_name:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def _save_plot(self, fig: plt.Figure, save_name: str):
        """Save plot to file."""
        if not save_name.endswith(f'.{self.plot_format}'):
            save_name += f'.{self.plot_format}'
        
        save_path = self.plots_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    def create_summary_report(self, results: Dict[str, Any],
                            save_name: str = "analysis_summary") -> plt.Figure:
        """
        Create a comprehensive summary report.
        
        Args:
            results: Analysis results dictionary
            save_name: Name for saving report
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Model Performance
        ax1 = fig.add_subplot(gs[0, 0])
        if 'accuracy' in results:
            ax1.bar(['Test Accuracy'], [results['accuracy']])
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Performance')
            ax1.set_ylim(0, 1)
        
        # Plot 2: Feature Importance
        ax2 = fig.add_subplot(gs[0, 1:])
        if 'feature_importance' in results:
            importance = results['feature_importance']
            top_indices = np.argsort(importance)[-10:]
            ax2.barh(range(len(top_indices)), importance[top_indices])
            ax2.set_yticks(range(len(top_indices)))
            ax2.set_yticklabels([f'Feature {i}' for i in top_indices])
            ax2.set_xlabel('Importance')
            ax2.set_title('Top 10 Feature Importance')
        
        # Plot 3: Confusion Matrix
        ax3 = fig.add_subplot(gs[1, 0])
        if 'confusion_matrix' in results:
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=ax3)
            ax3.set_title('Confusion Matrix')
        
        # Plot 4: Cross-validation scores
        ax4 = fig.add_subplot(gs[1, 1])
        if 'cv_scores' in results:
            ax4.boxplot(results['cv_scores'])
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Cross-validation Scores')
        
        # Plot 5: Training history (if available)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'training_history' in results:
            history = results['training_history']
            if 'loss' in history:
                ax5.plot(history['loss'], label='Loss')
                ax5.set_xlabel('Epoch')
                ax5.set_ylabel('Loss')
                ax5.set_title('Training Loss')
                ax5.legend()
        
        # Add text summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        summary_text = f"""
        Analysis Summary:
        • Model Type: {results.get('model_type', 'Unknown')}
        • Test Accuracy: {results.get('accuracy', 0):.3f}
        • Cross-validation Mean: {results.get('cv_mean', 0):.3f} ± {results.get('cv_std', 0):.3f}
        • Number of Features: {results.get('n_features', 0)}
        • Number of Samples: {results.get('n_samples', 0)}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle('Brainwave Analysis Summary Report', fontsize=16)
        
        if self.save_plots:
            self._save_plot(fig, save_name)
        
        if self.show_plots:
            plt.show()
        
        return fig
