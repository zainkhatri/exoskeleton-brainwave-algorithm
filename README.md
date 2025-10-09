# Brainwave Analysis

A comprehensive Python toolkit for EEG/EOG data analysis and gaze direction prediction using machine learning.

## Features

- **Multi-format Data Loading**: Support for FIF, EDF, BDF, EEGLAB, and BrainVision formats
- **Advanced Signal Processing**: Bandpass filtering, artifact removal, and feature extraction
- **Machine Learning Pipeline**: Multiple model types (Random Forest, SVM, Neural Networks)
- **Comprehensive Visualization**: Topographic maps, power spectra, and analysis reports
- **Real-time Capabilities**: Stream processing for live EEG data
- **Modular Architecture**: Clean, extensible codebase with proper separation of concerns

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/brainwave-analysis.git
cd brainwave-analysis

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from brainwave_analysis import BrainwaveAnalysisPipeline

# Initialize pipeline
pipeline = BrainwaveAnalysisPipeline()

# Run complete analysis with sample data
results = pipeline.run_full_pipeline(
    use_sample_data=True,
    hyperparameter_tuning=False,
    create_plots=True
)

# Print results
print(f"Accuracy: {results['final_results']['accuracy']:.3f}")
```

### Using Real Data

```python
# Load your own EEG/EOG data
results = pipeline.run_full_pipeline(
    eeg_file="path/to/your/eeg_data.fif",
    eog_file="path/to/your/eog_data.fif",
    use_sample_data=False,
    hyperparameter_tuning=True
)
```

## Project Structure

```
brainwave_analysis/
├── src/
│   ├── data/           # Data loading and handling
│   ├── preprocessing/  # Signal processing and filtering
│   ├── ml/            # Machine learning models
│   ├── visualization/ # Plotting and visualization
│   ├── utils/         # Configuration and utilities
│   └── pipeline.py    # Main analysis pipeline
├── configs/           # Configuration files
├── examples/          # Example scripts
├── tests/            # Unit tests
├── data/             # Data storage
├── output/           # Analysis results
└── plots/            # Generated plots
```

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# configs/default_config.yaml
data:
  sample_rate: 256
  epoch_duration: 1.0
  overlap: 0.5

filtering:
  low_cutoff: 1.0
  high_cutoff: 50.0
  filter_order: 5

ml:
  model_type: "random_forest"
  test_size: 0.2
  cross_validation_folds: 5
```

## Examples

### Example 1: Basic Analysis

```python
from brainwave_analysis import BrainwaveAnalysisPipeline

# Create pipeline
pipeline = BrainwaveAnalysisPipeline()

# Load and preprocess data
data_info = pipeline.load_data(use_sample_data=True)
preprocessing_results = pipeline.preprocess_data(data_info)

# Train model
training_results = pipeline.train_model(preprocessing_results)

# Evaluate
evaluation_results = pipeline.evaluate_model()

# Create visualizations
plots = pipeline.create_visualizations()
```

### Example 2: Custom Configuration

```python
from brainwave_analysis import BrainwaveAnalysisPipeline

# Use custom configuration
pipeline = BrainwaveAnalysisPipeline("configs/my_config.yaml")

# Run with hyperparameter tuning
results = pipeline.run_full_pipeline(
    use_sample_data=True,
    hyperparameter_tuning=True
)
```

### Example 3: Real-time Prediction

```python
import numpy as np
from brainwave_analysis import BrainwaveAnalysisPipeline

# Initialize and train pipeline
pipeline = BrainwaveAnalysisPipeline()
pipeline.run_full_pipeline(use_sample_data=True)

# Make predictions on new data
new_eeg_data = np.random.randn(64, 256)  # 64 channels, 256 samples
new_eog_data = np.random.randn(2, 256)   # 2 EOG channels

prediction = pipeline.predict(new_eeg_data, new_eog_data)
print(f"Gaze direction: {prediction['gaze_direction']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

## Visualization

The package provides comprehensive visualization capabilities:

- **Raw Data Plots**: Time series visualization of EEG/EOG signals
- **Topographic Maps**: Brain activity maps over time
- **Power Spectra**: Frequency domain analysis
- **Feature Importance**: Model interpretability
- **Confusion Matrices**: Model performance evaluation
- **Summary Reports**: Comprehensive analysis overview

## Scientific Background

This toolkit implements state-of-the-art methods for:

- **Signal Preprocessing**: Butterworth filtering, artifact removal, feature extraction
- **Feature Engineering**: Statistical, spectral, and temporal features
- **Machine Learning**: Multiple algorithms with cross-validation
- **Gaze Direction Prediction**: Binary classification (left vs. right)

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_pipeline.py -v
```

## Performance

The system is optimized for:

- **Efficiency**: Vectorized operations with NumPy/SciPy
- **Scalability**: Handles large datasets with memory-efficient processing
- **Accuracy**: Cross-validated models with proper evaluation metrics
- **Reproducibility**: Fixed random seeds and version control

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MNE-Python](https://mne.tools/) for EEG/MEG data handling
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Contact the development team
- Check the documentation and examples

## Future Roadmap

- [ ] Deep learning models (CNN, LSTM)
- [ ] Real-time streaming interface
- [ ] Web-based visualization dashboard
- [ ] Support for more EEG file formats
- [ ] Advanced artifact removal algorithms
- [ ] Multi-class gaze direction prediction
- [ ] Integration with popular EEG hardware

---

**Made for the neuroscience community**
