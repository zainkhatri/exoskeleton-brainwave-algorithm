# Brainwave Analysis

Research code from my junior-year work with the Meki lab at UC Berkeley, predicting gaze
direction (left vs. right) from EEG/EOG signals for exoskeleton control.

The pipeline loads EEG/EOG data (FIF, EDF, BDF, EEGLAB, BrainVision), runs it through
Butterworth bandpass filtering and artifact removal, extracts statistical/spectral/temporal
features, and trains a classifier (Random Forest, SVM, or a small neural net, picked via
config) to predict gaze direction. There's also a real-time prediction path for new data and
a set of plots (topographic maps, power spectra, confusion matrix, feature importance) to
actually see what the model's doing instead of just trusting an accuracy number.

## Install

```bash
git clone https://github.com/zainkhatri/exoskeleton-brainwave-algorithm.git
cd exoskeleton-brainwave-algorithm
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from brainwave_analysis import BrainwaveAnalysisPipeline

pipeline = BrainwaveAnalysisPipeline()
results = pipeline.run_full_pipeline(use_sample_data=True, create_plots=True)
print(f"Accuracy: {results['final_results']['accuracy']:.3f}")
```

Against real data instead of the MNE sample set:

```python
results = pipeline.run_full_pipeline(
    eeg_file="path/to/eeg_data.fif",
    eog_file="path/to/eog_data.fif",
    use_sample_data=False,
    hyperparameter_tuning=True,
)
```

Predicting on new data after training:

```python
prediction = pipeline.predict(new_eeg_data, new_eog_data)
print(prediction["gaze_direction"], prediction["confidence"])
```

## Layout

```
src/
├── data/           # loading FIF/EDF/BDF/EEGLAB/BrainVision
├── preprocessing/  # filtering, artifact removal, feature extraction
├── ml/             # GazeDirectionClassifier, EnsembleClassifier
├── visualization/  # plots
├── utils/          # config
└── pipeline.py     # ties it together
configs/            # YAML config (sample rate, filter cutoffs, model type, CV folds)
examples/
tests/
```

## Tests

```bash
pytest tests/
```
