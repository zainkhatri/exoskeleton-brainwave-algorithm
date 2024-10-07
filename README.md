# Brain Waves Analysis

**Brain Waves Analysis** is a Python-based program designed to analyze brain wave data (EEG and EOG signals) to determine the direction a person is looking (left or right). This program utilizes several powerful libraries for EEG data handling, visualization, and machine learning to create an interactive tool for neuroimaging analysis and real-time applications in research.

## Features

- **EEG & EOG Data Processing**: Handles both EEG and EOG data, offering functions to load and visualize these signals.
- **Machine Learning Integration**: Uses Random Forest classifiers to predict gaze direction based on EEG and EOG data features.
- **Real-Time Gaze Detection**: Capable of classifying new EEG data and triggering actions based on gaze direction (left or right).
- **Visualization Tools**: Provides multiple visual representations of brain wave data, including scalp topography and 3D field maps.
- **Data Cleaning and Feature Extraction**: Applies band-pass filtering to remove noise and extracts statistical features like mean, variance, skewness, and kurtosis.
- **Customizable for Specific Hardware**: Can be modified to suit specific hardware and file paths for real data input.

## Installation

To get started, you'll need to install a few libraries:

```bash
pip install mne
pip install pyvista
pip install vtk