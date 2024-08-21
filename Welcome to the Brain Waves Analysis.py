# Welcome to the Brain Waves Analysis. Feel free to change up or edit the code to fit specific hardware requirements.
# Zain - 7/20/2024

# Before we start, install the libraries in your local environment:
# pip install mne
#   * It's an open-source package used for EEG data. It uses functions to handle neuroimaging.
# pip install pyvista
#   * A plotting and 3D visualization library to create 3D plots for neuroimaging data.
# pip install vtk
#   * A visualization toolkit for image processing and visualization.

# This code helps us figure out which way someone is looking (left or right) by analyzing brain wave data (EEG and EOG signals).
# It starts by telling you to install some necessary libraries for handling and visualizing this data.
# There are functions to load sample data (just for practice) and real brain wave data when we have the file paths.
# If something goes wrong while loading the data, the code will let you know.
# We also added some cool visualizations to see the brain wave data in different ways, like maps showing activity on the scalp.
# There's a part of the code that cleans up the data (filters out noise) and calculates some basic stats (like average and variance).
# Then, we use these stats to train a machine learning model (a Random Forest classifier) that can predict if someone is looking left or right.
# Finally, the system can watch for these gaze directions and trigger actions based on what it detects, which is handy for real-time applications in research.

#   Now we start. We import the libraries we need.

import os
import numpy as np
import mne
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Function to load brain wave data
def load_brain_wave_data(
    sample_data_folder=None, sample_data_file_name="sample_audvis-ave.fif"
):
    try:
        if sample_data_folder is None:
            sample_data_folder = (
                mne.datasets.sample.data_path()
            )  # Get the path to the sample data.
        sample_data_file = os.path.join(
            sample_data_folder, "MEG", "sample", sample_data_file_name
        )  # Get the path to a specific data file.
        brain_wave_data = mne.read_evokeds(
            sample_data_file, baseline=(None, 0), proj=True, verbose=False
        )  # Read the data with some settings.
        return brain_wave_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


# Function to load real EEG and EOG data
def load_real_data(eeg_file, eog_file):
    try:
        eeg_data = mne.io.read_raw_fif(eeg_file, preload=True)
        eog_data = mne.io.read_raw_fif(eog_file, preload=True)
        return eeg_data, eog_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


brain_wave_data = load_brain_wave_data()

# Print the type of data we have and check if the settings were applied correctly.
for condition in brain_wave_data:
    print(
        f"Condition: {condition.comment}, baseline: {condition.baseline}"
    )  # Print each condition and its settings.

# Function to visualize brain wave data
def visualize_brain_wave_data(data):
    # We define the types of conditions we have in our data.
    conditions = ("auditoryLeft", "auditoryRight", "visualLeft", "visualRight")
    condition_data_map = dict(
        zip(conditions, data)
    )  # Map these conditions to our data.

    # Plot the brain wave data for one of the conditions.
    condition_data_map["auditoryLeft"].plot(
        exclude=[]
    )  # Plot the data for the 'auditoryLeft' condition.

    # Plot the brain wave data with different colors to show different sensors.
    condition_data_map["auditoryLeft"].plot(
        picks="mag", spatial_colors=True, gfp=True
    )  # Plot the magnetometer data.

    # Define specific times to plot the brain wave data on the scalp.
    time_points = np.linspace(0.05, 0.13, 5)  # Create a list of times.

    # Plot the brain wave data on the scalp for these times.
    condition_data_map["auditoryLeft"].plot_topomap(
        ch_type="mag", times=time_points, colorbar=True
    )  # Plot topographies for the magnetometer data.

    return condition_data_map


condition_data_map = visualize_brain_wave_data(brain_wave_data)

# We are getting somewhere now. We now calculate and plot the average mean of EEG.
# A custom function to get the maximum value.
def calculate_max(data):
    return data.max(axis=1)


# Compare different ways to combine the brain wave data.
for combine_method in ("mean", "median", "gfp", calculate_max):
    mne.viz.plot_compare_evokeds(
        condition_data_map, picks="eeg", combine=combine_method
    )  # Plot EEG data with different combining methods.

# Plotting the MEG data.
# Plot the brain wave data with different colors and line styles.
mne.viz.plot_compare_evokeds(
    condition_data_map,
    picks="MEG 1811",
    colors=dict(auditory=0, visual=1),
    linestyles=dict(left="solid", right="dashed"),
)

# Create a list of modified brain wave data for comparison.
modified_data_list = list()

# We have to iterate over a list of comments, starting the index from 1.
for index, comment in enumerate(
    ("example1", "example2", "", None, "example3"), start=1
):
    modified_data = brain_wave_data[0].copy()
    modified_data.comment = comment
    modified_data.data *= index  # Change the data so we can tell the traces apart.
    modified_data_list.append(modified_data)

# Plot the modified brain wave data for comparison.
# Hopefully, this will plot the modified data, which allows us to compare the different modifications with a visual.
mne.viz.plot_compare_evokeds(modified_data_list, picks="mag")

# Plot an image of the brain wave data for the 'visualRight' condition.
# This plots the 'visualRight' condition data as an image, which shows the data's structure over time.
condition_data_map["visualRight"].plot_image(picks="meg")

# Plot the brain wave data with topographic axes.
# This plots the EEG data, distinguishing between auditory and visual conditions using different colors and line styles.
mne.viz.plot_compare_evokeds(
    condition_data_map,
    picks="eeg",
    colors=dict(auditory=0, visual=1),
    linestyles=dict(left="solid", right="dashed"),
    axes="topo",
    styles=dict(auditory=dict(linewidth=1), visual=dict(linewidth=1)),
)

# Define paths for subjects directory and transformation file.
def get_paths(sample_data_file=None):
    if sample_data_file is None:
        sample_data_file = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(sample_data_file, "subjects")
    transformation_file = os.path.join(
        sample_data_file, "MEG", "sample", "sample_audvis_raw-trans.fif"
    )
    return subjects_dir, transformation_file


subjects_dir, transformation_file = get_paths()

# Plotting 3D Field Maps.
# Create field maps for the 'auditoryLeft' condition.
field_maps = mne.make_field_map(
    condition_data_map["auditoryLeft"],
    trans=transformation_file,
    subject="sample",
    subjects_dir=subjects_dir,
)

# Plot the field maps at a specific time point.
# This plots 3D field maps for the 'auditoryLeft' condition, providing a 3D visual of the brain's magnetic field.
condition_data_map["auditoryLeft"].plot_field(
    field_maps, time=0.1
)  # Plot 3D field maps for the 'auditoryLeft' condition.

# Function to apply a band-pass filter to the data
def apply_bandpass_filter(
    data, low_cutoff=1, high_cutoff=50, sample_rate=256, filter_order=5
):
    nyquist_frequency = 0.5 * sample_rate  # Calculate the Nyquist frequency
    low = low_cutoff / nyquist_frequency  # Normalize the low cutoff frequency
    high = high_cutoff / nyquist_frequency  # Normalize the high cutoff frequency
    b, a = butter(
        filter_order, [low, high], btype="band"
    )  # Design a Butterworth band-pass filter
    filtered_data = lfilter(b, a, data)  # Apply the filter to the data
    return filtered_data


# Function to extract features from the EEG data
def extract_features_from_eeg(eeg_data):
    feature_list = []  # Initialize a list to store the features
    for epoch in eeg_data:  # Loop through each epoch of EEG data
        features = [
            np.mean(epoch),  # Calculate the mean of the epoch
            np.var(epoch),  # Calculate the variance of the epoch
            skew(epoch),  # Calculate the skewness of the epoch
            kurtosis(epoch),  # Calculate the kurtosis of the epoch
        ]
        feature_list.append(features)  # Add the features to the list
    return np.array(feature_list)  # Convert the list of features to a numpy array


# Replace the simulated EEG data section with real data processing

# Paths to the real EEG and EOG data files
eeg_file = "path_to_real_eeg_data.fif"  # Replace these two with the real path to the EEG data file
eog_file = "path_to_real_eog_data.fif"

# Load real EEG and EOG data
real_eeg_data, real_eog_data = load_real_data(eeg_file, eog_file)
if real_eeg_data is not None and real_eog_data is not None:
    # Preprocess the real EEG data into epochs
    real_eeg_epochs = mne.make_fixed_length_epochs(
        real_eeg_data, duration=1.0, preload=True
    )
    real_eog_epochs = mne.make_fixed_length_epochs(
        real_eog_data, duration=1.0, preload=True
    )

    # Extract data arrays from the epochs
    real_eeg_data = real_eeg_epochs.get_data()
    real_eog_data = real_eog_epochs.get_data()

    # Apply band-pass filter to EEG and EOG data
    filtered_eeg_data = [apply_bandpass_filter(epoch) for epoch in real_eeg_data]
    filtered_eog_data = [apply_bandpass_filter(epoch) for epoch in real_eog_data]

    # Extract features from both EEG and EOG data
    eeg_features = extract_features_from_eeg(filtered_eeg_data)
    eog_features = extract_features_from_eeg(filtered_eog_data)

    # Combine features from EEG and EOG
    combined_features = np.hstack((eeg_features, eog_features))

    # Simulated labels for the purpose of demonstration (0 for left, 1 for right)
    # Replace these with actual labels corresponding to your real data
    labels = np.random.randint(0, 2, size=(combined_features.shape[0],))

    # Split the data into training and testing sets
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        combined_features, labels, test_size=0.2, random_state=42
    )

    # Initialize a Random Forest classifier with 100 trees
    classifier = RandomForestClassifier(n_estimators=100)

    # Train the classifier on the training data
    classifier.fit(training_data, training_labels)

    # Evaluate the classifier on the testing data and print the accuracy
    accuracy = classifier.score(testing_data, testing_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Function to classify new EEG data
    def classify_new_eeg_data(
        new_eeg_data,
        new_eog_data,
        low_cutoff=1,
        high_cutoff=50,
        sample_rate=256,
        filter_order=5,
    ):
        filtered_eeg_data = apply_bandpass_filter(
            new_eeg_data, low_cutoff, high_cutoff, sample_rate, filter_order
        )
        filtered_eog_data = apply_bandpass_filter(
            new_eog_data, low_cutoff, high_cutoff, sample_rate, filter_order
        )
        eeg_features = extract_features_from_eeg([filtered_eeg_data])
        eog_features = extract_features_from_eeg([filtered_eog_data])
        combined_features = np.hstack((eeg_features, eog_features))
        prediction = classifier.predict(combined_features)
        return prediction[0]

    # Example usage of classify_new_eeg_data function
    new_eeg_data = np.random.rand(256)  # Replace with new real EEG data
    new_eog_data = np.random.rand(256)  # Replace with new real EOG data
    classification = classify_new_eeg_data(new_eeg_data, new_eog_data)
    print(f"Classification: {'Left' if classification == 0 else 'Right'}")

    # Visualization function for gaze direction
    def visualize_gaze_direction(eeg_epochs, title):
        evoked_left = eeg_epochs["Left"].average()
        evoked_right = eeg_epochs["Right"].average()
        mne.viz.plot_compare_evokeds(
            [evoked_left, evoked_right], picks="eeg", title=title
        )

    # Assuming the epochs are labeled with conditions 'Left' and 'Right'
    visualize_gaze_direction(real_eeg_epochs, "Gaze Direction: Left vs Right")
else:
    print("Failed to load real EEG and EOG data.")

# CHANGE THE PATHS ON LINE 169 TO THE ACTUAL FILE PATHS WHERE THE EEG AND EOG FILES ARE AT

# Shoutout https://mne.tools/dev/auto_tutorials/intro/10_overview.html
# It was an overview of MEG/EEG analysis with MNE-Python

# Shoutout https://reybahl.medium.com/eeg-signal-analysis-with-python-fdd8b4cbd306
# EEG Signal Analysis in Python

# Didn't help that much but still shoutout : https://purl.stanford.edu/tc919dd5388
# Stanford's Digital Repository for their object category EEG Dataset

# It also prints out a couple of graphs and I'll explain what they do:

# 1. EEG and Gradiometers, Magnetometers, and EOG Graphs
#   * Set of graphs that display data from EEG sensors, gradiometers, magnetometers, and EOG sensors
#   - EEG Graph : shows the electrical activity of the brain
#   - Gradiometers Graph : shows the magnetic fields produced by brain activity
#   - Magnetometers Graph: measuring the gradient field and absolute magnetic field
#   - EOG Graph: displays electrical potential by eye movements

# 2. Magnetometers Graph
#   * Specific graph for the data from the magnetometer sensors to visualize the magnetic fields and show the intensity of magnetic activity

# 3. EEG Mean Graph with Brain Wave Detection
#   * This graph shows mean of EEG data with brain wave detection graph to show average electrical activity recorded by EEG sensors AND highlight specific brain wave patterns

# 4. EEG Median Graph
#   * This graph shows the median of the EEG data to visualize the central tendency of EEG data, highlight the outliers

# 5. EEG Global Field Power Graph
#   * Graph shows the Global Field Power of the EEG data to measure the overall strength of the brain's electrical field at any time

# 6. Extra EEG Graph
#   * This graph is included to provide additional insights if necessary