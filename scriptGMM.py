# --------------------------
# --- Import Libraries -----
# --------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

# ------------------------------------------------------
# --- Function to Load and Filter Patient Movement Data
# ------------------------------------------------------
def load_patient_data(preprocessed_dir='../preprocessed/', channel_filter='Relaxed1_Acceleration'):
    """
    Load accelerometer data for all patients, filter by specific movement channels.

    Args:
        preprocessed_dir (str): Directory containing preprocessed movement data.
        channel_filter (str): String to filter specific channels (e.g., Relaxed1_Acceleration).

    Returns:
        list: Filtered movement data (or None) for each patient.
    """
    # Generate all possible channel names (task + sensor + location + axis)
    channels = []
    for task in ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", "HoldWeight",
                 "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]:
        for device_location in ["LeftWrist", "RightWrist"]:
            for sensor in ["Acceleration", "Rotation"]:
                for axis in ["X", "Y", "Z"]:
                    channels.append(f"{task}_{sensor}_{device_location}_{axis}")

    # Get indices of channels that match the specified filter
    channel_indices = [i for i, ch in enumerate(channels) if channel_filter in ch]

    # Load the CSV file listing all patient file IDs
    df = pd.read_csv(f'{preprocessed_dir}file_list.csv')

    all_patient_data = []  # Store processed data for each patient

    for i, row in df.iterrows():
        file_idx = row['id']
        file_path = f"{preprocessed_dir}/movement/{file_idx:03d}_ml.bin"

        if not os.path.exists(file_path):
            print(f"Warning: File not found for subject {file_idx}. Skipping.")
            all_patient_data.append(None)
            continue

        try:
            # Read the binary file and reshape based on expected channels
            x = np.fromfile(file_path, dtype=np.float32)

            if x.size % len(channels) != 0:
                print(f"Warning: Data size mismatch for subject {file_idx}. Skipping.")
                all_patient_data.append(None)
                continue

            x = x.reshape((len(channels), -1))
            x_filtered = x[channel_indices, :]
            all_patient_data.append(x_filtered)

        except Exception as e:
            print(f"Error processing subject {file_idx}: {e}")
            all_patient_data.append(None)

    print(f"Loaded data for {len([x for x in all_patient_data if x is not None])} subjects successfully.")
    return all_patient_data

# ----------------------------------------
# --- Signal Processing Helper Functions
# ----------------------------------------

def bandpass_filter(data, fs=100, lowcut=3, highcut=12, order=6):
    """Apply a Butterworth bandpass filter to retain frequencies between 3–12 Hz (typical tremor band)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def compute_psd_features(data, fs=100):
    """Calculate peak power and area under the PSD curve within the 3–12 Hz tremor frequency band."""
    freqs, psd = welch(data, fs=fs, nperseg=fs, noverlap=fs // 2)
    band_mask = (freqs >= 3) & (freqs <= 12)
    psd_band = psd[band_mask]
    freqs_band = freqs[band_mask]

    peak_power = np.max(psd_band) if len(psd_band) else 0.0
    auc_power = np.trapezoid(psd_band, freqs_band)
    return peak_power, auc_power

def compute_envelope(data):
    """Estimate mean envelope of the signal by detecting peaks and troughs."""
    peaks, _ = find_peaks(data)
    troughs, _ = find_peaks(-data)
    points = np.sort(np.concatenate([peaks, troughs]))
    if len(points) < 2:
        return 0.0
    envelope = np.abs(data[points])
    return np.mean(envelope)

def compute_mean_acceleration(data):
    """Compute the average absolute acceleration (magnitude of motion)."""
    return np.mean(np.abs(data))

# ----------------------------------------
# --- Feature Extraction Per Subject
# ----------------------------------------

def extract_wrist_features(wrist_xyz, fs=100):
    """Compute 4 tremor-related features from 3-axis wrist data."""
    mag = np.sqrt(np.sum(wrist_xyz ** 2, axis=0))  # Combine XYZ into a single vector magnitude
    filtered = bandpass_filter(mag, fs=fs)

    peak_power, auc_power = compute_psd_features(filtered, fs)
    mean_env = compute_envelope(filtered)
    mean_acc = compute_mean_acceleration(filtered)

    return [mean_acc, mean_env, peak_power, auc_power]

# ----------------------------------------
# --- Main Feature Computation Loop
# ----------------------------------------

# Load filtered accelerometer data
participant_data = load_patient_data(preprocessed_dir='../preprocessed/', channel_filter='Relaxed1_Acceleration')

features = []  # List to hold feature vectors

for subject in participant_data:
    if subject is None:
        features.append([np.nan] * 8)
        continue

    left_xyz = subject[0:3, :]   # X, Y, Z from Left wrist
    right_xyz = subject[3:6, :]  # X, Y, Z from Right wrist

    left_feats = extract_wrist_features(left_xyz)
    right_feats = extract_wrist_features(right_xyz)

    features.append(left_feats + right_feats)

# ----------------------------------------
# --- Unsupervised Learning (GMM)
# ----------------------------------------

X = np.array(features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use BIC to find optimal number of GMM clusters
lowest_bic = np.inf
best_n = 1
bic = []

for n in range(1, 10):
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    current_bic = gmm.bic(X_scaled)
    bic.append(current_bic)
    if current_bic < lowest_bic:
        lowest_bic = current_bic
        best_n = n

# Fit GMM with best number of clusters
gmm = GaussianMixture(n_components=best_n, random_state=42)
clusters = gmm.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, clusters)
calinski = calinski_harabasz_score(X_scaled, clusters)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Score: {calinski:.2f}")

# ----------------------------------------
# --- t-SNE Visualization by Cluster
# ----------------------------------------

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='plasma', alpha=0.8)
plt.title("Unsupervised Tremor Profiling (Left + Right Wrist)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="GMM Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------
# --- Optional: Export Clustered Features
# ----------------------------------------

feature_names = [
    "Left_MeanAcc", "Left_Envelope", "Left_PeakPower", "Left_AUC",
    "Right_MeanAcc", "Right_Envelope", "Right_PeakPower", "Right_AUC"
]

df_out = pd.DataFrame(X, columns=feature_names)
df_out["Cluster"] = clusters
# df_out.to_csv("tremor_clusters.csv", index=False)  # Uncomment to save as CSV

# ----------------------------------------
# --- t-SNE Colored by Patient Condition
# ----------------------------------------

# Load patient conditions
file_list_path = '../preprocessed/file_list.csv'
df = pd.read_csv(file_list_path)
patientData = df['condition']

# Map to numeric values
patientData = patientData.replace({
    'Healthy': 0,
    "Parkinson's": 1
})
patientData = patientData.apply(lambda x: x if x in [0, 1] else 2)  # Encode all others as 2
df['condition_encoded'] = patientData  # Save encoded condition

color_rule = patientData.values

# Run new t-SNE for visualization
tsne_new = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne_new = tsne_new.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne_new[:, 0], X_tsne_new[:, 1], c=color_rule, cmap='viridis', alpha=0.8)
plt.title("New t-SNE Visualization Colored by Patient Condition")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="Condition (0: Healthy, 1: Parkinson's, 2: Differential diagnosis)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------
# --- t-SNE Colored by Age-at-Diagnosis Gap
# ----------------------------------------

# Compute age difference (only for Parkinson's)
age_difference_vector = df.apply(
    lambda row: abs(row['age'] - row['age_at_diagnosis']) if row['condition'] == "Parkinson's" else -1,
    axis=1
).values

# Create bins to categorize age difference
age_difference_bins = pd.cut(age_difference_vector, bins=[-1, 0, 10, 20, 30, 40, 50,60,70,80], labels=False)
color_rule = age_difference_bins

# New t-SNE projection colored by an age-at-diagnosis interval
tsne_new = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne_new = tsne_new.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne_new[:, 0], X_tsne_new[:, 1], c=color_rule, cmap='viridis', alpha=0.8)
plt.title("t-SNE Visualization Colored by absolute value of Age Difference from diagnosis")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="Age Difference Intervals")
plt.grid(True)
plt.tight_layout()
plt.show()
