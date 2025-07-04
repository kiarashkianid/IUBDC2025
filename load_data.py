"""get preprocessed data using relative paths"""
import numpy as np
import pandas as pd
import os

# Assuming your script is in the parent folder of 'preprocessed'
movement_path = os.path.join(
    "pads-parkinsons-disease-smartwatch-dataset-1.0.0",
    "preprocessed",
    "movement"
)
label_csv_path = os.path.join(
    "pads-parkinsons-disease-smartwatch-dataset-1.0.0",
    "preprocessed",
    "file_list.csv"
)

# Load labels
df = pd.read_csv(label_csv_path)
ids = df["id"].values
labels = df["label"].values

n_channels = 72
n_timesteps = 1789

X = []
new_labels = []

for id_, label in zip(ids, labels):
    bin_path = os.path.join(movement_path, f"{id_}_ml.bin")
    if os.path.exists(bin_path):
        arr = np.fromfile(bin_path, dtype=np.float32)
        # print(f"{id_}: {arr.shape}")
        arr = arr[:n_channels * n_timesteps]  # Trim!
        arr = arr.reshape((n_channels, n_timesteps))
        X.append(arr)
        new_labels.append(label)

X = np.stack(X)
y = np.array(new_labels)

print("Loaded successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)
print("Saved X.npy and y.npy with shapes:", X.shape, y.shape)
