# Beyond Diagnosis: Enhancing Parkinson’s Disease Classification and Symptom Profiling Using Wearable Data

## Overview

This project explores the use of wearable sensor data and machine learning techniques to improve the diagnosis and symptom profiling of Parkinson’s Disease (PD). By leveraging accelerometer data from Apple Watches and applying both supervised and unsupervised learning techniques, we aim to enhance early detection, classification, and symptom understanding of PD in a real-world context.

## Objectives

-  **Replicate** a supervised ML pipeline using Bag-of-Symbolic Fourier Approximation Symbols (BOSS) and Support Vector Machine (SVM) for PD classification
-  **Extend** prior work by applying unsupervised learning to discover tremor-related symptom clusters from accelerometer data

## Dataset

- **Source**: Parkinson’s Apple Watch Dataset (PADS)
- **Participants**: 469 individuals with wrist-mounted Apple Watches
- **Data Includes**:
  - 3-axis accelerometer signals
  - Symptom severity questionnaires
  - Patient metadata
(for the comprehensive reference list and the original paper look at the manuscript)

## Methods

### Supervised Classification

- **Pipeline**: BOSS + SVM
- **Tasks**:
  - PD vs. Healthy Controls (HC)
  - PD vs. Differential Diagnoses (DD)
- **Evaluation**: Nested 5-fold cross-validation

### Unsupervised Symptom Profiling

- **Signal Processing**:
  - Compute Euclidean magnitude from 3-axis accelerometer data
  - Bandpass filter between 3–12 Hz (tremor range)
- **Feature Extraction**:
  - Mean acceleration
  - Mean envelope amplitude
  - Peak power (tremor band)
  - Area under the PSD curve
- **Clustering**:
  - Gaussian Mixture Models (GMM)
  - Cluster selection via Bayesian Information Criterion (BIC)
  - Evaluation using silhouette score and Calinski-Harabasz index
- **Visualization**:
  - t-SNE embedding colored by cluster, diagnosis, and disease duration proxy

## Results

### Supervised Classification

| Task      | Accuracy (%) | F1 Score | Precision | Recall |
|-----------|--------------|----------|-----------|--------|
| PD vs. HC | 78.0 ± 5.0   | 0.75 ± 0.05 | 0.75 ± 0.07 | 0.77 ± 0.07 |
| PD vs. DD | 76.1 ± 3.0   | 0.63 ± 0.05 | 0.67 ± 0.14 | 0.63 ± 0.08 |

### Clustering Validity

- **Silhouette Score**: 0.688
- **Calinski-Harabasz Index**: 1718.96


## Video
- https://youtu.be/hsJStp7vkmU

## Authors

- **Charlotte Wong** – University of Toronto  
- **Kiarash Kianidehkordi** – University of Toronto




