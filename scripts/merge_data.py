import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
import json

# === Load Patient Data ===
# Define the folder path containing patient JSON files
patient_folder = r"C:\Users\kiara\Desktop\pads-parkinsons-disease-smartwatch-dataset-1.0.0\pads-parkinsons-disease-smartwatch-dataset-1.0.0\patients"

# Define the folder path containing questionnaire JSON files
questionnaire_folder = r"C:\Users\kiara\Desktop\pads-parkinsons-disease-smartwatch-dataset-1.0.0\pads-parkinsons-disease-smartwatch-dataset-1.0.0\questionnaire"

# List all JSON files in the patient folder
patient_files = [f for f in os.listdir(patient_folder) if f.endswith('.json')]

# Initialize an empty list to store patient data
patients = []

# Load each patient JSON file and append its data to the list
for file in patient_files:
    with open(os.path.join(patient_folder, file), 'r') as f:
        data = json.load(f)
        patients.append(data)

# Convert the list of patient data into a DataFrame
df_patients = pd.DataFrame(patients)

# Standardize column names to lowercase
df_patients.columns = df_patients.columns.str.lower()

# Display the column names and a preview of the patient DataFrame
print("Patient Data Columns:", df_patients.columns.tolist())
print(df_patients.head())

# === Load Questionnaire Data ===
# List all JSON files in the questionnaire folder
questionnaire_files = [f for f in os.listdir(questionnaire_folder) if f.endswith('.json')]

# Initialize an empty list to store questionnaire data
questionnaire_data = []

# Load each questionnaire JSON file and append its data to the list
for file in questionnaire_files:
    with open(os.path.join(questionnaire_folder, file), 'r') as f:
        data = json.load(f)
        subject_id = data['subject_id']  # Extract subject ID
        # Extract responses and map them to their respective link IDs
        responses = {item['link_id']: item['answer'] for item in data['item']}
        responses['subject_id'] = subject_id  # Add subject ID to the responses
        questionnaire_data.append(responses)

# Convert the list of questionnaire data into a DataFrame
df_questionnaire = pd.DataFrame(questionnaire_data)

# Display the column names and a preview of the questionnaire DataFrame
print("Questionnaire Data Columns:", df_questionnaire.columns.tolist())
print(df_questionnaire.head())

# === Merge DataFrames ===
# Rename 'id' column in the patient DataFrame to 'subject_id' for consistency
df_patients.rename(columns={'id': 'subject_id'}, inplace=True)

# Ensure both 'subject_id' columns are strings for consistency
df_patients['subject_id'] = df_patients['subject_id'].astype(str)
df_questionnaire['subject_id'] = df_questionnaire['subject_id'].astype(str)

# Merge the patient and questionnaire DataFrames on 'subject_id'
df_merged = pd.merge(df_patients, df_questionnaire, on='subject_id', how='inner')

# Display the column names and a preview of the merged DataFrame
print("Merged Data Columns:", df_merged.columns.tolist())
print(df_merged.head(10))

# Save the merged DataFrame to a CSV file
df_merged.to_csv("merged_data.csv", index=False)

# Check all unique values in the 'condition' column
unique_conditions = df_merged['condition'].unique()
print("Unique conditions before merging DD group:", unique_conditions)

# === Consolidate Conditions for Differential Diagnosis Group (DD) ===
# Define the conditions to merge into "Other Movement Disorders"
dd_conditions = ['Atypical Parkinsonism', 'Multiple Sclerosis', 'Essential Tremor']

# Replace those values with 'Other Movement Disorders' in the 'condition' column
df_merged['condition'] = df_merged['condition'].replace(dd_conditions, 'Other Movement Disorders')

# Optional: Rename 'Other Movement Disorders' to 'Differential Diagnosis (DD)' for clarity
df_merged['condition'] = df_merged['condition'].replace({'Other Movement Disorders': 'Differential Diagnosis'})

# === Filter DataFrame ===
# Define the conditions to keep in the filtered DataFrame
filtered_conditions = ["Parkinson's", 'Healthy', 'Differential Diagnosis']

# Filter the DataFrame to include only the specified conditions
df_filtered = df_merged[df_merged['condition'].isin(filtered_conditions)]

# Display a preview of the filtered DataFrame
print("Filtered DataFrame:")
print(df_filtered.head())

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv("filtered_data.csv", index=False)

# Check all unique values in the 'condition' column of the filtered DataFrame
unique_conditions = df_filtered['condition'].unique()
print("Final unique conditions:", unique_conditions)

# === Plot Histogram ===
# Plot a histogram for the three categories
df_filtered['condition'].value_counts().plot(
    kind='bar', figsize=(10, 6), color=['blue', 'green', 'orange'])

# Add title and labels to the plot
plt.title('Distribution of Conditions (Parkinson\'s, Healthy, Differential Diagnosis)')
plt.xlabel('Condition')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Display the plot
plt.show()
