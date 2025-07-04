import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



# === Load and Prepare Data ===
df_filtered = pd.read_csv("../data/filtered_data.csv")
print(df_filtered.head())
"""
# Separate features and target
X = df_filtered.drop(columns=['condition'])
y = df_filtered['condition']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X)

# Encode target if it's categorical
if y.dtype == 'object' or y.dtype.name == 'category':
    le = LabelEncoder()
    y = le.fit_transform(y)

# === Initialize Classifier ===
clf = DecisionTreeClassifier(random_state=42)

# === Define Scoring Metrics ===
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# === Perform 5-Fold Cross-Validation ===
cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring)

# === Display Cross-Validation Results ===
print("Cross-validation results:")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize()}: {scores.mean():.4f} ± {scores.std():.4f}")

# === Train Final Model on Entire Dataset ===
clf.fit(X, y)

# === Textual Visualization of Decision Tree ===
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\nTrained Decision Tree Rules:")
print(tree_rules)

# === Graphical Visualization of Decision Tree ===
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=[str(cls) for cls in set(y)],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Trained on Entire Dataset")
plt.show()
"""
