"""test boss"""
import numpy as np
from scripts.multi_boss import MultiBOSS
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. Load data
X = np.load("X.npy")
y = np.load("y.npy")

# 2. Map your class labels (adjust these if your dataset uses different numbers)
PD = 0
DD = 1
HC = 2

# 3. Filter for PD vs HC
mask_pd_hc = (y == PD) | (y == HC)
X_pd_hc = X[mask_pd_hc]
y_pd_hc = y[mask_pd_hc]
y_pd_hc = (y_pd_hc == PD).astype(int)

# 4. Filter for PD vs DD
mask_pd_dd = (y == PD) | (y == DD)
X_pd_dd = X[mask_pd_dd]
y_pd_dd = y[mask_pd_dd]
y_pd_dd = (y_pd_dd == PD).astype(int)

# 5. Define the run_cv function (put this above the calls if you want)
def run_cv(X, y):
    mb = MultiBOSS(data_shape=(72, 1789))
    X_boss = mb.fit_transform(X, y)
    clf = SVC(kernel='linear', probability=True, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X_boss, y, cv=cv, scoring='balanced_accuracy')
    f1 = cross_val_score(clf, X_boss, y, cv=cv, scoring='f1')
    prec = cross_val_score(clf, X_boss, y, cv=cv, scoring='precision')
    rec = cross_val_score(clf, X_boss, y, cv=cv, scoring='recall')
    print(f"Balanced accuracy: {np.mean(acc):.2f} ± {np.std(acc):.2f}")
    print(f"F1: {np.mean(f1):.2f} ± {np.std(f1):.2f}")
    print(f"Precision: {np.mean(prec):.2f} ± {np.std(prec):.2f}")
    print(f"Recall: {np.mean(rec):.2f} ± {np.std(rec):.2f}")

# 6. Run cross-validation for each task
print("Results for PD vs HC")
run_cv(X_pd_hc, y_pd_hc)

print("\nResults for PD vs DD")
run_cv(X_pd_dd, y_pd_dd)
