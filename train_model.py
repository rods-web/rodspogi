import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE

print("STEP 1: Loading Dataset")
df = pd.read_csv("data/diabetes.csv")
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Outcome'].value_counts()}")

print("\nSTEP 2: Data Preprocessing")
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
for col in zero_as_missing:
    df[col] = df[col].fillna(df[col].median())
print(f"Total missing: {df.isnull().sum().sum()}")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = X.columns.tolist()

print("\nSTEP 3: Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {X_train.shape}  Testing: {X_test.shape}")

print("\nSTEP 4: Feature Scaling")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved")

print("\nSTEP 5: Hybrid Feature Selection (RFE + RF)")
rf_for_selection = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_for_selection, n_features_to_select=6, step=1)
rfe.fit(X_train_scaled, y_train)
selected_mask = rfe.support_
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
print(f"Selected features: {selected_features}")

rf_for_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_for_importance.fit(X_train_scaled, y_train)
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_for_importance.feature_importances_
}).sort_values("Importance", ascending=False)
print(f"Gini Importance:\n{importance_df.to_string(index=False)}")

X_train_sel = X_train_scaled[:, selected_mask]
X_test_sel = X_test_scaled[:, selected_mask]
joblib.dump(selected_mask, "models/feature_mask.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

print("\nSTEP 6: SMOTE")
print(f"Before SMOTE: {np.bincount(y_train)}")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sel, y_train)
print(f"After SMOTE:  {np.bincount(y_train_bal)}")

print("\nSTEP 7: Hyperparameter Tuning (this takes 1-3 min)")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}
rf_base = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_bal, y_train_bal)
best_rf = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

print("\nSTEP 8: Evaluation")
y_pred = best_rf.predict(X_test_sel)
y_proba = best_rf.predict_proba(X_test_sel)[:, 1]
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

print("\nSTEP 9: Comparing with Baselines")
baselines = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}
results = [{
    "Model": "Enhanced RF (HFS-RF)",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba),
}]
for name, m in baselines.items():
    m.fit(X_train_bal, y_train_bal)
    p = m.predict(X_test_sel)
    pr = m.predict_proba(X_test_sel)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p),
        "Recall": recall_score(y_test, p),
        "F1": f1_score(y_test, p),
        "AUC": roc_auc_score(y_test, pr),
    })
results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))
results_df.to_csv("models/comparison_results.csv", index=False)

print("\nSTEP 10: Saving Model")
joblib.dump(best_rf, "models/enhanced_rf.pkl")
print("Model saved -> models/enhanced_rf.pkl")
print("TRAINING COMPLETE!")