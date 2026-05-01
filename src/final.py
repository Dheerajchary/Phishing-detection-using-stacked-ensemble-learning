# final.py
# Enhanced Multi-Layer Stacked Ensemble Learning Model for Phishing Detection

# Pipeline (all in one file):
#   1. Load dataset_small.csv
#   2. EDA — duplicates, missing values, outlier CLIPPING (IQR-based, no data loss)
#   3. Train/Test split FIRST (before SMOTE — prevents data leakage)
#   4. Class balancing — SMOTE on training set ONLY
#   5. Feature selection — RF + L1 + Correlation + PCA (intersection)
#   6. Layer 1: MLP, DecisionTree, LGBM, CatBoost, HistGradientBoosting
#              → predictions passed as input to Layer 2 (TRUE stacking)
#   7. Layer 2: GradientBoosting, RandomForest, KNN
#              → predictions passed as input to Layer 3 (TRUE stacking)
#   8. Layer 3 (Meta): XGBoost → Final prediction


import os
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to this file's location
# Works whether you run from project root or from inside src/
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)
import scipy.stats as stats
import xgboost as xgb
import joblib

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not installed — using GradientBoostingClassifier as substitute.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed — using AdaBoostClassifier as substitute.")

from imblearn.over_sampling import SMOTE

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("PHISHING DETECTION — ENHANCED MULTILAYER STACKED ENSEMBLE")
print("=" * 60)

print("\n[Step 1] Loading dataset...")
dataset = pd.read_csv(os.path.join(DATA_DIR, 'dataset_small.csv'))
print(f"  Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
print(f"  Class counts:\n{dataset['phishing'].value_counts().to_string()}")


# ─────────────────────────────────────────────────────────────────────
# STEP 2: EDA — duplicates, missing values, outlier CLIPPING
# ─────────────────────────────────────────────────────────────────────

print("\n[Step 2] Exploratory Data Analysis...")

print(f"\n  Missing values: {dataset.isnull().sum().sum()} total")

before = len(dataset)
dataset = dataset.drop_duplicates()
print(f"  Duplicates removed: {before - len(dataset)} rows")
print(f"  Rows after deduplication: {len(dataset)}")

# Correlation heatmap
plt.figure(figsize=(16, 9))
sns.heatmap(dataset.corr(), cmap='Blues', annot=False, linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'))
plt.show()
print("  Saved: results/correlation_heatmap.png")

# Outlier detection using IQR on features only (exclude target column)
X_raw = dataset.drop('phishing', axis=1)
y_raw = dataset['phishing']

q1  = X_raw.quantile(0.25)
q3  = X_raw.quantile(0.75)
iqr = q3 - q1
outliers   = ((X_raw < (q1 - 1.5 * iqr)) | (X_raw > (q3 + 1.5 * iqr)))
top_10_cols = outliers.sum().sort_values(ascending=False).head(10).index.tolist()

# Plot BEFORE clipping
fig, axes = plt.subplots(1, 10, figsize=(14, 6))
for i, col in enumerate(top_10_cols):
    sns.boxplot(X_raw[col].dropna(), color='red', orient='v', ax=axes[i])
    axes[i].set_xlabel(col, fontsize=7, rotation=45)
plt.suptitle("Outliers BEFORE Clipping")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outliers_before.png'))
plt.show()

#  CLIP outliers instead of removing rows — retains all data
# Values beyond IQR boundaries are capped, not deleted
for col in X_raw.columns:
    lower = q1[col] - 1.5 * iqr[col]
    upper = q3[col] + 1.5 * iqr[col]
    X_raw[col] = np.clip(X_raw[col], lower, upper)

print(f"  Outliers clipped (no rows removed) — all {len(X_raw)} rows retained")

# Plot AFTER clipping
fig, axes = plt.subplots(1, 10, figsize=(14, 6))
for i, col in enumerate(top_10_cols):
    sns.boxplot(X_raw[col].dropna(), color='green', orient='v', ax=axes[i])
    axes[i].set_xlabel(col, fontsize=7, rotation=45)
plt.suptitle("Outliers AFTER Clipping")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outliers_after.png'))
plt.show()
print("  Saved: results/outliers_before.png, outliers_after.png")


# ─────────────────────────────────────────────────────────────────────
# STEP 3: Train/Test Split FIRST (before SMOTE)
# ─────────────────────────────────────────────────────────────────────

print("\n[Step 3] Train/Test Split...")

# Split BEFORE SMOTE — test set is never seen by oversampling
# stratify=y_raw ensures both splits have same class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

print(f"  Train set : {X_train.shape[0]} rows")
print(f"  Test set  : {X_test.shape[0]} rows")
print(f"  Train class counts: {y_train.value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────
# STEP 4: Class Balancing — SMOTE on training set ONLY
# ─────────────────────────────────────────────────────────────────────

print("\n[Step 4] Class Balancing with SMOTE...")

plt.figure(figsize=(5, 4))
train_df = pd.concat([X_train, y_train], axis=1)
sns.countplot(data=train_df, x='phishing', palette=['red', 'green'])
plt.title('Class Distribution BEFORE SMOTE (Train set)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'class_before_smote.png'))
plt.show()
print(f"  Before: {y_train.value_counts().to_dict()}")

# SMOTE only on training data — test set untouched and realistic
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  After:  {pd.Series(y_train_res).value_counts().to_dict()}")

balanced_df = pd.concat([pd.DataFrame(X_train_res, columns=X_train.columns),
                          pd.Series(y_train_res, name='phishing')], axis=1)
plt.figure(figsize=(5, 4))
sns.countplot(data=balanced_df, x='phishing', palette=['red', 'green'])
plt.title('Class Distribution AFTER SMOTE (Train set)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'class_after_smote.png'))
plt.show()
print("  Saved: results/class_before_smote.png, class_after_smote.png")

X_train_bal = pd.DataFrame(X_train_res, columns=X_train.columns)
y_train_bal = pd.Series(y_train_res, name='phishing')


# ─────────────────────────────────────────────────────────────────────
# STEP 5: Feature Selection (fitted on training data only)
# ─────────────────────────────────────────────────────────────────────

print("\n[Step 5] Feature Selection...")

X_scaled = MinMaxScaler().fit_transform(X_train_bal)

# Method 1: Random Forest importance
sel_rf = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
sel_rf.fit(X_scaled, y_train_bal)
features_rf = X_train_bal.columns[sel_rf.get_support()].tolist()
print(f"  RF Importance     : {len(features_rf)} features")

# Method 2: L1-based (Lasso)
sel_l1 = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=1.0))
sel_l1.fit(X_scaled, y_train_bal)
features_l1 = X_train_bal.columns[sel_l1.get_support()].tolist()
print(f"  L1 Based          : {len(features_l1)} features")

# Method 3: Correlation coefficient
corr_scores, p_values = [], []
for feature in X_train_bal.columns:
    corr, p = stats.pearsonr(X_train_bal[feature], y_train_bal)
    corr_scores.append(corr)
    p_values.append(p)
corr_df = pd.DataFrame({'Feature': X_train_bal.columns,
                         'Correlation': corr_scores, 'p-value': p_values})
k = len(corr_df[corr_df['p-value'] < 0.05])
sel_corr = SelectKBest(f_regression, k=k)
sel_corr.fit(X_train_bal, y_train_bal)
features_corr = X_train_bal.columns[sel_corr.get_support()].tolist()
print(f"  Correlation Coeff : {len(features_corr)} features")

# Method 4: PCA (95% variance)
X_std = StandardScaler().fit_transform(X_scaled)
pca = PCA()
pca.fit(X_std)
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
pca = PCA(n_components=n_components)
pca.fit(X_std)
top_idx = np.abs(pca.components_[0]).argsort()[::-1][:n_components]
features_pca = X_train_bal.columns[top_idx].tolist()
print(f"  PCA               : {len(features_pca)} features")

# Final intersection — selected by 3+ methods
counter = Counter(features_rf + features_l1 + features_corr + features_pca)
selected_features = [f for f, count in counter.items() if count > 2]
print(f"  Final intersection — selected by 3+ methods: {len(selected_features)} features")

X_train_sel = X_train_bal[selected_features]
X_test_sel  = X_test[selected_features]

pd.concat([X_train_sel, y_train_bal], axis=1).to_csv(
    os.path.join(DATA_DIR, "selected_columns.csv"), index=False)
print("  Saved: data/selected_columns.csv")

# Scale — fit on train, apply same transform to test
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_sel)
X_test_sc  = sc.transform(X_test_sel)


# ─────────────────────────────────────────────────────────────────────
# STEP 6–8: Three-Layer Stacked Ensemble
# Each layer's predicted probabilities become the input to the next
# ─────────────────────────────────────────────────────────────────────

def eval_model(model, X_t, y_t, layer):
    y_pred = model.predict(X_t)
    acc  = accuracy_score(y_t, y_pred)
    prec = precision_score(y_t, y_pred)
    rec  = recall_score(y_t, y_pred)
    f1   = f1_score(y_t, y_pred)
    avg  = (acc + prec + rec + f1) / 4
    print(f"  {type(model).__name__} ({layer}): "
          f"Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f} | Avg={avg:.3f}")
    return y_pred


# ── Layer 1 ───────────────────────────────────────────────────────────
print("\n[Step 6] Layer 1 Training...")

layer1_models = [
    MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42),
    DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=42),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=5, random_state=42),
    AdaBoostClassifier(random_state=42),
    HistGradientBoostingClassifier(learning_rate=0.1, max_depth=5, random_state=42)
]
if LGBM_AVAILABLE:
    layer1_models[2] = lgb.LGBMClassifier(learning_rate=0.1, num_leaves=50, random_state=42)
if CATBOOST_AVAILABLE:
    layer1_models[3] = CatBoostClassifier(learning_rate=0.1, depth=6, verbose=False, random_state=42)

for model in layer1_models:
    model.fit(X_train_sc, y_train_bal)

print("\n  Layer 1 Results (on test set):")
for model in layer1_models:
    eval_model(model, X_test_sc, y_test, "Layer 1")

# Collect predicted probabilities from all Layer 1 models
# These become the feature matrix fed into Layer 2
L1_train = np.column_stack([m.predict_proba(X_train_sc)[:, 1] for m in layer1_models])
L1_test  = np.column_stack([m.predict_proba(X_test_sc)[:, 1]  for m in layer1_models])
print(f"\n  Layer 1 output shape → Layer 2 input: {L1_train.shape}")


# ── Layer 2 ───────────────────────────────────────────────────────────
print("\n[Step 7] Layer 2 Training...")

layer2_models = [
    GradientBoostingClassifier(learning_rate=0.1, max_depth=3, random_state=42),
    RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
    KNeighborsClassifier(n_neighbors=5)
]

# Layer 2 trains on Layer 1's predictions — not on raw features
for model in layer2_models:
    model.fit(L1_train, y_train_bal)

print("\n  Layer 2 Results (on test set):")
for model in layer2_models:
    eval_model(model, L1_test, y_test, "Layer 2")

# Collect Layer 2 predicted probabilities → input for Layer 3
L2_train = np.column_stack([m.predict_proba(L1_train)[:, 1] for m in layer2_models])
L2_test  = np.column_stack([m.predict_proba(L1_test)[:, 1]  for m in layer2_models])
print(f"\n  Layer 2 output shape → Layer 3 input: {L2_train.shape}")


# ── Layer 3 — Meta Model (XGBoost) ────────────────────────────────────
print("\n[Step 8] Layer 3 — Meta Model (XGBoost)...")

meta_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
# Meta model trains on Layer 2's predictions — not on raw features
meta_model.fit(L2_train, y_train_bal)
y_final = meta_model.predict(L2_test)

acc  = accuracy_score(y_test, y_final)
prec = precision_score(y_test, y_final)
rec  = recall_score(y_test, y_final)
f1   = f1_score(y_test, y_final)

print(f"\n{'=' * 60}")
print(f"  FINAL MODEL RESULTS (XGBoost — Meta Layer)")
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}  ({prec*100:.2f}%)")
print(f"  Recall    : {rec:.4f}  ({rec*100:.2f}%)")
print(f"  F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
print(f"{'=' * 60}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_final)
print(f"\n  Confusion Matrix:\n{cm}")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Meta Model (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
plt.show()
print("  Saved: results/confusion_matrix.png")

# ROC Curve
y_prob = meta_model.predict_proba(L2_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
plt.show()
print("  Saved: results/roc_curve.png")

# Save everything needed for inference
joblib.dump(meta_model,       os.path.join(MODELS_DIR, 'meta_model_xgboost.pkl'))
joblib.dump(sc,               os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(layer1_models,    os.path.join(MODELS_DIR, 'layer1_models.pkl'))
joblib.dump(layer2_models,    os.path.join(MODELS_DIR, 'layer2_models.pkl'))
joblib.dump(selected_features,os.path.join(MODELS_DIR, 'selected_features.pkl'))
print("\nModel saved  → models/meta_model_xgboost.pkl")
print("Scaler saved → models/scaler.pkl")
print("All plots    → results/")