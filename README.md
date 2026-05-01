<div align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-189AC8?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/LightGBM-2E4057?style=for-the-badge&logo=lightgbm&logoColor=white"/>

<br/><br/>

# 🔐 Phishing Detection using Enhanced Multilayer Stacked Ensemble Learning

<br/>

> *Phishing attacks remain one of the most dangerous threats in the digital age — fraudsters craft deceptive websites that look identical to legitimate ones, tricking users into surrendering sensitive information. This project tackles the problem head-on with an Enhanced Multi-Layer Stacked Ensemble Learning model that combines four feature selection techniques and three progressive layers of machine learning algorithms, each layer feeding its predictions into the next.*

<br/>

</div>

---

## 🏗️ Model Architecture

```
Phishing Dataset (58,000 samples, 111 features)
                      │
                      ▼
 ┌─────────────────────────────────────────────┐
 │  EDA + Cleaning                             │
 │  Remove duplicates                          │
 |    Handle outliers (IQR clip)               │
 └────────────────────┬────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Train / Test Split (80% / 20%)             │
 └────────────────────┬────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Class Balancing — SMOTE (train set only)   │
 └────────────────────┬────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Feature Selection                          │
 │  RF Importance + L1 + Correlation + PCA     │
 │  → Final features (selected by 3+ methods)  │
 └────────────────────┬────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Layer 1 — Base Models                      │
 │  MLP · DecisionTree · LGBM                  │
 │  CatBoost · HistGradientBoosting            │
 └────────────────────┬────────────────────────┘
                      │  predicted probabilities 
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Layer 2 — Ensemble Models                  │
 │  GradientBoosting · RandomForest · KNN      │
 └────────────────────┬────────────────────────┘
                      │  predicted probabilities
                      ▼
 ┌─────────────────────────────────────────────┐
 │  Layer 3 — Meta Model (XGBoost)             │
 │  Final prediction: Phishing or Legitimate   │
 └─────────────────────────────────────────────┘
```

---

## 📊 Final Results

**Layer 1 : Base Models**

| Algorithm              | Accuracy | Precision | Recall | F1-Score |
| ---------------------- | -------- | --------- | ------ | -------- |
| MLPClassifier          | 0.942    | 0.947     | 0.943  | 0.945    |
| DecisionTreeClassifier | 0.928    | 0.928     | 0.938  | 0.933    |
| LightGBM               | 0.951    | 0.955     | 0.953  | 0.954    |
| AdaBoostClassifier     | 0.885    | 0.892     | 0.892  | 0.892    |
| HistGradientBoosting   | 0.939    | 0.938     | 0.948  | 0.943    |

**Layer 2 : Ensemble Models**

| Algorithm            | Accuracy | Precision | Recall | F1-Score |
| -------------------- | -------- | --------- | ------ | -------- |
| GradientBoosting     | 0.951    | 0.955     | 0.952  | 0.954    |
| RandomForest         | 0.951    | 0.956     | 0.951  | 0.954    |
| KNeighborsClassifier | 0.948    | 0.953     | 0.949  | 0.951    |


**Layer 3 : Meta Model**

| Meta Model  | Accuracy  | Precision | Recall    | F1-Score  |
| ----------- | --------- | --------- | --------- | --------- |
| **XGBoost** | **0.948** | **0.952** | **0.950** | **0.951** |


**ROC-AUC: 0.97**

---

## 🗂️ Project Structure

```
phishing-detection/
├── src/
│   └── final.py            ← entire pipeline in one file
├── data/
│   ├── dataset_small.csv   ← dataset (download separately)
│   └── README.md           ← dataset instructions
├── models/                 ← stores .pkl files after training
├── results/                ← stores plots after training
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone & install dependencies

```bash
git clone https://github.com/Dheerajchary/phishing-detection.git
cd phishing-detection
pip install -r requirements.txt
```

### 2. Download the dataset

See `data/README.md` for instructions. Place the file at `data/dataset_small.csv`.

### 3. Run the pipeline

```bash
python src/final.py
```

All steps run automatically in order — EDA, balancing, feature selection, training all 3 layers, evaluation, and saving the model.

---

## 🔬 Feature Selection Methods

| Method | Features Selected |
|--------|------------------|
| PCA (95% variance threshold) | ~13 |
| Random Forest Feature Importance | ~36 |
| L1-based (Lasso) | ~44 |
| Correlation Coefficient (p < 0.05) | ~51 |
| **Final (selected by 3+ methods)** | **~34** |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| ML Framework | scikit-learn, XGBoost, LightGBM, CatBoost |
| Data Processing | pandas, numpy, scipy |
| Visualization | matplotlib, seaborn |
| Class Balancing | imbalanced-learn (SMOTE) |
| Environment | Google Colab / Local Python |

---

## 📚 Dataset

> G. Vrbancic, "Phishing websites dataset," Mendeley Data, vol. 1, 2020.
> https://data.mendeley.com/datasets/72ptz43s9v/1

~58,000 samples · 111 features · Binary classification (phishing=1, legitimate=0)

---

<div align="center">

Made by [Dheeraj Kumar Vadla](https://github.com/Dheerajchary)

</div>
