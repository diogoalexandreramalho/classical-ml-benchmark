# Parkinson's Disease & Forest Cover Type Classification

End-to-end ML pipeline for two classification problems: detecting Parkinson's disease from speech features (binary) and predicting forest cover type from cartographic data (multiclass, 7 classes).

## Overview

| | Parkinson's Disease | Forest Covertype |
|---|---|---|
| **Task** | Binary classification | Multiclass (7 classes) |
| **Instances** | 756 | 581,012 |
| **Features** | 754 | 54 |
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification) | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Covertype) |

## Key Results

> **Note:** Replace the values below with your actual results. These are placeholders.

| Model | Parkinson's (Accuracy) | Covertype (Accuracy) |
|---|---|---|
| Naïve Bayes | _X%_ | _X%_ |
| Decision Tree | _X%_ | _X%_ |
| KNN | _X%_ | _X%_ |
| Random Forest | _X%_ | _X%_ |
| XGBoost | _X%_ | _X%_ |
| Gradient Boosting | _X%_ | _X%_ |

## Pipeline

```
Raw Data → Cleaning → Feature Scaling → Feature Selection → PCA → Data Balancing → Model Training → Evaluation
```

### Preprocessing
- **Feature scaling** — normalization of input features (`Normalize.py`)
- **Feature selection** — dimensionality reduction to remove noisy/redundant features (`Feature_Selection.py`)
- **PCA** — principal component analysis for further dimensionality reduction (`PCA.py`)
- **Data balancing** — handling class imbalance in the Parkinson's dataset (`data_balancing.py`)

### Unsupervised Learning
- **Clustering** — exploratory analysis to understand data structure (`Clustering.py`, `Unsupervised.py`)
- **Association Rule Mining** — pattern discovery across features (`ARM.py`)

### Classification Models
Each model was tuned via hyperparameter optimization to maximize performance:

| Script | Model |
|---|---|
| `naive_bayes.py` | Naïve Bayes |
| `Decision_Tree.py` | Decision Trees |
| `KNN.py` | K-Nearest Neighbors |
| `RandomForest.py` | Random Forests |
| `GradientBoost.py` | Gradient Boosting |
| `XGBoost.py` | XGBoost |

Results are compared in `compareResults.py` and visualized in the `Charts/` folder.

## Project Structure

```
├── Data/                  # Raw datasets
├── Charts/                # Generated plots and visualizations
├── Correlations/          # Correlation analysis outputs
├── Groups/                # Grouped analysis outputs
├── test_cases/            # Test configurations
├── main.py                # Main entry point
├── Preprocessing.py       # Data preprocessing pipeline
├── data_cleaning.py       # Data cleaning utilities
├── Normalize.py           # Feature scaling
├── Feature_Selection.py   # Feature selection methods
├── PCA.py                 # Principal Component Analysis
├── data_balancing.py      # Class balancing (SMOTE, etc.)
├── naive_bayes.py         # Naïve Bayes classifier
├── Decision_Tree.py       # Decision Tree classifier
├── KNN.py                 # K-Nearest Neighbors
├── RandomForest.py        # Random Forest classifier
├── GradientBoost.py       # Gradient Boosting classifier
├── XGBoost.py             # XGBoost classifier
├── Classification.py      # Classification utilities
├── Clustering.py          # Clustering analysis
├── Unsupervised.py        # Unsupervised learning methods
├── ARM.py                 # Association Rule Mining
├── analysis.py            # Statistical analysis
├── compareResults.py      # Model comparison
├── plot_functions.py      # Plotting utilities
└── print_statistics.py    # Summary statistics
```

## Getting Started

### Requirements

```
pip install numpy pandas scikit-learn xgboost matplotlib seaborn mlxtend
```

### Usage

```bash
# Run the full pipeline
python main.py
```

## Datasets

**Parkinson's Disease** — Speech recordings from 188 PD patients and 64 healthy controls (Istanbul University). Features extracted using MFCCs, wavelet transforms, vocal fold features, and TWQT.

**Forest Covertype** — Cartographic data from the US Forest Service for 30×30m cells. Features include elevation, slope, distance to water, sunlight, and soil type. Target: 7 tree species.

## Tech Stack

Python · scikit-learn · XGBoost · pandas · NumPy · Matplotlib · Seaborn

## Authors

Diogo Ramalho, André Guerra and Miguel Trinca
