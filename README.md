# Airline Passenger Satisfaction Prediction

A machine learning pipeline that predicts whether an airline passenger is satisfied or dissatisfied, based on flight and service attributes. Includes full EDA, feature engineering, automated model comparison, hyperparameter tuning, and a Streamlit app for live predictions.

## Problem

Given a passenger satisfaction survey (129,880 records, 24 attributes covering service ratings, flight details, and demographics), the goal is to identify which factors most influence satisfaction and build a model that predicts satisfaction for new passengers.

Dataset adapted from the airline passenger satisfaction survey by John D (Kaggle), cleaned for classification.

## Approach

**1. Data Cleaning**
- Loaded 129,880 records × 24 columns
- Dropped missing values and the non-predictive `id` column

**2. Exploratory Data Analysis**
- Automated profiling via `pandas_profiling` (`ProfileReport`) to surface missing values, distributions, and correlated features
- Visualized satisfaction breakdown by gender, customer type, class, and travel type
- Outlier detection and removal using the IQR method across all numerical features

**3. Feature Engineering**
- Label-encoded all categorical columns (`Gender`, `Customer Type`, `Type of Travel`, `Class`, target)
- Correlation analysis against the target (`satisfaction_v2`) to identify the strongest drivers — top features: **Inflight entertainment, Ease of online booking, Online support, On-board service, Online boarding**
- Combined `Departure Delay` and `Arrival Delay` (correlation ≈ 0.95) into a single `Delay` feature
- Dropped `Gate location` and `Flight Distance` — negligible correlation with satisfaction

**4. Modeling**
- Ran AutoML comparison across 29 classifiers via `LazyPredict` to shortlist candidates
- Manually compared top candidates (RandomForest, LightGBM, XGBoost, SVC, KNN) with and without feature scaling
- **RandomForestClassifier** selected as the best tradeoff of accuracy and training time
- Tuned hyperparameters with `GridSearchCV` (5-fold CV, 32 candidate combinations) → best params: `criterion='entropy', max_depth=50, n_estimators=200`

**5. Evaluation**
- **Accuracy: 95.6%** on the held-out test set
- Precision/recall balanced across both classes (~0.95–0.96 F1-score)
- Validated with 10-fold cross-validation and a confusion matrix

**6. Deployment**
- Serialized the trained model and label encoder (`pickle` + `bz2` compression)
- Built a Streamlit app (`App.py`) that collects passenger inputs and returns a satisfaction prediction with probability score

## Repository Contents

| File | Description |
|---|---|
| `Airline passenger satsification.ipynb` | Full pipeline: EDA, feature engineering, model comparison, tuning, evaluation |
| `App.py` | Streamlit app for live satisfaction prediction |
| `model.pkl` | Serialized trained RandomForestClassifier (bz2-compressed) |
| `Transformer.pkl` | Serialized label encoder for categorical inputs (bz2-compressed) |
| `requirements.txt` | Python dependencies |

## Tech Stack

`Python` `Pandas` `NumPy` `scikit-learn` `LightGBM` `XGBoost` `LazyPredict` `pandas-profiling` `Streamlit` `Seaborn` / `Matplotlib`

## Running Locally

```bash
pip install -r requirements.txt
streamlit run App.py
```

## Key Findings

- In-flight service quality (entertainment, online support/booking, on-board service) drives satisfaction far more than flight logistics like distance or delay
- Business class and loyal customers show meaningfully higher satisfaction rates
- RandomForest outperformed gradient boosting methods (XGBoost, LightGBM) on this dataset while remaining efficient to train

## Possible Extensions

- Add SHAP-based explainability to the Streamlit app so predictions come with a feature-level explanation
- Replace label encoding with one-hot encoding for a cleaner treatment of nominal categories
- Package the model behind a REST API (FastAPI) instead of a script-based Streamlit app
