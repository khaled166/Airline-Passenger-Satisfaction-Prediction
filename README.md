<div align="center">

# ✈️ Airline Passenger Satisfaction Prediction

### Predicting passenger satisfaction from 129K+ survey records using ML — with a live Streamlit prediction app

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-006ACC?style=flat)
![LightGBM](https://img.shields.io/badge/-LightGBM-02569B?style=flat)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white)

**Final Model: RandomForestClassifier — 95.6% Accuracy**

</div>

---

## 📋 Overview

Given a passenger satisfaction survey (**129,880 records, 24 attributes**), this project identifies the strongest drivers of airline passenger satisfaction and builds a model to predict it for new passengers — deployed as an interactive web app.

> Dataset adapted from the airline passenger satisfaction survey by John D (Kaggle), cleaned for classification.

---

## 🔍 Exploratory Data Analysis

**Key findings:**
- 🥇 **Inflight entertainment**, **ease of online booking**, and **online support** are the strongest satisfaction drivers
- 🥈 Business class and loyal customers report meaningfully higher satisfaction
- 🔗 Departure and arrival delay are highly correlated (0.95) → merged into a single `Delay` feature
- 🗑️ `Gate location` and `Flight Distance` dropped — negligible correlation with the target

---

## ⚙️ Pipeline

```
Raw Data (129,880 rows) 
    ↓
Cleaning (drop nulls, drop id)
    ↓
EDA + Automated Profiling (pandas-profiling)
    ↓
Outlier Removal (IQR method)
    ↓
Feature Engineering (encode categoricals, merge delay features)
    ↓
AutoML Screening (LazyPredict — 29 models)
    ↓
Manual Model Comparison (5 top candidates)
    ↓
Hyperparameter Tuning (GridSearchCV)
    ↓
Final Model + Streamlit App
```

---

## 🤖 Model Comparison

Screened 29 classifiers automatically with **LazyPredict**, then compared top candidates manually with and without feature scaling:

| Model | Accuracy | F1 Score | Time Taken (s) |
|---|:---:|:---:|:---:|
| **🏆 RandomForestClassifier** | **0.96** | **0.96** | 46.5 |
| ExtraTreesClassifier | 0.96 | 0.96 | 93.6 |
| XGBClassifier | 0.95 | 0.95 | 33.3 |
| LGBMClassifier | 0.95 | 0.95 | 6.7 |
| SVC | 0.95 | 0.95 | 681.3 |
| KNeighborsClassifier | 0.93 | 0.93 | 299.8 |

**RandomForestClassifier** was selected — best accuracy, and dramatically faster than SVC or KNN for comparable performance.

### Hyperparameter Tuning (GridSearchCV, 5-fold CV)

```
Best parameters: criterion='entropy', max_depth=50, n_estimators=200
Best CV score: 0.957
```

---

## 📊 Results

| Metric | Score |
|---|:---:|
| **Accuracy** | 95.6% |
| **Precision** | 0.955 |
| **Recall** | 0.956 |
| **F1-Score** | 0.956 |

Validated with 10-fold cross-validation for robustness.

---

## 🚀 Live App

The trained model is deployed behind a **Streamlit** app — enter passenger and flight details, get an instant satisfaction prediction with a probability score.

### Run it locally

```bash
git clone https://github.com/khaled166/Airline-Passenger-Satisfaction-Prediction.git
cd Airline-Passenger-Satisfaction-Prediction
pip install -r requirements.txt
streamlit run App.py
```

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `Airline passenger satsification.ipynb` | Full pipeline: EDA, feature engineering, model comparison, tuning, evaluation |
| `App.py` | Streamlit app for live satisfaction prediction |
| `model.pkl` | Serialized trained RandomForestClassifier (bz2-compressed) |
| `Transformer.pkl` | Serialized label encoder for categorical inputs (bz2-compressed) |
| `requirements.txt` | Python dependencies |

---

## 💡 Key Takeaways

- Service-quality features (entertainment, support, booking ease) outweigh logistical factors (distance, delay) in driving satisfaction
- Tree-based ensembles (RandomForest, XGBoost, LightGBM) significantly outperformed linear models on this dataset
- Feature scaling had minimal effect on tree-based models but changed rankings for SVC and KNN

## 🔭 Possible Extensions

- [ ] Add SHAP explainability to the Streamlit app for per-prediction feature importance
- [ ] Replace label encoding with one-hot encoding for cleaner categorical treatment
- [ ] Package the model behind a FastAPI REST endpoint instead of a script-based app
- [ ] Deploy the Streamlit app publicly (Streamlit Community Cloud) and link it here

---

<div align="center">

**Built by [Khaled SeifAldin](https://github.com/khaled166)**

</div>
