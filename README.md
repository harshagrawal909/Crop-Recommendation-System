# 🌾 Crop Recommendation System for Farmers

A machine learning-based system that recommends the most suitable crop for farmers based on soil nutrients and environmental conditions. Built as part of a B.Tech Computer Science project at **KIIT Deemed to be University**, March 2026.

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| Harsh Agrawal | 23053476 |
| Ankit Yadav | 23053796 |
| Tushal Chauhan | 23053798 |
| Kaustuv Dhungel | 23053549 |

---

## 📌 Overview

Traditional crop selection relies on farmer experience and local knowledge, which often leads to sub-optimal results. This system uses supervised machine learning to recommend the best crop based on real soil and climate data, helping farmers make smarter, data-driven decisions.

Three ML algorithms are trained and compared:
- ✅ **XGBoost** — 99% accuracy *(best model, selected for deployment)*
- 🌲 **Random Forest** — 98% accuracy
- 📐 **SVM (RBF Kernel)** — 98% accuracy

---

## 📂 Project Structure

```
Crop-Recommendation-System/
│
├── data/
│   └── crop_data.csv              # Dataset with 2,200 records (N, P, K, Temp, Humidity, pH, Rainfall, Crop)
│
├── notebooks/
│   └── analysis.ipynb             # EDA, preprocessing, model training & evaluation
│
├── src/
│   ├── preprocessing.py           # Data cleaning, encoding, and feature scaling
│   ├── train.py                   # Model training (Random Forest, XGBoost, SVM)
│   └── predict.py                 # Load model and predict crop from input parameters
│
├── models/
│   └── xgboost_model.pkl          # Saved best-performing XGBoost model
│
├── app/
│   └── app.py                     # (Optional) Streamlit/Flask web app for crop prediction
│
├── results/
│   ├── accuracy.png               # Model accuracy comparison chart
│   └── confusion_matrix.png       # Confusion matrix for best model
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 📊 Dataset

- **Source:** [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Size:** 2,200 records × 8 features
- **Target:** 22 crop classes

| Feature | Description | Type |
|---------|-------------|------|
| N | Nitrogen content in soil (kg/ha) | Numerical |
| P | Phosphorus content in soil (kg/ha) | Numerical |
| K | Potassium content in soil (kg/ha) | Numerical |
| Temperature | Average ambient temperature (°C) | Numerical |
| Humidity | Relative humidity (%) | Numerical |
| pH | Soil pH value | Numerical |
| Rainfall | Rainfall in mm | Numerical |
| Crop Label | Target crop class | Categorical |

---

## ⚙️ System Pipeline

```
User Input (N, P, K, Temp, Humidity, pH, Rainfall)
        ↓
Data Preprocessing (Missing values, Label Encoding, StandardScaler)
        ↓
Feature Engineering & Crop Categorization
        ↓
ML Model Training (Random Forest / XGBoost / SVM)
        ↓
Stacking Ensemble (Meta-Learner combines predictions)
        ↓
Final Crop Recommendation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/harshagrawal909/Crop-Recommendation-System.git
cd Crop-Recommendation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run preprocessing and training

```bash
python src/preprocessing.py
python src/train.py
```

### 4. Predict a crop

```bash
python src/predict.py
```

### 5. (Optional) Launch the web app

```bash
streamlit run app/app.py
```

---

## 🧪 Example Prediction

| Parameter | Value |
|-----------|-------|
| N (Nitrogen) | 90 |
| P (Phosphorus) | 42 |
| K (Potassium) | 43 |
| Temperature | 21°C |
| Humidity | 82% |
| pH | 6.5 |
| Rainfall | 200 mm |
| **Recommended Crop** | 🌾 **Rice** |

---

## 📈 Model Results

| Model | Accuracy | Technique |
|-------|----------|-----------|
| **XGBoost** | **99%** | Gradient Boosting |
| Random Forest | 98% | Bagging Ensemble |
| SVM (RBF Kernel) | 98% | Margin-based Classifier |

XGBoost was selected as the final deployment model due to its highest accuracy and strong generalization.

---

## 📚 References

1. Pudumalar et al. (2017). *Crop recommendation system for precision agriculture.* IEEE ICoAC.
2. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
4. Cortes & Vapnik (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.
5. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.

---

## 📄 License

This project was developed for academic purposes at KIIT Deemed to be University, March 2026.
