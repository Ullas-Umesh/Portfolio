# 🧠 Fetal Health Classification Project

## 👤 Author
**Ullas Umesh**  
MSc Data Science  
Student ID: N1221317  
Date: 20/05/2024

---

## 📌 Introduction
Cardiotocography (CTG) is widely used to monitor fetal well-being. This project uses CTG-derived features to classify fetal health into three categories:  
- **Normal**  
- **Suspect**  
- **Pathological**

## 🎯 Project Goal
To train, evaluate, and deploy machine learning models to predict fetal health, and identify the most accurate classifier for deployment in Azure.

---

## 🧬 Dataset Overview
- Source: [Kaggle - Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
- 2126 samples, 21 features + 1 target variable (`fetal_health`)
- Health classes:
  - `1`: Normal
  - `2`: Suspect
  - `3`: Pathological

---

## 🧹 Data Preprocessing
- **Outliers removed** using Z-score > 3
- **StandardScaler** used for feature scaling
- No missing values
- `health_status` column added for interpretability
- Final dataset: 1719 rows

---

## 📊 Exploratory Data Analysis (EDA)
- Histograms plotted for all features
- Correlation heatmap confirmed strong inter-feature relationships
- No features were dropped

---

## 🤖 Models Implemented

### 1. Support Vector Machine (SVM)
- Accuracy: **0.8856**

### 2. Multilayer Perceptron (MLP)
- Accuracy: **0.8449**
- Model convergence confirmed

### 3. Convolutional Neural Network (CNN)
- Architecture: 3 Conv1D + MaxPooling + Dense layers
- Accuracy: **0.8900**

### 4. Random Forest
- Accuracy: **0.9476**

### 5. XGBoost
- Accuracy: **0.9500**

### 🔍 Hyperparameter Tuning
- Used GridSearchCV
- Tuned XGBoost accuracy: **0.9600**

---

## 🔗 Unsupervised Learning (Clustering)

| Model                   | Accuracy |
|------------------------|----------|
| Gaussian Mixture Model | 0.32     |
| KMeans                 | 0.38     |
| Agglomerative Clustering | 0.54   |

---

## ☁️ Azure Deployment
- AutoML used on Azure ML Studio
- Deployed XGBoost model as REST API
- Accuracy from Azure deployment: **0.9476**
- Endpoint: *(hidden for privacy)*

---

## ⚖️ Ethical & Social Considerations
- **Privacy & Security**: Data compliance (GDPR, HIPAA)
- **Fairness**: Avoiding bias in healthcare predictions
- **Transparency**: Use of explainable AI techniques
- **Accountability**: Shared responsibility in clinical use

---

## ✅ Conclusion
- **CNN** and **XGBoost** delivered top performance
- **XGBoost (tuned)** was deployed using Azure ML
- Project shows promise for real-time fetal health prediction in clinical settings

---

## 📚 References
- Kaggle CTG Dataset  
- IEEE & Elsevier papers on ML, CNN, and ethical AI  
