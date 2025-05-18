# 🧠 Fetal Health Classification Project


## 📌 Introduction
Cardiotocography (CTG) is widely used to monitor fetal well-being. This project uses CTG-derived features to classify fetal health into three categories:  
- **Normal**  
- **Suspect**  
- **Pathological**

---

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

```python
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Remove outliers
z_scores = np.abs(stats.zscore(data_file.drop('fetal_health', axis=1)))
data_file = data_file[(z_scores < 3).all(axis=1)]

# Convert label to int
data_file['fetal_health'] = data_file['fetal_health'].astype('int')

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_file.drop('fetal_health', axis=1))
```

---

## 📊 Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt

# Plot histograms for each feature
data_file.hist(figsize=(20, 15), bins=20, color='skyblue')
plt.show()
```

```python
# Correlation heatmap
import seaborn as sns
plt.figure(figsize=(18, 10))
sns.heatmap(data_file.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
```

---

## 🤖 Models Implemented

### 1. Support Vector Machine (SVM)
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm = SVC(kernel='rbf', C=3.0, gamma='scale')
svm.fit(X_train, y_train)
accuracy_score(y_test, svm.predict(X_test))
```

### 2. Multilayer Perceptron (MLP)
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(25, 18, 10, 5), max_iter=600, alpha=5.0)
mlp.fit(X_train, y_train)
```

### 3. Convolutional Neural Network (CNN)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

model = Sequential([
    Conv1D(16, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
```

### 4. Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
```

### 5. XGBoost
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
xgb.fit(X_train, y_train)
```

---

## 🔍 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1],
    'max_depth': [3, 6],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)
```

---

## 🔗 Unsupervised Learning (Clustering)

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)
kmeans = KMeans(n_clusters=3)
agg = AgglomerativeClustering(n_clusters=3)
```

---

## ☁️ Azure Deployment

```python
# Sending JSON to Azure REST API
import urllib.request
import json

data = {
    "Inputs": {
        "data": [
            {
                "baseline value": 150,
                ...
            }
        ]
    },
    "GlobalParameters": {
        "method": "predict"
    }
}

body = str.encode(json.dumps(data))
headers = {'Content-Type':'application/json', 'Authorization':('Bearer YOUR_API_KEY')}
req = urllib.request.Request('https://<azure-endpoint>', body, headers)
response = urllib.request.urlopen(req)
```

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
