# ❤️ Predicting Heart Disease with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)  

## 📌 Project Overview
This project builds machine learning models to **predict heart disease** from clinical and demographic patient data.  
The workflow includes:
- Data preprocessing & cleaning  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Model training & evaluation  
- Visualization of results  

## ⚙️ Tech Stack
- **Python 3.8+**
- **NumPy, pandas** → data manipulation  
- **Matplotlib, Seaborn** → visualization  
- **Scikit-learn** → machine learning models & evaluation  
- **Jupyter Notebook** → interactive experimentation  

## 📂 Dataset
Dataset: `data/heart-disease.csv`  

**Features include**:  
- Age, sex, chest pain type  
- Resting blood pressure, cholesterol  
- Maximum heart rate achieved  
- Exercise-induced angina, ST depression, slope, etc.  

**Target variable**:  
- `0` → No Heart Disease  
- `1` → Heart Disease Present  

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/heart-disease-ml.git
cd heart-disease-ml
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Jupyter Notebook
```bash
jupyter notebook heart_disease.ipynb
```

## 📊 Models Implemented
- Logistic Regression  
- Random Forest  
- K-Nearest Neighbors (KNN)   

**Evaluation Metrics**:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

## 📈 Visualizations
- Correlation heatmap of features  
- Feature distributions  
- ROC & Precision-Recall curves  

## ✅ Results
- Random Forest and Logistic Regression achieved the highest performance.  
- Proper scaling and tuning significantly improved results.  

## 🛠️ Future Work
- Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV)  
- Try deep learning models (TensorFlow/Keras)  
- Deploy as a web app with **Streamlit/Flask**
  
---

⭐ If you find this project useful, consider giving it a **star** on GitHub!
