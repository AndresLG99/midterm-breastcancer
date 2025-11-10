# 🧠 Breast Cancer Classification using Supervised Learning  
### Midterm Project — Reproduction of a Research Paper

---

## 👥 Team Members
- Aaron Bracho
- Andres Lopez  
- Lucas Nishimoto  
- Melissa Osorio

---

## 📘 Research Paper
**Title:** *The Power of Simplicity: Why Simple Linear Models Outperform Complex Machine Learning Techniques – Case of Breast Cancer Diagnosis*  
**Authors:** Arshad, Shahriar & Anjum (2023)  
**Source:** [arXiv:2306.02449](https://arxiv.org/abs/2306.02449)

### **Summary**
This research paper compares the performance of simple supervised learning algorithms — **Logistic Regression**, **Decision Trees**, and **K-Nearest Neighbors (KNN)** — on the Breast Cancer Wisconsin (Diagnostic) dataset.  

The main insight is that **simpler, interpretable models can perform as well as or better than complex ones** when paired with proper preprocessing and scaling.  

This project reproduces the study using only algorithms and techniques covered in our Data Science course.

---

## 🎯 Objective
To reproduce the experiments from the paper using:
- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  

and to apply:
- Feature Scaling  
- PCA (Principal Component Analysis) for dimensionality reduction  
- Cross-Validation for performance evaluation  

---

## 📊 Dataset
**Name:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
**Instances:** 569  
**Features:** 30 numeric features + 1 target variable  
**Target Variable:**
- `0` → Malignant (cancerous)  
- `1` → Benign (non-cancerous)

### **Main Features**
- Mean radius  
- Mean texture  
- Mean perimeter  
- Mean area  
- Mean smoothness  
- Mean concavity, symmetry, and more  

### **Dataset Notes**
- No missing values  
- All features are continuous  
- Moderate class imbalance (212 malignant / 357 benign)

---

## ⚙️ Methodology

### **1. Data Preprocessing**
- Load the dataset with `pandas` or `scikit-learn` built-in loader  
- Check for missing or duplicate values (none expected)  
- Apply **StandardScaler** for normalization  
- Split into **70% training** and **30% testing** with stratification  

### **2. Feature Extraction (PCA)**
- Use **PCA** to retain 95% of total variance  
- Compare model performance **with vs without PCA**

### **3. Classification Models**
| Algorithm | Scikit-learn Class | Notes |
|------------|--------------------|--------|
| Logistic Regression | `LogisticRegression()` | Baseline linear classifier |
| Decision Tree | `DecisionTreeClassifier()` | Tree-based model with interpretable structure |
| K-Nearest Neighbors | `KNeighborsClassifier()` | Distance-based non-parametric method |

Each model is evaluated using both a **holdout (70/30)** split and **10-fold cross-validation**.

### **4. Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC (optional, if covered in class)

Visual outputs include:
- Confusion Matrix  
- ROC Curve  
- PCA explained variance chart  

---

## 🔁 Cross-Validation
- **Method:** Stratified 10-Fold Cross Validation  
- Ensures equal class representation in every fold  
- Provides a robust performance estimate  

---

## 📁 Project Structure
```
midterm-breastcancer/
│
├─ README.md
├─ requirements.txt
├─ DS 301 - Midterm Project.pdf
│
├─ data/
│  └─ breast_cancer_wdbc.csv
│
├─ notebooks/
│  ├─ 01_EDA_and_preprocessing.ipynb
│  ├─ 02_model_training.ipynb
│  └─ 03_results_and_visualization.ipynb
│
├─ src/
│  ├─ data_load.py
│  ├─ preprocessing.py
│  ├─ train_models.py
│
├─ models/
│  ├─ logistic_regression.joblib
│  ├─ decision_tree.joblib
│  └─ knn.joblib
│
├─ results/
│  └─ metrics_summary.csv
│
└─ slides/
   └─ presentation.pdf
```

---

## 🚀 How to Run

### **1. Clone Repository**
```bash
git clone https://github.com/<your-username>/midterm-breastcancer.git
cd midterm-breastcancer
```

### **2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run Code**
You can either:
- Open and execute the notebooks in `/notebooks`, **or**
- Run the Python script directly:
```bash
python src/train_models.py
```

### **5. View Results**
- Final metrics: `results/metrics_summary.csv`  
- Saved models: `models/`  

---

## 📈 Expected Results

| Model | Accuracy | Precision | Recall | F1-score |
|--------|-----------|------------|----------|-----------|
| Logistic Regression | ~97% | ~97% | ~97% | ~97% |
| Decision Tree | ~94% | ~94% | ~93% | ~93% |
| KNN | ~96% | ~96% | ~96% | ~96% |

*(Values may vary slightly based on random state or PCA settings.)*

---

## 💡 Suggested Improvements
- Add **Lasso/Ridge Logistic Regression** for regularization  
- Use **GridSearchCV** for hyperparameter tuning  
- Explore **LDA** for feature extraction  
- Apply **ensemble averaging** or **bagging** (as future work)  

---

## 🧰 Tools & Libraries
- Python 3.13  
- pandas  
- numpy  
- matplotlib / seaborn  
- scikit-learn  
- joblib  

---

## 🧾 References
- Arshad, Shahriar & Anjum (2023). *The Power of Simplicity: Why Simple Linear Models Outperform Complex Machine Learning Techniques – Case of Breast Cancer Diagnosis.* [arXiv:2306.02449](https://arxiv.org/abs/2306.02449)  
- UCI Machine Learning Repository: *Breast Cancer Wisconsin (Diagnostic)* dataset.  
- scikit-learn documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

---

## 🎓 Course Topics Applied
✅ Decision Trees  
✅ K-Nearest Neighbors (KNN)  
✅ Classification & Logistic Regression  
✅ Lasso & Ridge Regression *(suggested improvement)*  
✅ Cross Validation Techniques  
✅ Multiple Linear Regression *(theoretical foundation)*  
✅ PCA (Feature Extraction)  
✅ Feature Scaling Techniques  
✅ Data Preprocessing  

---

## 📬 Contact
For questions or collaboration:  
**Aaron Bracho** — abracho7@gmail.com

**Andres Lopez** — ing.andlopgam5@gmail.com

**Lucas Nishimoto** — luyuki2001@gmail.com

**Melissa Osorio** — melissaosorio851@gmail.com

---

> *"Simplicity is the ultimate sophistication." – Leonardo da Vinci*
