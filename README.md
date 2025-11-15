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
**Source:** [arXiv:2306.02449](https://arxiv.org/pdf/2306.02449)

## 📘 Project Overview  
This project reproduces key parts of a research paper that compares the performance of **simple linear models** versus **complex machine learning algorithms** for **breast cancer diagnosis**.

The original study evaluates **Logistic Regression (LR)**, **Decision Trees (DT)**, and **Support Vector Machines (SVM)** using the *Breast Cancer Wisconsin Dataset*.  
The findings show that **Logistic Regression outperforms the more complex models**, offering higher accuracy, better interpretability, and faster computation.

Our project reproduces the paper’s methodology using the models covered in class:  
✔ Logistic Regression  
✔ Decision Tree  
❌ SVM (excluded, as not covered in class)

---

## 📄 Research Paper Summary  

The research concludes that **simpler models can outperform complex techniques**, especially in medical applications where interpretability and reliability are crucial.

### **Key findings from the paper:**
- **Logistic Regression**  
  - Mean Test Score: **97.28%**  
  - Std Dev: **1.62%**  
  - Computation Time: **35.56 ms**
- **Support Vector Machine:** 96.44%  
- **Decision Tree:** 93.73%

### **Conclusion of the paper**
> “In the medical domain, simplicity can outperform complexity. Logistic Regression remains an effective, interpretable, and computationally efficient choice for breast cancer diagnosis.”

---

## 📊 Dataset  
We use the **Breast Cancer Wisconsin (Original) Dataset**, sourced from the **UCI Machine Learning Repository**.

### **Features**
The dataset includes 9 numerical diagnostic features:

- Clump Thickness  
- Uniformity of Cell Size  
- Uniformity of Cell Shape  
- Marginal Adhesion  
- Single Epithelial Cell Size  
- Bare Nuclei  
- Bland Chromatin  
- Normal Nucleoli  
- Mitoses  

**Target Variable:**  
- *Class*: 2 = Benign → mapped to **0**  
- *Class*: 4 = Malignant → mapped to **1**

---

## 🛠️ Project Structure  
├── data/
│ ├── breast-cancer-wisconsin.data # Original dataset
│ └── data.csv # Cleaned CSV
│
├── models/
│ ├── logistic_regression.py
│ ├── decision_tree.py
│
├── src/
│ ├── data_load.py
│
├── index.py
├── README.md

---

## 🔄 Methodology

### ✔️ Data Preprocessing  
- Removed unnecessary `Id` column  
- Handled missing values in `Bare_Nuclei`  
- Converted class values to 0 (benign) and 1 (malignant)  
- Balanced dataset using **upsampling**  
- Scaled data for visualization  
- Generated:
  - Parallel Coordinates Plot  
  - KDE Feature Distributions  

### ✔️ Models Used

#### **Logistic Regression**
LogisticRegression(max_iter=1000, C=1.0)

#### Decision Tree
DecisionTreeClassifier(random_state=42)

### **Evaluation Metrics**
- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## ▶️ How to Run

### **1. Clone repository**
```bash
git clone https://github.com/<your-repo-name>/breast-cancer-classification.git
cd breast-cancer-classification
```
```bash
2. Install required libraries
pip install -r requirements.txt
```
```bash
3. (Optional) Convert dataset
convert_to_csv()
```

```bash
4. Run
python index.py 
```
