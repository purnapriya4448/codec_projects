# 📰 Fake News Detection using NLP and Machine Learning

## 📌 Overview

Fake news detection is a crucial challenge in the modern digital era. This project aims to build a machine learning pipeline that can effectively identify and classify news articles as **real** or **fake** using **Natural Language Processing (NLP)** techniques.

## 🧠 Techniques Used

- Natural Language Processing (NLP)
- Machine Learning Classifiers (SVM, Logistic Regression)
- Deep Learning (LSTM)
- Transformer-based Models (BERT)

---

## 📂 Dataset

We used a publicly available dataset containing news articles labeled as either **fake** or **real**.

- **Columns:**
  - `title`: Headline of the news article
  - `text`: Full content of the news article
  - `label`: 1 for fake, 0 for real

---

## ⚙️ Project Structure

📁 fake-news-detection/
│
├── data/
│ └── fake_or_real_news.csv
│
├── notebooks/
│ ├── EDA_and_Preprocessing.ipynb
│ ├── ML_Models_SVM_LogReg.ipynb
│ ├── LSTM_Model.ipynb
│ └── BERT_Model.ipynb
│
├── models/
│ └── saved_model.pkl
│
├── requirements.txt
├── README.md
└── app.py (optional for deployment)



---

## 🚀 Workflow

1. **Data Loading & Cleaning**
   - Handle missing values
   - Combine `title` and `text` columns

2. **Text Preprocessing**
   - Lowercasing
   - Removing punctuation and stopwords
   - Lemmatization

3. **Vectorization**
   - TF-IDF for traditional ML models
   - Tokenization for LSTM/BERT

4. **Model Building**
   - ✅ SVM and Logistic Regression
   - ✅ LSTM using Keras
   - ✅ BERT using HuggingFace Transformers

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix

---

## 📊 Results

| Model            | Accuracy |
|------------------|----------|
| Logistic Regression | 92%      |
| SVM                  | 93%      |
| LSTM                 | 95%      |
| BERT                 | 97%      |

---

## 🧪 Installation & Usage

### 1. Clone the repository
git clonehttps://github.com/purnapriya4448/codec_projects/edit/main/README.md fake-news-detection
2. Install dependencies

pip install -r requirements.txt
3. Run Jupyter Notebooks
You can run the notebooks for each model individually:

jupyter notebook notebooks/BERT_Model.ipynb
🖥️ Optional: Web App
A simple Flask app (app.py) is also provided to make predictions on custom input text

🧑‍🔬 Author
Purnapriya Tammana
Data Science Intern | NLP Enthusiast
🔗 LinkedIn:https:https://www.linkedin.com/in/purnapriya-tammana-27a0a7346
🔗 GitHub:https://github.com/purnapriya4448/codec_projects


📜 License
This project is open-source and available under the MIT License.

🙌 Acknowledgments
Dataset: Kaggle – Fake and Real News Dataset

Libraries: Scikit-learn, TensorFlow, HuggingFace Transformers, NLTK, Flask
# project 2: Customer Churn Prediction using Machine Learning

## Overview

Customer churn is when a customer stops using a company's products or services. Predicting churn helps businesses proactively retain customers and reduce revenue loss. This project aims to predict whether a customer will churn based on historical usage patterns and demographics using supervised machine learning techniques.

## Goals

- Analyze customer behavior
- Identify key churn indicators
- Build a predictive model with high accuracy
- Deploy the model for real-time predictions (optional)

---

## Dataset

We use a telecom dataset containing customer information, services used, and whether they have churned.

### Key Features:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `MonthlyCharges`, `TotalCharges`
- `Contract`, `PaymentMethod`, `InternetService`, etc.
- `Churn` (Target variable: Yes = 1, No = 0)

---

## Project Structure
📁 customer-churn-prediction/
│
├── data/
│ └── churn_data.csv
│
├── notebooks/
│ ├── EDA_and_Cleaning.ipynb
│ ├── Churn_Model_Training.ipynb
│ └── Churn_Model_Evaluation.ipynb
│
├── models/
│ └── churn_model.pkl
│
├── requirements.txt
├── README.md
└── app.py # Flask app (optional)



## Workflow

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical features (Label Encoding / One-Hot Encoding)
   - Scale numerical features

2. **Exploratory Data Analysis (EDA)**
   - Correlation matrix
   - Churn vs. categorical and numerical features

3. **Modeling**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Evaluate using Accuracy, Precision, Recall, F1-score, ROC AUC

4. **Hyperparameter Tuning**
   - GridSearchCV / RandomizedSearchCV

5. **Model Deployment (Optional)**
   - Flask app to predict churn on new customer data

---

## Results

| Model             | Accuracy | ROC AUC |
|------------------|----------|---------|
| Logistic Regression | 80%     | 0.85    |
| Random Forest        | 85%     | 0.88    |
| XGBoost              | 87%     | 0.90    |

---

## Installation

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
Run the notebook:

jupyter notebook notebooks/Churn_Model_Training.ipynb
Web App (Optional)
To run the Flask app:

python app.py
Author
Purnapriya Tammana
Data Science Intern
LinkedIn:https://www.linkedin.com/in/purnapriya-tammana-27a0a7346/
 GitHub:https://github.com/purnapriya4448/codec_projects/edit/main/README.md

License
This project is licensed under the MIT License.

Acknowledgments
Dataset: Kaggle – Telco Customer Churn

Tools: Scikit-learn, Pandas, Matplotlib, Seaborn, Flask, XGBoost


