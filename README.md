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
🔗 LinkedIn:https://www.linkedin.com/in/purnapriya-tammana
🔗 GitHub:https://github.com/purnapriya4448/codec_projects


📜 License
This project is open-source and available under the MIT License.

🙌 Acknowledgments
Dataset: Kaggle – Fake and Real News Dataset

Libraries: Scikit-learn, TensorFlow, HuggingFace Transformers, NLTK, Flask
