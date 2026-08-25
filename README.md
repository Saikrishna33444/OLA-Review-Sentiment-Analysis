# 🚕 OLA Review Sentiment Analysis

### NLP + Machine Learning | Customer Review Classification

> An end-to-end Natural Language Processing project that analyzes OLA customer reviews and classifies them into **Negative, Neutral, or Positive** sentiment using **TF-IDF and Machine Learning**.

---

## 📌 Problem Statement

OLA receives a large volume of customer reviews. Manually analyzing these reviews is time-consuming.

This project aims to automatically analyze customer feedback and classify each review into:

🔴 **Negative** | 🟡 **Neutral** | 🟢 **Positive**

This can help identify customer dissatisfaction, understand customer experience, and analyze feedback at scale.

---

## 🎯 Objective

Build a machine-learning system that:

* Processes raw customer reviews
* Converts ratings into sentiment labels
* Cleans and preprocesses text
* Converts text into numerical features
* Compares multiple ML algorithms
* Selects the best-performing model
* Predicts sentiment for new reviews

---

## 🔄 Project Workflow

```text
                OLA Customer Reviews
                         │
                         ▼
                  Data Collection
                         │
                         ▼
                  Data Exploration
                         │
                         ▼
              Rating → Sentiment Label
                         │
                         ▼
                 Text Preprocessing
                         │
                         ▼
                    Train / Test
                         │
                         ▼
                      TF-IDF
                         │
                         ▼
                  Model Comparison
                         │
                         ▼
                Logistic Regression
                         │
                         ▼
                  GridSearchCV
                         │
                         ▼
                    Evaluation
                         │
                         ▼
                 Sentiment Prediction
```

---

## 📊 Dataset

The dataset contains **10,000 OLA customer reviews**.

Sentiment is derived from the customer rating:

| Rating |  Sentiment  |
| :----: | :---------: |
|  ⭐ 1–2 | 🔴 Negative |
|   ⭐ 3  |  🟡 Neutral |
|  ⭐ 4–5 | 🟢 Positive |

### Class Distribution

| Sentiment   | Reviews | Percentage |
| ----------- | ------: | ---------: |
| 🔴 Negative |   7,372 |     73.72% |
| 🟢 Positive |   2,245 |     22.45% |
| 🟡 Neutral  |     383 |      3.83% |

### ⚠️ Important

The sentiment labels are **derived from ratings**, rather than manually annotated from the review text. Therefore, they are treated as a proxy for sentiment.

---

# 🔍 Exploratory Data Analysis

Before building the model, the dataset is analyzed for:

* Missing values
* Duplicate records
* Data types
* Rating distribution
* Sentiment distribution
* Class imbalance
* Review characteristics

### Key Finding

The dataset is highly imbalanced, with **Negative reviews representing 73.72%** of the data and **Neutral reviews only 3.83%**.

This is important when evaluating model performance.

---

# 🧹 Text Preprocessing

Raw review text is cleaned before applying machine learning.

### Processing Pipeline

```text
Raw Review
    ↓
Lowercase
    ↓
Remove unwanted characters/noise
    ↓
Stopword handling
    ↓
Clean Review
```

The purpose is to reduce unnecessary variation and noise in the text before feature extraction.

---

# 🔢 TF-IDF Feature Extraction

Machine-learning algorithms cannot directly work with raw text.

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert reviews into numerical feature vectors.

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
```

### Why TF-IDF?

TF-IDF considers:

* How frequently a word occurs in a review
* How common that word is across all reviews

This allows informative words to receive greater importance.

### Unigrams + Bigrams

```text
Unigrams:
good
driver
service

Bigrams:
very good
bad service
not good
```

Using `(1,2)` allows the model to capture both individual words and two-word combinations.

---

# ✂️ Train/Test Split

The data is divided into:

```text
80% → Training
20% → Testing
```

The split uses:

```python
random_state=42
stratify=y
```

### `random_state`

Ensures reproducibility.

### `stratify=y`

Maintains the class distribution between training and testing datasets.

---

# 🤖 Machine Learning Models

Multiple classification algorithms were evaluated:

| Model               |
| ------------------- |
| K-Nearest Neighbors |
| Naive Bayes         |
| Logistic Regression |
| Linear SVM          |
| Decision Tree       |
| Random Forest       |

The models were compared using their classification performance.

> **Ridge Classifier was removed from the final project.**

---

# 🏆 Final Model — Logistic Regression

**Logistic Regression** was selected as the final model because it performed strongly on the TF-IDF representation and is well suited to high-dimensional sparse text features.

The final model also uses:

```python
class_weight="balanced"
```

to give greater importance to minority classes.

---

# ⚙️ Hyperparameter Tuning

After selecting Logistic Regression, **GridSearchCV** was used for hyperparameter tuning.

### Why GridSearchCV?

It evaluates different hyperparameter combinations using cross-validation and identifies a strong configuration for the model.

Parameters considered include:

```text
C
solver
max_iter
class_weight
```

---

# 📈 Model Evaluation

The final model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Weighted F1
* Confusion Matrix

### Final Results

| Metric            |      Score |
| ----------------- | ---------: |
| Training Accuracy | **93.61%** |
| Test Accuracy     | **89.00%** |
| Macro F1          |   **0.64** |
| Weighted F1       |   **0.89** |

---

# 🧩 Confusion Matrix

```text
                 Predicted
              Neg   Neu   Pos

Actual Neg    1404   38    32
Actual Neu      61    9     7
Actual Pos      64   18   367
```

### Key Observation

The model performs considerably better on **Negative and Positive** reviews than on **Neutral** reviews.

The main reason is the very small number of Neutral examples in the dataset.

Therefore:

> **89% accuracy should not be considered the only indicator of model quality.**

Macro F1 and class-level precision/recall are also important.

---

# 🔮 Prediction Pipeline

Once the model is trained, a new review follows this process:

```text
New Customer Review
        ↓
Text Cleaning
        ↓
Trained TF-IDF Vectorizer
        ↓
Logistic Regression
        ↓
┌──────────┬─────────┬──────────┐
│ Negative │ Neutral │ Positive │
└──────────┴─────────┴──────────┘
```

The **same trained TF-IDF vectorizer** is used for new reviews.

---

# 💾 Model Persistence

The trained components can be saved and reused:

```text
final_model.pkl
tfidf_vectorizer.pkl
```

Both are required because:

* **TF-IDF Vectorizer** → converts new text into numerical features
* **ML Model** → predicts sentiment from those features

---

# 🛠️ Tech Stack

| Technology             | Purpose                 |
| ---------------------- | ----------------------- |
| 🐍 Python              | Programming             |
| 🐼 Pandas              | Data manipulation       |
| 🔢 NumPy               | Numerical operations    |
| 📝 NLTK                | NLP preprocessing       |
| 🤖 Scikit-learn        | Machine Learning        |
| 📊 Matplotlib          | Visualization           |
| 📊 Seaborn             | Visualization           |
| 🔤 TF-IDF              | Text feature extraction |
| 🧠 Logistic Regression | Final classifier        |
| 🔎 GridSearchCV        | Hyperparameter tuning   |
| 💾 Pickle              | Model persistence       |
| 📓 Jupyter / Colab     | Development             |

---

# ⚠️ Limitations

* Sentiment labels are derived from ratings.
* Dataset is highly imbalanced.
* Neutral-class prediction is weak.
* TF-IDF has limited understanding of context.
* Sarcasm and mixed sentiment can be difficult to classify.
* Performance on this dataset may not represent performance on future real-world reviews.

---

# 🚀 Future Improvements

* Improve Neutral-class representation
* Use better sentiment annotations
* Perform detailed error analysis
* Experiment with additional imbalance-handling techniques
* Use a scikit-learn Pipeline
* Compare with Word2Vec/embeddings
* Experiment with transformer models such as BERT
* Evaluate the model on newer customer reviews

---

# 📁 Project Structure

```text
OLA-Review-Sentiment-Analysis/
│
├── README.md
├── ola_review_sentiment_analysis.ipynb
│
├── data/
│   └── ola_review_dataset.csv
│
├── models/
│   ├── final_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
└── .gitignore
```

---

# 🎤 Project Summary

> **"I developed an OLA customer review sentiment analysis system using NLP and machine learning. I first explored and cleaned the review dataset and converted customer ratings into Negative, Neutral, and Positive labels. I then preprocessed the review text and used TF-IDF with unigrams and bigrams to convert text into numerical features. I compared multiple machine-learning algorithms and selected Logistic Regression based on its performance with the TF-IDF features. I handled class imbalance using balanced class weights and tuned the model using GridSearchCV. The final model achieved 89% test accuracy, but the evaluation showed that the Neutral class remains challenging due to the highly imbalanced dataset."**

---

## ⭐ Key Takeaway

**Raw Reviews → NLP Preprocessing → TF-IDF → ML Models → Logistic Regression → Hyperparameter Tuning → Evaluation → Sentiment Prediction**
