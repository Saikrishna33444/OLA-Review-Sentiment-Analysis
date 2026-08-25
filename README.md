# OLA Review Sentiment Analysis

## 📌 Project Overview

**OLA Review Sentiment Analysis** is a Natural Language Processing (NLP)
and Machine Learning project that analyzes customer reviews and
classifies them into three sentiment categories:

-   🔴 **Negative**
-   🟡 **Neutral**
-   🟢 **Positive**

The project converts customer ratings into sentiment labels,
preprocesses review text, transforms text into numerical features using
**TF-IDF**, compares multiple machine-learning algorithms, and selects
**Logistic Regression** as the final model.

The goal is to demonstrate an end-to-end classical NLP and
machine-learning workflow for customer feedback analysis.

------------------------------------------------------------------------

## 🎯 Problem Statement

Ride-hailing platforms receive a large number of customer reviews.
Manually reading and analyzing these reviews is time-consuming.

This project aims to automatically determine the sentiment of a customer
review so that large amounts of feedback can be analyzed more
efficiently.

### Business Use Cases

-   Identify negative customer experiences
-   Monitor customer satisfaction
-   Summarize large volumes of reviews
-   Support customer-service teams
-   Identify areas that may require improvement

------------------------------------------------------------------------

## 🧠 Project Workflow

``` text
Customer Reviews
       ↓
Data Loading
       ↓
Data Cleaning & EDA
       ↓
Rating → Sentiment Label
       ↓
Text Preprocessing
       ↓
Train/Test Split
       ↓
TF-IDF Feature Extraction
       ↓
Model Training
       ↓
Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Final Logistic Regression Model
       ↓
Evaluation
       ↓
New Review Prediction
```

------------------------------------------------------------------------

## 📊 Dataset

The project uses an OLA customer-review dataset containing **10,000
reviews**.

The sentiment labels are derived from the numerical rating.

### Sentiment Mapping

  Rating   Sentiment
  -------- -----------
  1--2     Negative
  3        Neutral
  4--5     Positive

### Class Distribution

  Sentiment     Count   Percentage
  ----------- ------- ------------
  Negative      7,372       73.72%
  Positive      2,245       22.45%
  Neutral         383        3.83%

### Important Dataset Limitation

The sentiment labels are **rating-based labels**, not manually annotated
sentiment labels.

For example, a 5-star review can still contain a complaint, and a 3-star
review can contain mixed sentiment. Therefore, the labels should be
considered a proxy for sentiment rather than perfect ground truth.

------------------------------------------------------------------------

# 🔎 Exploratory Data Analysis

Before training the model, the dataset is examined to understand:

-   Dataset shape
-   Column names and data types
-   Missing values
-   Duplicate records
-   Rating distribution
-   Sentiment distribution
-   Review characteristics

EDA is important because machine-learning performance depends heavily on
the quality and distribution of the input data.

------------------------------------------------------------------------

# 🧹 Text Preprocessing

Raw customer reviews cannot be directly provided to traditional
machine-learning algorithms.

The text is cleaned before feature extraction.

Typical preprocessing includes:

1.  Convert text to lowercase
2.  Remove unnecessary characters
3.  Remove unwanted punctuation/noise
4.  Remove stopwords where appropriate
5.  Normalize the review text

### Why preprocessing?

For example:

``` text
"I LOVE this Ride!!!"
```

can be normalized to something closer to:

``` text
"love ride"
```

This reduces unnecessary variation in the text representation.

### Important consideration

Stopword removal should be used carefully in sentiment analysis because
words such as **"not"** can affect sentiment.

------------------------------------------------------------------------

# 🔢 TF-IDF Feature Extraction

Machine-learning algorithms require numerical input, but reviews are
text.

The project uses **TF-IDF (Term Frequency--Inverse Document Frequency)**
to convert reviews into numerical feature vectors.

``` python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
```

## What is TF-IDF?

TF-IDF gives higher importance to terms that are useful for a particular
document while reducing the importance of terms that occur very
frequently across many documents.

### TF --- Term Frequency

Measures how frequently a word appears in a document.

### IDF --- Inverse Document Frequency

Reduces the weight of words that occur in many documents.

### Why TF-IDF?

Compared with simple word counts, TF-IDF considers both:

-   How frequently a word occurs in a review
-   How common the word is across the complete collection of reviews

------------------------------------------------------------------------

## 🔤 Unigrams and Bigrams

The project uses:

``` python
ngram_range=(1, 2)
```

This means the vectorizer considers both:

### Unigrams

``` text
good
driver
service
```

### Bigrams

``` text
very good
bad service
not good
```

Bigrams are useful because combinations of words can carry important
sentiment information.

For example:

``` text
good
not good
```

do not have the same meaning.

------------------------------------------------------------------------

# ✂️ Train/Test Split

The dataset is divided into training and testing sets.

``` python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### `test_size=0.2`

20% of the data is reserved for testing.

### `random_state=42`

Makes the split reproducible.

### `stratify=y`

Maintains approximately the same class distribution in both training and
testing datasets.

This is especially important because the dataset is imbalanced.

------------------------------------------------------------------------

# ⚠️ Avoiding Data Leakage

The TF-IDF vectorizer is fitted only on the training data:

``` python
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

The test data is transformed using the vocabulary and IDF information
learned from the training data.

We should **not** fit TF-IDF on the complete dataset before the
train/test split because that can introduce information from the test
set into the feature-building process.

------------------------------------------------------------------------

# 🤖 Machine Learning Models

Several classical machine-learning algorithms were compared.

The project evaluates models including:

-   K-Nearest Neighbors (KNN)
-   Naive Bayes
-   Logistic Regression
-   Linear Support Vector Machine
-   Decision Tree
-   Random Forest

The model comparison showed that **Logistic Regression** was the
strongest choice for the final model in this project.

> Ridge Classifier was removed from the project and is not part of the
> final model comparison.

------------------------------------------------------------------------

# 🏆 Why Logistic Regression?

Logistic Regression is a strong choice for text classification because:

-   It works well with high-dimensional feature spaces
-   It works effectively with sparse TF-IDF representations
-   It is computationally efficient
-   It is relatively interpretable
-   It performed strongly compared with the tested alternatives

Although the name contains "Regression", Logistic Regression is commonly
used for classification problems.

------------------------------------------------------------------------

# ⚖️ Handling Class Imbalance

The dataset contains many more Negative reviews than Neutral reviews.

To reduce the tendency of the classifier to favor the majority class,
the final Logistic Regression model uses:

``` python
class_weight="balanced"
```

This gives greater importance to minority classes during training.

However, class weighting does not completely solve the problem.

The Neutral class remains difficult to predict accurately.

------------------------------------------------------------------------

# 🔧 Hyperparameter Tuning

The project uses **GridSearchCV** to tune the Logistic Regression model.

GridSearchCV tests different combinations of hyperparameters using
cross-validation and selects the best configuration according to the
chosen scoring criterion.

### Hyperparameters can include

-   `C`
-   `solver`
-   `max_iter`
-   `class_weight`

### Parameter vs Hyperparameter

**Parameter:** learned by the model from training data.

Example:

``` text
Model coefficients
```

**Hyperparameter:** configured before/during training and selected
through experimentation.

Examples:

``` text
C
solver
max_iter
```

------------------------------------------------------------------------

# 📈 Model Evaluation

The model is evaluated using more than accuracy.

Important metrics include:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Macro F1
-   Weighted F1
-   Confusion Matrix

## Why accuracy alone is not enough

The dataset is highly imbalanced.

A model can achieve high accuracy by performing well on the majority
Negative class while performing poorly on the minority Neutral class.

Therefore, class-level metrics and **Macro F1** are important.

------------------------------------------------------------------------

# 📊 Final Model Results

The final Logistic Regression model achieved approximately:

  Metric                Result
  ------------------- --------
  Training Accuracy     93.61%
  Test Accuracy         89.00%
  Macro F1                0.64
  Weighted F1             0.89

The class-level results show that Negative and Positive reviews are
predicted much better than Neutral reviews.

### Key observation

The Neutral class has a much lower F1-score than the other classes.

This means that **89% accuracy should not be interpreted as perfect
sentiment classification**.

The Neutral class is the main area for improvement.

------------------------------------------------------------------------

# 🧩 Confusion Matrix

The final confusion matrix is approximately:

``` text
[[1404,  38,  32],
 [  61,   9,   7],
 [  64,  18, 367]]
```

Rows represent actual classes and columns represent predicted classes.

The Neutral class has relatively few correct predictions compared with
the other classes, demonstrating the difficulty caused by the small
number of Neutral examples.

------------------------------------------------------------------------

# 🧪 New Review Prediction

After training, the model can predict sentiment for a new customer
review.

The prediction process is:

``` text
New Review
    ↓
Text Cleaning
    ↓
Trained TF-IDF Vectorizer
    ↓
Logistic Regression
    ↓
Negative / Neutral / Positive
```

The trained TF-IDF vectorizer must be reused during prediction.

A new vectorizer should **not** be fitted on every new review.

------------------------------------------------------------------------

# 💾 Model Persistence

The trained model and TF-IDF vectorizer are saved so they can be reused
later without retraining.

The saved components are conceptually:

``` text
final_model.pkl
tfidf_vectorizer.pkl
```

Both are required:

-   The TF-IDF vectorizer converts new text into the expected numerical
    representation.
-   The trained Logistic Regression model predicts the sentiment from
    that representation.

------------------------------------------------------------------------

# 🚀 Recommended Production Improvement

For a cleaner production implementation, the preprocessing and model can
be combined into a single scikit-learn `Pipeline`:

``` python
Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])
```

This makes the complete transformation and prediction workflow easier to
reproduce and reduces the risk of preprocessing inconsistencies.

Other possible improvements include:

-   Better handling of the minority Neutral class
-   More extensive hyperparameter tuning
-   Error analysis
-   Manual sentiment annotation for better ground truth
-   Testing on newer/unseen reviews
-   Comparing TF-IDF with modern transformer-based approaches such as
    BERT
-   Monitoring model performance after deployment

------------------------------------------------------------------------

# ⚠️ Project Limitations

1.  **Rating-based sentiment labels** are only a proxy for actual
    textual sentiment.
2.  The dataset is **highly imbalanced**.
3.  The Neutral class has significantly weaker performance.
4.  TF-IDF does not fully understand word order or long-range context.
5.  Sarcasm and mixed sentiment can be difficult for the model.
6.  The model is based on classical NLP rather than a transformer-based
    language model.
7.  Test performance on this dataset does not guarantee performance on
    future real-world reviews.

------------------------------------------------------------------------

# 🛠️ Technologies Used

-   **Python**
-   **Pandas** --- data manipulation
-   **NumPy** --- numerical operations
-   **Matplotlib / Seaborn** --- visualization
-   **NLTK** --- NLP preprocessing
-   **Scikit-learn** --- machine learning and evaluation
-   **TF-IDF** --- text feature extraction
-   **Logistic Regression** --- final classifier
-   **GridSearchCV** --- hyperparameter tuning
-   **Pickle** --- model persistence
-   **Jupyter Notebook / Google Colab** --- development environment

------------------------------------------------------------------------

# 📁 Suggested Repository Structure

``` text
ola-review-sentiment-analysis/
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

Do not commit private credentials, API keys, or unnecessarily large
generated files.

------------------------------------------------------------------------

# ▶️ How to Run

## 1. Clone the repository

``` bash
git clone <your-repository-url>
cd ola-review-sentiment-analysis
```

## 2. Install dependencies

``` bash
pip install -r requirements.txt
```

## 3. Open the notebook

``` bash
jupyter notebook
```

Then open:

``` text
ola_review_sentiment_analysis.ipynb
```

Run the cells in order.

------------------------------------------------------------------------

# 💡 Example

Input:

``` text
"The driver was very polite and the ride was excellent."
```

Possible output:

``` text
Positive
```

Input:

``` text
"Driver was late and the service was terrible."
```

Possible output:

``` text
Negative
```

------------------------------------------------------------------------

# 🎤 Interview Explanation

A concise explanation of the project:

> "I built an OLA customer review sentiment analysis system using NLP
> and classical machine learning. I first explored and cleaned the
> review dataset, then converted ratings into Negative, Neutral, and
> Positive sentiment labels. Because machine-learning models require
> numerical features, I used TF-IDF with unigrams and bigrams to
> represent the review text. I compared multiple classification
> algorithms and selected Logistic Regression because it performed
> strongly on the high-dimensional sparse TF-IDF features. I then used
> GridSearchCV for hyperparameter tuning and evaluated the final model
> using accuracy, precision, recall, F1-score, and a confusion matrix.
> One important finding was that the dataset is highly imbalanced and
> the Neutral class remains difficult to predict, so accuracy alone is
> not sufficient to evaluate the model."

------------------------------------------------------------------------

# 🔮 Future Improvements

-   Use manually annotated sentiment labels
-   Improve Neutral-class representation
-   Perform systematic error analysis
-   Experiment with class weighting and resampling strategies
-   Use a scikit-learn Pipeline
-   Compare TF-IDF with Word2Vec/embeddings
-   Evaluate transformer models such as BERT
-   Build a production web interface
-   Monitor performance on new customer reviews

------------------------------------------------------------------------

## 👨‍💻 Project Objective

The main objective is not only to achieve high accuracy, but to build an
understandable end-to-end NLP classification pipeline and critically
evaluate its strengths and limitations.
