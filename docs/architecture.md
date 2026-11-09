# 🏗️ Project Architecture – Social Media Sentiment Analysis Dashboard

## Overview

This document describes the system architecture, data flow, module responsibilities, and design decisions of the Social Media Sentiment Analysis Dashboard. The updated system integrates machine-learning-based sentiment prediction, interactive visualizations, live text analysis, and a persistent Analysis History module backed by SQLite.

---

## System Architecture (Text Block Diagram)

```text
╔══════════════════════════════════════════════════════════════════╗
║           SOCIAL MEDIA SENTIMENT ANALYSIS DASHBOARD             ║
╚══════════════════════════════════════════════════════════════════╝

INPUT LAYER
┌────────────────────────────────────────────────────────────────┐
│  Synthetic Dataset (CSV)                                        │
│                                                                  │
│  Twitter | Instagram | Facebook | Reddit | Google Reviews       │
│                                                                  │
│  Brands: Apple, Amazon, Flipkart, Netflix, Samsung, Myntra,     │
│          Zomato, Swiggy, Uber Eats, BigBasket                   │
│                                                                  │
│  Columns: id, platform, brand, text, sentiment, likes,          │
│           retweets, date                                         │
│  Dataset: 1,500 records | March 1–31, 2026                      │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
DATA LOADING LAYER
┌────────────────────────────────────────────────────────────────┐
│  Pandas DataFrame                                                │
│  • Load and inspect CSV records                                  │
│  • Validate columns and data types                               │
│  • Prepare text and metadata for analysis                        │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
CLEANING LAYER (src/clean_text.py)
┌────────────────────────────────────────────────────────────────┐
│  Example Input:                                                  │
│  "Zomato delivery was super fast! 🎉 @zomato #Zomato"           │
│                                                                  │
│  → Convert text to lowercase                                     │
│  → Remove URLs and @mentions                                     │
│  → Remove hashtags and emojis                                    │
│  → Remove punctuation and numbers                                │
│  → Remove unnecessary whitespace                                 │
│                                                                  │
│  Example Output:                                                 │
│  "zomato delivery was super fast"                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
NLP PREPROCESSING LAYER (src/preprocess.py)
┌────────────────────────────────────────────────────────────────┐
│  • Tokenization                                                  │
│  • Stopword removal                                              │
│  • Preserve negation words: not, no, never, nor                 │
│  • Lemmatization                                                 │
│  • Rejoin tokens into processed text                             │
│                                                                  │
│  Output: processed text ready for feature extraction             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
FEATURE EXTRACTION LAYER (src/feature_extraction.py)
┌────────────────────────────────────────────────────────────────┐
│  Algorithm: TF-IDF Vectorization                                 │
│                                                                  │
│  Configuration:                                                  │
│  • max_features = 5000                                           │
│  • ngram_range = (1, 2)                                          │
│  • min_df = 2                                                    │
│  • sublinear_tf = True                                           │
│                                                                  │
│  Actual learned vocabulary: 791 features                         │
│  Output: Numerical feature matrix                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
MODEL TRAINING LAYER (src/train_model.py)
┌────────────────────────────────────────────────────────────────┐
│  Stratified Train/Test Split: 80% / 20%                         │
│                                                                  │
│  ┌──────────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │ Logistic Regression  │ │ Naive Bayes     │ │ Random Forest│ │
│  │ max_iter = 1000      │ │ alpha = 0.5     │ │ 100 trees    │ │
│  └──────────────────────┘ └─────────────────┘ └──────────────┘ │
│                                                                  │
│  • Evaluate models using classification metrics                 │
│  • Compare model performance                                     │
│  • Select Logistic Regression as the preferred model             │
│  • Save/load trained model artifacts as configured              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
PREDICTION LAYER (src/predict.py)
┌────────────────────────────────────────────────────────────────┐
│  Machine Learning Pipeline:                                     │
│  Input text → Cleaning → NLP → TF-IDF → Model prediction        │
│                                                                  │
│  VADER Sentiment Analysis:                                       │
│  Input text → Compound score → Sentiment classification         │
│                                                                  │
│  Output: Positive / Negative / Neutral                           │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
VISUALIZATION AND EVALUATION LAYER (src/evaluate.py)
┌────────────────────────────────────────────────────────────────┐
│  • Sentiment distribution                                       │
│  • Sentiment by brand                                           │
│  • Sentiment by platform                                        │
│  • Sentiment trend over time                                    │
│  • VADER compound score distribution                            │
│  • Model accuracy comparison                                    │
│  • Confusion matrices                                           │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
DASHBOARD LAYER (app/dashboard.py)
┌────────────────────────────────────────────────────────────────┐
│  Interactive Streamlit Dashboard                                │
│                                                                  │
│  1. Overview        – Summary metrics and sentiment charts      │
│  2. Brand Analysis  – Brand-level sentiment comparison          │
│  3. Trend Analysis  – Sentiment patterns over time               │
│  4. Live Predictor  – Analyse newly entered text                │
│  5. Data Explorer   – Explore records and download CSV data     │
│  6. Insights        – Automatically generated business insights  │
│  7. Analysis History– Review saved sentiment analyses           │
│                                                                  │
│  Sidebar Filters: Platform | Brand | Sentiment | Date Range     │
└────────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
LIVE ANALYSIS MODULE                ANALYSIS HISTORY MODULE
┌────────────────────────┐         ┌────────────────────────────┐
│ User enters new text   │         │ SQLite Database             │
│          │             │         │ data/sentiment_analysis_    │
│          ▼             │         │ history.db                  │
│ Predict sentiment      │         │             ▲               │
│          │             │         │             │               │
│          ▼             │         │ Save and retrieve records   │
│ Save analysis record ──┼────────►│             │               │
└────────────────────────┘         │             ▼               │
                                   │ Historical Analysis Table   │
                                   │ • Input text and sentiment  │
                                   │ • Analysis date and details │
                                   │ • Positive/Negative/Neutral │
                                   │   comparisons               │
                                   │ • Pie and bar charts        │
                                   │ • Date/month filtering      │
                                   │ • Historical trend analysis │
                                   │ • CSV export                │
                                   └────────────────────────────┘
```

---

## Data Flow Summary

### 1. Dataset Processing and Model Training

```text
CSV Dataset
    └──▶ Pandas DataFrame
          └──▶ Text Cleaning
                └──▶ NLP Preprocessing
                      └──▶ TF-IDF Vectorization
                            └──▶ Train/Test Split
                                  └──▶ Model Training
                                        └──▶ Model Evaluation
                                              └──▶ Sentiment Prediction
                                                    └──▶ Dashboard Charts
```

### 2. Live Prediction and Analysis History

```text
User Enters Text
    └──▶ Text Cleaning
          └──▶ NLP Preprocessing
                └──▶ TF-IDF Transformation
                      └──▶ Sentiment Prediction
                            └──▶ Display Prediction
                                  └──▶ Save Analysis to SQLite
                                        └──▶ Analysis History
                                              ├──▶ Historical Records
                                              ├──▶ Sentiment Comparisons
                                              ├──▶ Pie Chart
                                              ├──▶ Bar Chart
                                              ├──▶ Date/Month Filters
                                              └──▶ CSV Download
```

---

## Analysis History – New Feature

The Analysis History module extends the dashboard beyond static dataset analysis by maintaining a persistent record of sentiment analyses.

### Main Functions

- **Persistent storage:** Stores analysis records in the SQLite database located at `data/sentiment_analysis_history.db`.
- **Historical records:** Displays previously saved analyses in a table for review.
- **Sentiment comparison:** Compares Positive, Negative, and Neutral analysis results.
- **Interactive charts:** Presents historical sentiment distributions through pie and bar charts.
- **Date and month filtering:** Allows users to examine records for selected time periods.
- **Historical trends:** Helps users explore changes in recorded sentiment over time.
- **CSV export:** Allows users to download historical analysis records for further analysis.
- **Live prediction integration:** Connects newly analysed text with the history module so the result can be saved and reviewed later.

The dashboard initializes its history database from the current dataset when required. Consequently, the database file does not need to be distributed in the final project ZIP if the initialization process is included in the application.

---

## Key Design Decisions

### Why TF-IDF instead of Word2Vec or BERT?

- TF-IDF converts text into numerical features suitable for traditional machine-learning algorithms.
- It is relatively lightweight and can run on a standard computer without a GPU.
- Unigrams and bigrams help represent individual terms and short phrases.
- It provides a practical feature-extraction approach for an academic sentiment-analysis project.

### Why use three machine-learning models?

- **Logistic Regression:** Provides a classification baseline and is selected as the preferred model in this project.
- **Naive Bayes:** Offers a lightweight probabilistic approach commonly used for text classification.
- **Random Forest:** Provides an ensemble-based comparison against the other models.

Comparing these algorithms helps demonstrate how different classification approaches perform on the same dataset.

### Why preserve negation words?

Words such as `not`, `no`, `never`, and `nor` can change the meaning of a sentence. For example, "not good" expresses a different opinion from "good". Retaining negation words during stopword removal helps preserve important sentiment information.

### Why use synthetic data?

- It avoids dependence on live social media APIs during demonstrations.
- It supports repeatable testing with a controlled dataset.
- It provides a consistent structure for comparing brands, platforms, and sentiment categories.
- It allows the project to demonstrate the complete data-processing and visualization workflow.

**Evaluation note:** The stored evaluation reports show 100% test accuracy for the three models on the current controlled dataset. Because the dataset contains repeated or similar text templates, these results should not be interpreted as evidence of 100% accuracy on real-world social media content.

### Why use SQLite for Analysis History?

- SQLite provides persistent local storage without requiring a separate database server.
- It allows analysis records to remain available after the dashboard is restarted.
- It supports retrieval and display of historical analyses.
- It is suitable for a lightweight, single-user academic prototype.

### Why use Streamlit?

- Streamlit makes it possible to build an interactive dashboard using Python.
- It supports data tables, charts, filters, and user input.
- It integrates conveniently with Pandas and machine-learning workflows.
- It enables users to explore sentiment results without building a separate frontend application.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Text Processing | NLTK |
| Feature Extraction | TF-IDF, scikit-learn |
| Machine Learning | Logistic Regression, Naive Bayes, Random Forest |
| Lexicon-Based Analysis | VADER |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn, Plotly (where used by the relevant module) |
| History Database | SQLite |
| Dataset Format | CSV |

---

## Important Implementation Notes

- The supplied dataset contains 1,500 records covering March 1–31, 2026.
- The historical dataset and newly submitted live analyses can have different dates because live analyses use the current system date and time.
- The dashboard uses a local SQLite database for Analysis History; it does not require a remote database server.
- The project uses a synthetic dataset rather than a live feed from social media platforms.
- Actual prediction quality on unseen real-world text requires independent evaluation using representative, human-labelled data.