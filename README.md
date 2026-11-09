📊 Social Media Sentiment Analysis Dashboard
> A Python-based Natural Language Processing (NLP) and Machine Learning project for analysing social media posts and customer reviews and classifying them as \*\*Positive, Negative, or Neutral\*\* through an interactive \*\*Streamlit + Plotly dashboard\*\*.
Project type: Minor Project / Seminar Project  
Dataset period: March 2026  
Dataset size: 1,500 records  
Sentiment classes: Positive, Negative, Neutral
---
📌 Project Overview
The Social Media Sentiment Analysis Dashboard analyses text from multiple platforms and brands to identify the sentiment expressed in each post or review. The project demonstrates a complete NLP and machine-learning workflow, starting from data generation and text cleaning and ending with interactive visualisation and live sentiment prediction.
The system uses TF-IDF for feature extraction and compares three supervised machine-learning algorithms:
Logistic Regression
Multinomial Naive Bayes
Random Forest
A VADER-based sentiment analysis path is also included as a rule-based comparison method.
The project uses a controlled synthetic dataset so that the complete workflow can be reproduced without API keys, platform rate limits, web scraping, or private user data.
---

🎯 Objectives
The main objectives of the project are to:
Generate and analyse a structured sentiment dataset.
Clean and preprocess social-media and review text using NLP techniques.
Convert text into numerical features using TF-IDF.
Train and compare multiple machine-learning classifiers.
Perform sentiment prediction using both ML and VADER approaches.
Visualise sentiment distribution by brand and platform.
Analyse sentiment trends over time.
Provide an interactive dashboard for exploration and live prediction.
Store and display historical sentiment-analysis records.
Provide downloadable filtered analysis data for further use.
---
🧠 What Is This Project?
Sentiment Analysis is the process of using Natural Language Processing (NLP) and Machine Learning to automatically detect the emotional tone behind text — whether a customer is happy, frustrated, or neutral about a product, brand, or service.
Real-world applications:
🛒 E-Commerce (Amazon, Flipkart): Track product reviews & return complaints
🍕 Food Delivery (Zomato, Swiggy): Monitor delivery experience feedback
🎬 Streaming (Netflix): Analyze show reviews & subscription churn signals
🏦 Banks & Fintech: Detect complaints before they escalate
📱 Startups: Monitor brand reputation across social platforms
---

🚀 Live Dashboard Features
Feature	Description
📊 Overview	Sentiment distribution (pie + bar)
🏢 Brand Analysis	Per-brand sentiment scorecard
📈 Trend Analysis	Daily sentiment timeline
🤖 Live Predictor	Type any text → instant sentiment
🔍 Data Explorer	Full dataset with download option
💡 Business Insights	Auto-generated brand health report
🗃️ Analysis History | View and track all saved sentiment analysis records 

---


🌐 Platforms and Brands
Platforms
The dataset represents feedback from five platforms:
Twitter
Instagram
Facebook
Reddit
Google Reviews
Brands
The project contains ten brands:
Apple
Amazon
Flipkart
Netflix
Samsung
Myntra
Zomato
Swiggy
Uber Eats
BigBasket
---
📊 Dataset
The current project dataset contains 1,500 synthetic records with the following fields:
Field	Description
`id`	Unique record identifier
`platform`	Source platform
`brand`	Brand associated with the text
`text`	Original review/post text
`sentiment`	Positive, Negative, or Neutral label
`likes`	Number of likes/reactions
`retweets`	Number of retweets/shares
`date`	Analysis/data date
Current date range
All supplied dataset and generated prediction/preprocessing records have been updated to March 2026, covering 1 March 2026 to 31 March 2026.
The date field is used by the dashboard for trend analysis, date filtering, monthly analysis, and historical views.
> \*\*Important:\*\* The dataset is synthetic and is intended for educational demonstration. The reviews/posts should not be interpreted as actual statements made by customers or as official opinions about the listed brands.
---
🏗️ System Architecture
```text
                 ┌───────────────────────────┐
                 │   Synthetic CSV Dataset   │
                 │  1,500 records / March 26 │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │      Text Cleaning        │
                 │ URLs, mentions, emojis,   │
                 │ punctuation and numbers   │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │    NLP Preprocessing      │
                 │ Tokenization, stopwords,  │
                 │ negation preservation,    │
                 │ lemmatization              │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │       TF-IDF Features     │
                 │ Unigrams + bigrams        │
                 └─────────────┬─────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │              ML Model Training               │
        │ Logistic Regression | Naive Bayes | RF      │
        └─────────────────────┬───────────────────────┘
                              │
                              ▼
                 ┌───────────────────────────┐
                 │ Sentiment Prediction      │
                 │ Positive / Negative /     │
                 │ Neutral + VADER           │
                 └─────────────┬─────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────────┐          ┌──────────────────────┐
   │ Static Evaluation    │          │ Streamlit Dashboard  │
   │ Charts, reports,     │          │ Interactive charts,  │
   │ confusion matrices   │          │ prediction \& history │
   └──────────────────────┘          └──────────────────────┘
```
Data flow
```text
CSV File
   ↓
Pandas DataFrame
   ↓
Text Cleaning
   ↓
NLP Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Model Training / Prediction
   ↓
Sentiment Results
   ↓
Charts + Dashboard + Analysis History
```
---
🚀 Dashboard Features
Dashboard Section	Main Function
Overview	KPI cards, sentiment distribution, pie chart, bar chart and platform analysis
Brand Analysis	Compare Positive, Negative and Neutral sentiment across brands
Trend Analysis	View daily/monthly sentiment trends for March 2026
Live Predictor	Enter one or multiple texts and receive sentiment predictions
Data Explorer	Filter records by platform, brand, sentiment and date
Analysis History	Save, view and analyse previous sentiment-analysis records
Insights	Display automatic observations and brand-level sentiment insights
Analysis History
The updated dashboard includes a historical-analysis feature that stores analysed records in a local SQLite database:
```text
data/sentiment\_analysis\_history.db
```
Saved information can include:
Analysed text/keywords
Sentiment result
Confidence/score information where available
Brand and platform information
Analysis date and time
Historical sentiment category
The history section also provides Positive/Negative/Neutral comparisons, charts, monthly filtering and downloadable records.
---
🧹 Text Preprocessing
The text-processing pipeline performs the following operations:
Convert text to lowercase.
Remove URLs.
Remove user mentions.
Remove hashtags where appropriate.
Remove emojis and unwanted symbols.
Remove punctuation and numbers.
Tokenize the text.
Remove common stopwords.
Preserve important negation words such as `not`, `no`, `never`, and `nor`.
Apply lemmatization.
Reconstruct the processed text.
Preserving negation words is important because phrases such as “not good” can express a different sentiment from “good.”
---
🔢 TF-IDF Feature Extraction
The project uses scikit-learn's `TfidfVectorizer` to convert processed text into numerical features.
Main configuration:
```text
max\_features = 5000
ngram\_range = (1, 2)
min\_df = 2
sublinear\_tf = True
```
The use of unigrams and bigrams allows the model to capture individual words as well as short phrases.
---
🤖 Machine Learning Models
Three classifiers are trained and compared:
1. Logistic Regression
A strong linear classifier for sparse TF-IDF text features. The saved project model is Logistic Regression.
2. Multinomial Naive Bayes
A traditional and efficient classifier commonly used for text classification.
3. Random Forest
An ensemble learning method used to provide a different modelling approach for comparison.
The project uses an 80:20 stratified train/test split.
---
🧠 VADER Sentiment Analysis
VADER is included as a rule-based sentiment-analysis method. It produces a compound sentiment score and maps the result to Positive, Negative, or Neutral categories.
The VADER path provides a useful comparison with the supervised machine-learning models because it does not require model training.
---
📈 Evaluation Results
The supplied trained artifacts report the following test-set results:
Model	Test Accuracy	Macro Precision	Macro Recall	Macro F1
Logistic Regression	100%	1.00	1.00	1.00
Multinomial Naive Bayes	100%	1.00	1.00	1.00
Random Forest	100%	1.00	1.00	1.00
The stored best model is Logistic Regression. Because all three models obtained the same stored test accuracy, this selection is effectively a tie under the current implementation.
> \*\*Evaluation caution:\*\* The dataset is synthetic and contains repeated/near-duplicate text templates. Therefore, the reported 100% test accuracy should \*\*not\*\* be interpreted as real-world model accuracy. A stronger future evaluation should use independently collected and labelled data or split unique templates before testing.
The stored VADER comparison reports approximately 83.13% agreement with the supplied sentiment labels.
---
📊 Generated Visualisations
The project generates the following visual outputs:
Sentiment distribution pie/bar chart
Sentiment by brand
Sentiment by platform
Sentiment trend over time
VADER compound-score distribution
Model accuracy comparison
Logistic Regression confusion matrix
Naive Bayes confusion matrix
Random Forest confusion matrix
The main trend visualisation and date-based project data are configured for March 2026.
---
📁 Project Structure
```text
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/
│   ├── generate\_dataset.py
│   ├── social\_media\_data.csv
│   └── sentiment\_analysis\_history.db      # Created by dashboard when history is used
│
├── src/
│   ├── \_\_init\_\_.py
│   ├── clean\_text.py
│   ├── preprocess.py
│   ├── feature\_extraction.py
│   ├── train\_model.py
│   ├── predict.py
│   └── evaluate.py
│
├── app/
│   └── dashboard.py
│
├── models/
│   ├── best\_model.pkl
│   ├── tfidf\_vectorizer.pkl
│   └── label\_encoder.pkl
│
├── outputs/
│   ├── preprocessed\_data.csv
│   ├── predictions\_output.csv
│   ├── classification\_reports.txt
│   └── insights\_report.txt
│
├── images/
│   ├── sentiment\_distribution.png
│   ├── sentiment\_by\_brand.png
│   ├── sentiment\_by\_platform.png
│   ├── sentiment\_trend\_over\_time.png
│   ├── vader\_compound\_distribution.png
│   ├── model\_accuracy\_comparison.png
│   └── \*\_confusion\_matrix.png
│
├── notebooks/
│   └── sentiment\_analysis\_eda.ipynb
│
├── docs/
│   └── architecture.md
│
├── main.py
├── requirements.txt
└── README.md
```
---
⚙️ Installation
Requirements
Python 3.9 or higher
pip
Windows, macOS or Linux
1. Create/activate a virtual environment
Windows:
```powershell

cd .. - change one step back folder
cd C:\\Projects\SentimentDashboard

Create Environment:

python -m venv venv

.\\venv\\Scripts\\activate

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\\venv\\Scripts\\activate


python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('vader_lexicon')"

For edit code:
notepad .\app\dashboard.py

for executing dashboard 
streamlit run app/dashboard.py


```
macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
---
▶️ Running the Project
Recommended: launch the existing dashboard
If the supplied dataset, models and outputs are already present, you do not need to retrain the models just to view the dashboard.
```bash
streamlit run app/dashboard.py
```
Then open the local Streamlit address shown in the terminal, normally:
```text
http://localhost:8501
```
Full pipeline
To regenerate the complete project pipeline:
```bash
python main.py --all
```
Individual commands
```bash
python main.py --generate
python main.py --preprocess
python main.py --train
python main.py --evaluate
python main.py --predict
```
> \*\*Note:\*\* Running `--generate` or `--all` regenerates project data and may change the supplied March 2026 dataset/output files. For the submitted version, use the supplied files unless regeneration is required.

Option A – Full Pipeline (Recommended for first run)
```bash
python main.py --all
```
This runs: generate → preprocess → train → evaluate → predict
Option B – Step by step
```bash
# 1. Generate synthetic dataset
python main.py --generate

# 2. Preprocess text
python main.py --preprocess

# 3. Train ML models
python main.py --train

# 4. Generate charts
python main.py --evaluate

# 5. Demo predictions
python main.py --predict

# 6. Launch dashboard
python main.py --dashboard
```
Launch Dashboard directly
```bash
streamlit run app/dashboard.py
```
Then open → http://localhost:8501

streamlit run app/dashboard.py

notepad .\app\dashboard.py

---
🖥️ Using the Dashboard
Start the application with:
```bash
   streamlit run app/dashboard.py
   ```
Open the Streamlit URL in a browser.
Use the sidebar to filter by:
Platform
Brand
Sentiment
Date range
Explore the dashboard tabs.
Use Live Predictor to analyse new text.
Use Analysis History to review saved analyses.
Use the available download controls to export filtered data where provided.
---
💾 Analysis History Database
The dashboard uses SQLite for local historical records. The database file is created automatically when the history functionality is initialised.
```text
data/sentiment\_analysis\_history.db
```
The existing March 2026 dataset can be used to initialise the historical records. New records submitted through the live predictor are stored as new analysis events with their analysis timestamp.
---
🛠️ Technology Stack
Category	Technology
Programming Language	Python
Data Processing	Pandas, NumPy
NLP	NLTK
Feature Extraction	TF-IDF / scikit-learn
Machine Learning	Logistic Regression, Naive Bayes, Random Forest
Rule-Based Sentiment	VADER
Static Visualisation	Matplotlib, Seaborn
Interactive Visualisation	Plotly
Dashboard	Streamlit
Database	SQLite
Model Storage	Pickle / Joblib
---
🎓 Learning Outcomes
This project demonstrates:
End-to-end NLP pipeline development
Text cleaning and preprocessing
Negation-aware preprocessing
TF-IDF feature engineering
Supervised sentiment classification
Comparison of multiple ML algorithms
Rule-based VADER sentiment analysis
Model evaluation and confusion matrices
Interactive data visualisation
Streamlit dashboard development
Historical analysis storage using SQLite
Data filtering and CSV export
Reproducible project organisation
---
🔮 Future Scope
Possible future improvements include:
Replace synthetic data with ethically collected real-world data.
Add multilingual and code-mixed sentiment analysis.
Implement aspect-based sentiment analysis.
Compare TF-IDF models with BERT/transformer-based models.
Add LLM-based sentiment classification.
Add model drift and performance monitoring.
Add explainability techniques such as feature importance or SHAP.
Add secure user authentication and deployment.
Connect the dashboard to a production database and continuous data-ingestion pipeline.
---
⚠️ Project Limitations
This project is an educational prototype. The dataset is synthetic, and the brands/platforms are used as categories for demonstrating sentiment-analysis techniques. The project does not claim to collect or represent the actual opinions of customers on these services.
The high stored test accuracy is also affected by repeated synthetic text templates and should not be treated as evidence of production-level performance.
---
👤 Author
Purushottam Sah 
Minor Project / Seminar Project  
Social Media Sentiment Analysis Dashboard  
2026
Replace `\[Your Name]` with the student name before final submission if required.
---
📄 License
This project is intended for educational and academic use.
---
⭐ Project Summary
The Social Media Sentiment Analysis Dashboard combines NLP, TF-IDF feature extraction, machine learning, VADER sentiment analysis, interactive Plotly visualisations, Streamlit, and SQLite-based historical analysis into a single academic project. The current submitted dataset and date-based outputs are configured for March 2026.