# 📊 Social Media Sentiment Analysis Dashboard

### 🧠 Real-Time Social Intelligence — Powered by NLP & Machine Learning

An interactive dashboard developed using Python, Natural Language Processing (NLP), and Machine Learning to analyse social media sentiment, understand audience opinions, compare brand perceptions, and discover meaningful insights through interactive data visualisations.

**✦ Designed & Developed by Purushottam Sah**

---

## 🚀 Project Overview

The **Social Media Sentiment Analysis Dashboard** is a data-driven application designed to analyse social media posts and identify public sentiment as Positive, Negative, or Neutral.

The project combines Natural Language Processing, Machine Learning, and interactive visualisation techniques to transform textual social media data into meaningful information. It provides an intuitive dashboard where users can explore sentiment distributions, compare brands, examine trends, and predict the sentiment of custom text.

The application aims to demonstrate how NLP and machine learning can support social media monitoring, brand perception analysis, and data-driven decision-making.

## ✨ Key Features

### 📊 1. Overview Dashboard
- View the total number of social media posts.
- Explore positive, negative, and neutral sentiment distributions.
- Visualise sentiment patterns through interactive charts.
- Get an overview of the analysed dataset.

### 🏢 2. Brand Analysis
- Analyse sentiment associated with different brands.
- Compare brand-related audience opinions.
- Explore positive, negative, and neutral discussions.
- Understand brand perception through visual analytics.

### 📈 3. Trend Analysis
- Explore sentiment patterns over time.
- Analyse changes in social media discussions.
- Visualise sentiment trends and platform activity.

### 🤖 4. Live Sentiment Predictor
- Enter custom text for sentiment analysis.
- Predict sentiment using the available ML model.
- Compare predictions with VADER (Enhanced).
- Understand the sentiment classification of individual posts.

### 🔍 5. Data Explorer
- Explore individual social media records.
- Filter posts by platform, brand, sentiment, and date.
- Examine the underlying dataset.

### 💡 6. Business Insights
- Explore important sentiment patterns.
- Identify potential brand perception concerns.
- Understand audience reactions.
- Support informed business decision-making.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| NLTK | Natural Language Processing |
| VADER | Rule-based sentiment analysis |
| Scikit-learn | Machine Learning and classification |
| TextBlob | Text processing |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Plotly | Interactive charts |
| WordCloud | Text visualisation |

---

## 🧠 Sentiment Classification

The application analyses social media text and categorises it into three sentiment classes:

- 🟢 **Positive:** Expresses favourable opinions, satisfaction, or positive experiences.
- 🔴 **Negative:** Expresses criticism, dissatisfaction, or unfavourable opinions.
- ⚪ **Neutral:** Contains objective or emotionally neutral information.

The dashboard provides two prediction approaches:

1. **Machine Learning Model:** Uses the trained classification model to predict sentiment.
2. **VADER (Enhanced):** Uses a lexicon and rule-based approach to analyse sentiment polarity.

---

## 📂 Project Structure

```text
Social-Media-Sentiment-Analysis-Dashboard/
│
├── app/
│   └── dashboard.py
│
├── data/
│   └── Dataset files
│
├── models/
│   └── Trained models
│
├── src/
│   └── Source code
│
├── notebooks/
│   └── Jupyter notebooks
│
├── outputs/
│   └── Generated outputs
│
├── docs/
│   └── Project documentation
│
├── images/
│   └── Dashboard screenshots
│
├── main.py
├── requirements.txt
└── README.md
```

*Note: The structure is illustrative. Update folder and file names to match the actual repository.*

---

## ⚙️ Installation and Setup

Follow these steps to run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Directory

```bash
cd Social-Media-Sentiment-Analysis-Dashboard
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 5: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app/dashboard.py
```

### Step 7: Open the Dashboard

Once the application starts, open the following address in your browser:

```text
http://localhost:8501
```

---

## 📸 Dashboard Preview

Add screenshots of your actual dashboard to the `images` folder.

For example:

### Dashboard Overview

![Dashboard Overview](images/dashboard-overview.png)

### Brand Analysis

![Brand Analysis](images/brand-analysis.png)

### Sentiment Prediction

![Live Sentiment Predictor](images/live-predictor.png)

*Replace these image paths with your actual screenshot filenames.*

---

## 🎯 Project Objectives

- Develop an interactive social media sentiment analysis dashboard.
- Apply NLP techniques to process and analyse textual data.
- Implement machine learning for sentiment classification.
- Compare sentiment across different brands and platforms.
- Visualise social media sentiment patterns and trends.
- Generate meaningful insights from analysed social media data.

---

## 🔮 Future Enhancements

- Integration with live social media APIs.
- Real-time social media data collection.
- Support for multilingual sentiment analysis.
- Advanced deep learning-based sentiment classification.
- Improved model evaluation and prediction accuracy.
- Advanced brand comparison and sentiment reporting.
- Automated sentiment monitoring and notifications.

---

## ⚠️ Dataset Disclaimer

This project uses a sample dataset for demonstration and analytical purposes. If the dataset contains synthetic or generated records, the results should not be interpreted as actual public opinion or real-world social media trends.

Machine learning performance should be evaluated using appropriate training and testing procedures to avoid data leakage and misleading accuracy results.

---

## 📜 License

This project is developed for educational, learning, and demonstration purposes. An appropriate open-source license can be added to define permissions for reuse, modification, and distribution.

---

## 👨‍💻 Author

<div align="center">

# **Purushottam Sah**

### ✦ Designed & Developed by Purushottam Sah

🧠 Python | NLP | Machine Learning | Data Analytics

**Social Media Sentiment Analysis Dashboard**

*Turning Social Conversations into Meaningful Insights. 🚀*

</div>

---

⭐ If you find this project interesting, consider giving the repository a star!
