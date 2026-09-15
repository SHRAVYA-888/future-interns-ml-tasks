# Machine Learning Internship Tasks

This repository contains machine learning models and NLP pipelines developed for internship assignments. It covers time-series forecasting, automated text classification, and candidate screening using TF-IDF and cosine similarity.

---

## 📌 Project Overview

### 1. Task 1: Sales Forecasting
- **Goal:** Predict future sales trends based on historical data.
- **Model:** `RandomForestRegressor`
- **Key Concepts:** Feature engineering (date/time extraction), train-test split, evaluation metrics (MAE, RMSE), and data visualization with `matplotlib`.

### 2. Task 2: Support Ticket Classification
- **Goal:** Categorize customer support tickets and assign priority levels automatically.
- **Model:** Logistic Regression with TF-IDF Vectorization
- **Key Concepts:** Text preprocessing, stop-word removal, TF-IDF feature extraction, and multi-class classification.

### 3. Task 3: Resume Screener & Skill Gap Analyzer
- **Goal:** Rank job candidates against a job description and highlight missing skills.
- **Technique:** TF-IDF & Cosine Similarity
- **Key Concepts:** Text similarity scoring, skill extraction, keyword matching, and automated candidate ranking.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.x
- **Libraries:**
  - `pandas` & `numpy` (Data manipulation)
  - `scikit-learn` (Machine learning algorithms & TF-IDF)
  - `matplotlib` (Data visualization)
  - `nltk` / `spacy` (Text processing)

---

## 🚀 How to Run the Scripts

Clone the repository and run each script directly from your terminal:

```bash
# Task 1: Sales Forecasting
python task1_forecasting.py

# Task 2: Support Ticket Classifier
python task2_ticket_classifier.py

# Task 3: Resume Screener
python task3_resume_screener.py
