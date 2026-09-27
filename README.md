# Spam Email Classifier using Machine Learning and NLP

A complete, production-grade Natural Language Processing (NLP) and Machine Learning web application designed for Artificial Intelligence & Data Science undergraduate academic capstone projects and placement portfolios.

---

## 1. Project Overview

Spam email classification is a fundamental binary text classification problem in Artificial Intelligence. Unsolicited, fraudulent, or malicious emails (Spam) pose critical security and financial threats to users and organizations, whereas legitimate emails (Ham) must be preserved without false filtering.

This application uses **Natural Language Processing (NLP)** and **Supervised Machine Learning**:
1. **Linguistic Preprocessing**: Normalizes text by mapping URLs, emails, phone numbers, and currency values into semantic tokens while handling case-folding and punctuation.
2. **TF-IDF Vectorization**: Calculates Term Frequency - Inverse Document Frequency with sublinear term-frequency scaling and n-gram ranges `(1, 2)` to capture phrases like *"wire transfer"* or *"free prize"*.
3. **Machine Learning Classification**: Compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM, automatically selecting the optimal model based on held-out F1-score.
4. **Explainability & Analytics**: Calculates structural message statistics (characters, words, URLs, numbers, special characters, uppercase chars, exclamation marks) and highlights influential TF-IDF feature indicators.
5. **Full-Stack Interface**: Features demo authentication, an interactive dashboard with Chart.js analytics, real-time message analysis, and SQLite-backed prediction history.

---

## 2. Key Features

- **Real Machine Learning Pipeline**: Scikit-learn TF-IDF vectorizer + trained classifiers (MultinomialNB, Logistic Regression, Linear SVM).
- **No Rule-Based Shortcuts**: All predictions derive strictly from vectorized TF-IDF statistical dot products and class conditional probabilities.
- **Model Explainability**: Highlights influential tokens that contributed to the classification outcome and provides contextual linguistic explanations.
- **Message Structural Statistics**: Computes 7 granular metrics: character count, word count, URLs, numbers, special characters, uppercase characters, and exclamation marks.
- **Academic Benchmark Comparison**: Evaluates and displays Accuracy, Precision, Recall, F1-Score, and a dynamic Confusion Matrix.
- **Interactive Analytics Dashboard**: Visualizes class distribution and prediction metrics with Chart.js.
- **Historical Audit Trail**: SQLite database logs previews, confidence values, lengths, and timestamps with search and filter capabilities (All, Spam, Ham).
- **REST API**: JSON endpoints (`/predict`, `/health`, `/api/model-info`) for headless integration.
- **Clean Modern UI**: Responsive, professional layout built with Bootstrap 5 and custom CSS.

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js, Bootstrap Icons |
| **Backend** | Python 3.10+, Flask, Flask-CORS, Gunicorn |
| **Machine Learning** | Scikit-learn, Pandas, NumPy, Joblib |
| **NLP** | TF-IDF Vectorization, N-grams `(1,2)`, Sublinear TF, Tokenization, Stopwords |
| **Database** | SQLite3 (`spam_predictions.db`) |
| **Environment** | VS Code, Windows 11 / Linux / macOS |

---

## 4. Machine Learning Workflow

```
[Raw Email Message]
         │
         ▼
[Text Preprocessing]
  ├── Lowercase conversion
  ├── URL normalization  -> url_token
  ├── Email normalization -> email_token
  ├── Currency symbols    -> currency_token
  ├── Number/Phone        -> number_token / phone_token
  └── Extraneous whitespace & character stripping
         │
         ▼
[TF-IDF Feature Extraction]
  ├── Sublinear term-frequency scaling: 1 + log(tf)
  ├── Inverse document frequency: log((1 + N)/(1 + df)) + 1
  ├── N-gram range: (1, 2) unigrams and bigrams
  └── English stop-words filtering
         │
         ▼
[Trained Machine Learning Model]
  ├── Multinomial Naive Bayes (log P(w|c) + log P(c))
  ├── Logistic Regression (sigmoid activation: 1 / (1 + e^-z))
  └── Linear SVM (Calibrated probabilities)
         │
         ▼
[Inference & Explainability Engine]
  ├── Class Prediction: SPAM or NOT SPAM (HAM)
  ├── Calibrated Confidence Percentage
  ├── Influential TF-IDF Term Indicators
  └── 7 Message Structural Statistics
         │
         ▼
[SQLite Audit Logging & Response]
```

---

## 5. Dataset Description

The bundled dataset is located at `data/spam_email_dataset.csv`.
- **Nature**: Realistic synthetic demonstration dataset crafted to mirror real-world spam distributions (phishing, lottery scams, fake banking alerts, crypto fraud, miracle cures, and legitimate work, academic, and personal correspondence).
- **Labeling Notice**: Clearly labeled as a structured synthetic benchmark dataset for academic training and local testing.
- **Extensibility**: Users can substitute this CSV with larger public benchmarks such as the *SMS Spam Collection* (5,572 rows) or the *Enron Spam Dataset* simply by matching the `label` and `text` column headers.

---

## 6. Installation & Setup (VS Code on Windows 11)

Follow these exact commands in your **VS Code Terminal** (PowerShell or Command Prompt):

### Step 1: Open Project in VS Code
Open VS Code and navigate to the project directory:
```powershell
cd spam_email_classifier
```

### Step 2: Create a Python Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
On Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and rerun the activate command).*

Or on Windows Command Prompt (`cmd`):
```cmd
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 7. Model Training

Run the training pipeline script:
```powershell
python train_model.py
```

### What `train_model.py` does:
1. Loads and deduplicates `data/spam_email_dataset.csv`.
2. Inspects missing values and verifies class distribution.
3. Preprocesses text and splits data using **Stratified 75/25 Train-Test Split**.
4. Fits the `TfidfVectorizer` and builds the feature matrix.
5. Trains and cross-evaluates:
   - **Multinomial Naive Bayes**
   - **Logistic Regression (Class-weighted)**
   - **Linear SVM (Calibrated)**
6. Compares Accuracy, Precision, Recall, and F1-Scores.
7. Displays the classification report and confusion matrix in the console.
8. Automatically serializes the best performing model to:
   - `models/spam_classifier.pkl`
   - `models/tfidf_vectorizer.pkl`
   - `models/model_metadata.json`

---

## 8. Running the Application

Start the Flask web server:
```powershell
python app.py
```

The terminal will confirm:
```
* Running on http://127.0.0.1:5000
* Debug mode: on
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 9. Demo Login Credentials

For demonstration and academic presentation purposes, use the demo credentials:
- **Username**: `admin`
- **Password**: `admin123`

*Note: Session-based cookie authentication is used to protect `/dashboard`, `/classifier`, and `/history`.*

---

## 10. API Documentation

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "Spam Email Classifier Backend",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "timestamp": "2026-09-27T08:30:00"
}
```

### 2. Predict Message
```http
POST /predict
Content-Type: application/json
```
**Request Body:**
```json
{
  "text": "Congratulations! You have won a $1,000 Walmart Gift Card. Click http://bit.ly/claim-prize now before it expires!"
}
```

**Response:**
```json
{
  "id": 1,
  "prediction": "SPAM",
  "confidence": 98.4,
  "message_length": 115,
  "word_count": 18,
  "statistics": {
    "character_count": 115,
    "word_count": 18,
    "urls_count": 1,
    "numbers_count": 2,
    "special_chars_count": 7,
    "uppercase_count": 7,
    "exclamation_count": 2
  },
  "explanation": "This message displays semantic patterns frequently associated with unsolicited or fraudulent communications...",
  "key_indicators": ["prize", "url_token", "currency_token", "won", "card"],
  "timestamp": "2026-09-27 08:30:15"
}
```

---

## 11. Troubleshooting Common Errors

### 1. `ModuleNotFoundError: No module named 'flask'` (or `sklearn`)
- **Cause**: Virtual environment is either not activated or packages were installed in global Python.
- **Fix**: Verify your prompt starts with `(venv)`. Run:
  ```powershell
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```

### 2. `FileNotFoundError: models/spam_classifier.pkl not found`
- **Cause**: The application was started before training the ML pipeline.
- **Fix**: Run `python train_model.py` to create the `models/` directory and serialized `.pkl` files.

### 3. `Address already in use` or Port 5000 Conflict
- **Cause**: Another service or prior Flask process is using port 5000.
- **Fix**: Run on a different port:
  ```powershell
  python -c "import os; from app import app; app.run(port=5001)"
  ```
  or kill the existing process on Windows:
  ```powershell
  netstat -ano | findstr :5000
  taskkill /PID <PID_NUMBER> /F
  ```

### 4. `sqlite3.OperationalError: database is locked`
- **Cause**: Multiple write transactions or file permission locking in the directory.
- **Fix**: Close any external SQLite browsing tools holding an active write lock on `spam_predictions.db`.

### 5. `Script execution is disabled on this system` (PowerShell Error)
- **Fix**: Run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  then activate the virtual environment.

---

## 12. Future Enhancements

- **Transformer / BERT Models**: Fine-tuning pre-trained transformer embeddings (e.g., DistilBERT or RoBERTa) for deep semantic comprehension.
- **MIME & Attachment Inspection**: Scanning attachment file signatures and headers for polymorphic malware payload detection.
- **Domain Reputation & DKIM/SPF Lookup**: Real-time integration with DNSBL and SPF/DMARC email authentication records.
- **Multilingual Spam Support**: Cross-lingual embeddings (mBERT/XLM-R) for multi-language spam detection.
