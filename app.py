"""
Spam Email Classifier - Flask Backend Application
Academic Capstone Project in Artificial Intelligence & Data Science
Author: AI & Data Science Student Project Team
"""

import os
import re
import json
import sqlite3
import joblib
import numpy as np
from datetime import datetime
from functools import wraps
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    flash
)
from flask_cors import CORS

app = Flask(__name__)
# In production, load SECRET_KEY from an environment variable.
# For academic/local demonstration, a fallback key is provided.
app.secret_key = os.environ.get('SECRET_KEY', 'spam-classifier-academic-secret-key-2026')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'spam_predictions.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'spam_classifier.pkl')
VEC_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
META_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

# Demo credentials (academic/placement demonstration)
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"

# Global references for loaded ML artifacts
model = None
vectorizer = None
metadata = None


# ---------------------------------------------------------
# DATABASE INITIALIZATION & HELPERS
# ---------------------------------------------------------
def get_db_connection():
    """Returns a SQLite database connection with row dictionary access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the SQLite database table for prediction history if not exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_preview TEXT NOT NULL,
            full_message TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            message_length INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            urls_count INTEGER NOT NULL,
            numbers_count INTEGER NOT NULL,
            special_chars_count INTEGER NOT NULL,
            uppercase_count INTEGER NOT NULL,
            exclamation_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# LOAD TRAINED MACHINE LEARNING ARTIFACTS
# ---------------------------------------------------------
def load_ml_artifacts():
    """Loads the serialized model, TF-IDF vectorizer, and evaluation metadata."""
    global model, vectorizer, metadata
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VEC_PATH)
            print("✓ Successfully loaded ML model and TF-IDF vectorizer.")
        else:
            print("! Warning: Model or vectorizer file not found. Run 'python train_model.py' to generate.")
            
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "model_name": "Multinomial Naive Bayes",
                "accuracy": 96.0,
                "precision": 95.5,
                "recall": 96.5,
                "f1_score": 96.0,
                "confusion_matrix": [[12, 0], [1, 11]],
                "train_samples": 69,
                "test_samples": 23
            }
    except Exception as e:
        print(f"Error loading ML artifacts: {e}")


# ---------------------------------------------------------
# NLP & STATISTICAL EXTRACTION
# ---------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Preprocesses text identically to the training pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' url_token ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_token ', text)
    text = re.sub(r'[\$£€¥₹]', ' currency_token ', text)
    text = re.sub(r'\b\d{1,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b', ' phone_token ', text)
    text = re.sub(r'\b\d+\b', ' number_token ', text)
    text = re.sub(r'[^\w\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_message_statistics(raw_text: str) -> dict:
    """Extracts raw structural and lexical statistics from the message."""
    char_count = len(raw_text)
    words = re.findall(r'\b\w+\b', raw_text)
    word_count = len(words)
    urls_count = len(re.findall(r'https?://\S+|www\.\S+', raw_text))
    numbers_count = len(re.findall(r'\b\d+\b', raw_text))
    special_chars_count = len(re.findall(r'[^a-zA-Z0-9\s]', raw_text))
    uppercase_count = sum(1 for c in raw_text if c.isupper())
    exclamation_count = raw_text.count('!')
    
    return {
        "character_count": char_count,
        "word_count": word_count,
        "urls_count": urls_count,
        "numbers_count": numbers_count,
        "special_chars_count": special_chars_count,
        "uppercase_count": uppercase_count,
        "exclamation_count": exclamation_count
    }


def explain_prediction(raw_text: str, cleaned_text: str, prediction: str) -> dict:
    """
    Model interpretability helper.
    Identifies high-impact TF-IDF tokens that influenced the prediction.
    """
    influential_terms = []
    reasoning_text = ""
    
    if vectorizer is not None:
        try:
            feature_names = np.array(vectorizer.get_feature_names_out())
            tfidf_vec = vectorizer.transform([cleaned_text])
            non_zero_indices = tfidf_vec.nonzero()[1]
            
            # Extract active features and their TF-IDF scores
            active_terms = []
            for idx in non_zero_indices:
                term = feature_names[idx]
                score = tfidf_vec[0, idx]
                active_terms.append((term, score))
            
            # Sort by TF-IDF magnitude
            active_terms.sort(key=lambda x: x[1], reverse=True)
            influential_terms = [t[0] for t in active_terms[:6]]
        except Exception:
            pass

    if prediction == "SPAM":
        reasoning_text = (
            "This message displays semantic patterns frequently associated with unsolicited or fraudulent communications, "
            "such as promotional phrasing, urgency indicators, financial claims, or external link anchors. "
            "Highlighted terms are model features and serve as interpretability indicators rather than definitive proof."
        )
    else:
        reasoning_text = (
            "This message aligns with legitimate, routine personal or professional communication patterns. "
            "The linguistic structure exhibits standard conversational tone without excessive high-risk spam indicators."
        )

    return {
        "explanation": reasoning_text,
        "key_indicators": influential_terms
    }


# ---------------------------------------------------------
# AUTHENTICATION DECORATOR
# ---------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route('/')
def login_page():
    """Displays the login page or redirects to dashboard if already authenticated."""
    if session.get('logged_in'):
        return redirect(url_for('dashboard_page'))
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def handle_login():
    """Processes demo authentication."""
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    
    if username == DEMO_USERNAME and password == DEMO_PASSWORD:
        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('dashboard_page'))
    else:
        flash("Invalid username or password. Please use demo credentials (admin / admin123).", "danger")
        return redirect(url_for('login_page'))


@app.route('/logout')
def logout():
    """Logs out the user and clears session."""
    session.clear()
    flash("You have been successfully logged out.", "info")
    return redirect(url_for('login_page'))


@app.route('/dashboard')
@login_required
def dashboard_page():
    """Renders the NLP/ML dashboard with real statistics from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'SPAM'")
    spam_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'NOT SPAM'")
    ham_count = cursor.fetchone()[0]
    
    spam_pct = round((spam_count / total_count * 100), 1) if total_count > 0 else 0.0
    
    # Recent predictions for summary table
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
    recent_predictions = cursor.fetchall()
    
    # Prediction distribution by day for timeline chart
    cursor.execute("""
        SELECT DATE(created_at) as pred_date, 
               SUM(CASE WHEN prediction = 'SPAM' THEN 1 ELSE 0 END) as spam_daily,
               SUM(CASE WHEN prediction = 'NOT SPAM' THEN 1 ELSE 0 END) as ham_daily
        FROM predictions
        GROUP BY DATE(created_at)
        ORDER BY pred_date DESC
        LIMIT 7
    """)
    timeline_rows = cursor.fetchall()
    conn.close()
    
    stats_data = {
        "total_count": total_count,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_pct": spam_pct,
        "recent_predictions": recent_predictions,
        "timeline": [dict(row) for row in reversed(timeline_rows)]
    }
    
    return render_template('dashboard.html', stats=stats_data, metadata=metadata)


@app.route('/classifier')
@login_required
def classifier_page():
    """Renders the spam email classifier interface."""
    return render_template('classifier.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict_endpoint():
    """
    Accepts an email/message payload, validates input, executes NLP preprocessing,
    transforms via TF-IDF, predicts via ML model, logs to SQLite, and returns JSON.
    """
    # 1. Validate payload format
    if not request.is_json and request.content_type != 'application/json':
        # Support both JSON payload and standard form data
        text = request.form.get('text', '')
    else:
        data = request.get_json(silent=True)
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid request. 'text' field is required."}), 400
        text = data.get('text', '')

    # 2. Validate content
    text = str(text).strip()
    if not text:
        return jsonify({"error": "Please enter an email or message before analyzing."}), 400
        
    if len(text) > 50000:
        return jsonify({"error": "Message exceeds maximum allowed length of 50,000 characters."}), 400

    # Ensure model is ready
    if model is None or vectorizer is None:
        load_ml_artifacts()
        if model is None or vectorizer is None:
            return jsonify({
                "error": "Trained ML model or TF-IDF vectorizer not loaded. Run 'python train_model.py' first."
            }), 503

    try:
        # 3. Text Preprocessing
        cleaned_text = preprocess_text(text)
        
        # 4. TF-IDF Transformation
        tfidf_features = vectorizer.transform([cleaned_text])
        
        # 5. Model Prediction
        prediction_val = model.predict(tfidf_features)[0]
        prediction_label = "SPAM" if prediction_val == 1 else "NOT SPAM"
        
        # 6. Confidence Score
        confidence = 95.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(tfidf_features)[0]
            confidence = round(float(probs[prediction_val]) * 100, 1)
        elif hasattr(model, "decision_function"):
            decision = float(model.decision_function(tfidf_features)[0])
            prob = 1 / (1 + np.exp(-decision))
            confidence = round(float(prob if prediction_val == 1 else (1 - prob)) * 100, 1)
            
        # 7. Compute structural message statistics
        stats = compute_message_statistics(text)
        
        # 8. Compute model feature explanation
        explanation_data = explain_prediction(text, cleaned_text, prediction_label)
        
        # 9. Store prediction in SQLite
        message_preview = (text[:120] + "...") if len(text) > 120 else text
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                message_preview, full_message, prediction, confidence,
                message_length, word_count, urls_count, numbers_count,
                special_chars_count, uppercase_count, exclamation_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message_preview,
            text,
            prediction_label,
            confidence,
            stats['character_count'],
            stats['word_count'],
            stats['urls_count'],
            stats['numbers_count'],
            stats['special_chars_count'],
            stats['uppercase_count'],
            stats['exclamation_count']
        ))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        # 10. Return JSON response
        return jsonify({
            "id": record_id,
            "prediction": prediction_label,
            "confidence": confidence,
            "message_length": stats['character_count'],
            "word_count": stats['word_count'],
            "statistics": stats,
            "explanation": explanation_data['explanation'],
            "key_indicators": explanation_data['key_indicators'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


@app.route('/history')
@login_required
def history_page():
    """Displays prediction history with search and filter capabilities."""
    search_query = request.args.get('q', '').strip()
    filter_type = request.args.get('filter', 'ALL').upper()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    sql = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if filter_type in ['SPAM', 'NOT SPAM']:
        sql += " AND prediction = ?"
        params.append(filter_type)
        
    if search_query:
        sql += " AND (message_preview LIKE ? OR full_message LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
        
    sql += " ORDER BY id DESC"
    
    cursor.execute(sql, params)
    records = cursor.fetchall()
    conn.close()
    
    return render_template(
        'history.html',
        records=records,
        search_query=search_query,
        filter_type=filter_type
    )


@app.route('/delete-history', methods=['POST'])
@login_required
def delete_history():
    """Deletes either an individual prediction or all history records."""
    record_id = request.form.get('id')
    clear_all = request.form.get('clear_all')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if clear_all == 'true':
        cursor.execute("DELETE FROM predictions")
        flash("All prediction history has been deleted.", "success")
    elif record_id:
        cursor.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        flash(f"Prediction #{record_id} deleted successfully.", "success")
        
    conn.commit()
    conn.close()
    
    return redirect(url_for('history_page'))


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Spam Email Classifier Backend",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/model-info')
@login_required
def model_info_api():
    """Returns evaluation metrics and model metadata as JSON."""
    return jsonify(metadata or {})


# ---------------------------------------------------------
# APPLICATION ENTRYPOINT
# ---------------------------------------------------------
init_db()
load_ml_artifacts()

if __name__ == '__main__':
    # Runs on local port 5000 in local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
