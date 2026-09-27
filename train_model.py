"""
Spam Email Classifier - Model Training Pipeline
Academic Project for Artificial Intelligence & Data Science
Author: AI & Data Science Student Project Team
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# 1. TEXT PREPROCESSING
# ---------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes email/message text for NLP feature extraction.
    Preserves key spam signals: URLs, emails, currencies, and numbers.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Normalize URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' url_token ', text)
    
    # Normalize email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_token ', text)
    
    # Normalize currency symbols
    text = re.sub(r'[\$£€¥₹]', ' currency_token ', text)
    
    # Normalize phone/toll-free numbers and digits
    text = re.sub(r'\b\d{1,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b', ' phone_token ', text)
    text = re.sub(r'\b\d+\b', ' number_token ', text)
    
    # Remove excessive punctuation while retaining word characters and tokens
    text = re.sub(r'[^\w\s_]', ' ', text)
    
    # Remove extraneous whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def main():
    print("=" * 70)
    print("   SPAM EMAIL CLASSIFIER - MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)

    # 1. Dataset Loading
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'spam_email_dataset.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    print(f"\n[1/8] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 2. Dataset Inspection
    print(f"\n[2/8] Dataset Inspection:")
    print(f"  - Total Rows: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    print("  - Sample Records:")
    print(df.head(3))
    
    # 3. Missing-value & Duplicate Check
    missing_count = df.isnull().sum().to_dict()
    print(f"\n[3/8] Data Quality Checks:")
    print(f"  - Missing Values: {missing_count}")
    df = df.dropna(subset=['label', 'text']).copy()
    
    duplicate_count = df.duplicated(subset=['text']).sum()
    print(f"  - Duplicate text records: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
        print(f"  - Cleaned dataset rows after deduplication: {len(df)}")
        
    # 4. Label Analysis & Class Imbalance Check
    label_counts = df['label'].str.lower().value_counts()
    print(f"\n[4/8] Class Distribution Analysis:")
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"  - {label.upper():8s}: {count:4d} samples ({pct:.1f}%)")
        
    # Encode labels: spam -> 1, ham -> 0
    df['clean_label'] = df['label'].str.lower().str.strip()
    df['target'] = df['clean_label'].map({'spam': 1, 'ham': 0})
    if df['target'].isnull().any():
        raise ValueError("Dataset contains labels other than 'spam' and 'ham'")

    # 5. Preprocessing
    print(f"\n[5/8] Applying Text Preprocessing...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # 6. Stratified Train-Test Split
    X = df['cleaned_text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25, 
        random_state=42, 
        stratify=y
    )
    print(f"  - Train Set Size: {len(X_train)} samples")
    print(f"  - Test Set Size : {len(X_test)} samples")
    
    # 7. TF-IDF Feature Extraction
    print(f"\n[6/8] Extracting Features using TfidfVectorizer...")
    # Configure TF-IDF parameters with academic rationale:
    # - sublinear_tf=True: applies sublinear scaling (1 + log(tf)) to prevent dominant terms
    # - ngram_range=(1, 2): captures unigrams and bigrams like 'free prize', 'wire transfer'
    # - min_df=1: accommodates demonstration dataset size
    # - max_df=0.95: eliminates corpus-universal words
    # - stop_words='english': filters standard english function words
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  - Vocabulary Size: {len(vectorizer.vocabulary_)} features")
    
    # 8. Model Training & Comparison
    print(f"\n[7/8] Model Training & Benchmark Comparison:")
    models_to_test = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            class_weight='balanced', 
            random_state=42
        ),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, class_weight='balanced'),
            cv=3
        )
    }
    
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model = None

    print("-" * 75)
    print(f"{'Model Name':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)

    for name, clf in models_to_test.items():
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm,
            "classifier_instance": clf
        }
        
        print(f"{name:<28} | {acc*100:>9.2f}% | {prec*100:>9.2f}% | {rec*100:>9.2f}% | {f1*100:>9.2f}%")
        
        # Select best model based on F1-score (harmonic mean of precision and recall)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = clf

    print("-" * 75)
    print(f"\n  ★ Selected Best Model: {best_model_name} (F1-Score: {best_f1*100:.2f}%)")

    # Detailed report for the best model
    best_y_pred = best_model.predict(X_test_tfidf)
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, best_y_pred, target_names=['HAM (Not Spam)', 'SPAM']))
    
    cm_best = confusion_matrix(y_test, best_y_pred)
    print(f"Confusion Matrix:")
    print(f"                 Predicted Ham   Predicted Spam")
    print(f"  Actual Ham          {cm_best[0][0]:<12d} {cm_best[0][1]:<12d}")
    print(f"  Actual Spam         {cm_best[1][0]:<12d} {cm_best[1][1]:<12d}")

    # 9. Serialization using Joblib
    print(f"\n[8/8] Serializing Trained Model & Vectorizer...")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'spam_classifier.pkl')
    vec_save_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    meta_json_path = os.path.join(models_dir, 'model_metadata.json')
    meta_pkl_path = os.path.join(models_dir, 'model_metadata.pkl')
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(vectorizer, vec_save_path)
    
    metadata = {
        "model_name": best_model_name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": results[best_model_name]["accuracy"],
        "precision": results[best_model_name]["precision"],
        "recall": results[best_model_name]["recall"],
        "f1_score": results[best_model_name]["f1_score"],
        "confusion_matrix": results[best_model_name]["confusion_matrix"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(df),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "all_models_benchmark": {
            k: {
                "accuracy": v["accuracy"],
                "precision": v["precision"],
                "recall": v["recall"],
                "f1_score": v["f1_score"],
                "confusion_matrix": v["confusion_matrix"]
            } for k, v in results.items()
        }
    }
    
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    joblib.dump(metadata, meta_pkl_path)
    
    print(f"  ✓ Saved Model: {model_save_path}")
    print(f"  ✓ Saved Vectorizer: {vec_save_path}")
    print(f"  ✓ Saved Metadata: {meta_json_path} & {meta_pkl_path}")
    print("\nTraining completed successfully! Ready for Flask backend.")
    print("=" * 70)


if __name__ == '__main__':
    main()
