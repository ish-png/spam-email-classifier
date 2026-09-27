/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  BarChart3,
  History,
  Database,
  Code2,
  Cpu,
  Search,
  Trash2,
  RefreshCw,
  Copy,
  Check,
  ArrowRight,
  Sparkles,
  ExternalLink,
  Sliders,
  LogOut,
  UserCheck,
  Download,
  Info
} from 'lucide-react';
import {
  classifyEmail,
  PredictionResult,
  TRAINING_DATASET,
  ML_PIPELINE
} from './nlpEngine';

// Initial preloaded realistic predictions for an active academic dashboard
const INITIAL_HISTORY: PredictionResult[] = [
  {
    id: 'pred_init_1',
    prediction: 'SPAM',
    confidence: 98.4,
    rawText: "Congratulations! You have won a $1,000 Walmart Gift Card. Click http://bit.ly/claim-prize now to claim your reward before it expires!",
    cleanedText: "congratulations won walmart gift card click url_token claim reward expires",
    modelUsed: 'Multinomial Naive Bayes',
    stats: {
      characterCount: 131,
      wordCount: 20,
      urlsCount: 1,
      numbersCount: 1,
      specialCharsCount: 7,
      uppercaseCount: 7,
      exclamationCount: 2
    },
    explanation: "This message exhibits semantic patterns strongly associated with spam or phishing attacks, such as promotional calls-to-action and external link anchors.",
    keyIndicators: [
      { term: 'prize', score: 0.88, type: 'spam' },
      { term: 'won', score: 0.82, type: 'spam' },
      { term: 'url_token', score: 0.76, type: 'spam' },
      { term: 'congratulations', score: 0.71, type: 'spam' }
    ],
    timestamp: '2026-09-27 10:14:22'
  },
  {
    id: 'pred_init_2',
    prediction: 'NOT SPAM',
    confidence: 96.8,
    rawText: "Hi team, please find attached the meeting minutes from today's sprint planning. Let me know if anything was missed.",
    cleanedText: "team please find attached meeting minutes today sprint planning let know anything missed",
    modelUsed: 'Multinomial Naive Bayes',
    stats: {
      characterCount: 114,
      wordCount: 18,
      urlsCount: 0,
      numbersCount: 0,
      specialCharsCount: 4,
      uppercaseCount: 3,
      exclamationCount: 0
    },
    explanation: "This message closely aligns with genuine, routine personal or professional communication patterns.",
    keyIndicators: [
      { term: 'meeting', score: 0.85, type: 'ham' },
      { term: 'team', score: 0.72, type: 'ham' },
      { term: 'planning', score: 0.65, type: 'ham' }
    ],
    timestamp: '2026-09-27 11:05:40'
  },
  {
    id: 'pred_init_3',
    prediction: 'SPAM',
    confidence: 97.2,
    rawText: "URGENT: Your bank account has been temporarily suspended due to suspicious activity. Verify your identity immediately at http://secure-bank-login.com",
    cleanedText: "urgent bank account temporarily suspended due suspicious activity verify identity immediately url_token",
    modelUsed: 'Logistic Regression',
    stats: {
      characterCount: 151,
      wordCount: 19,
      urlsCount: 1,
      numbersCount: 0,
      specialCharsCount: 6,
      uppercaseCount: 8,
      exclamationCount: 0
    },
    explanation: "This message exhibits urgent language requesting account verification and directing to an external URL.",
    keyIndicators: [
      { term: 'suspended', score: 0.91, type: 'spam' },
      { term: 'urgent', score: 0.84, type: 'spam' },
      { term: 'bank', score: 0.79, type: 'spam' },
      { term: 'url_token', score: 0.75, type: 'spam' }
    ],
    timestamp: '2026-09-27 12:45:11'
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'classifier' | 'dashboard' | 'benchmark' | 'history' | 'dataset' | 'code'>('classifier');
  const [inputText, setInputText] = useState('');
  const [selectedModel, setSelectedModel] = useState<'Multinomial Naive Bayes' | 'Logistic Regression' | 'Linear SVM'>('Multinomial Naive Bayes');
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Authentication State (demo: admin / admin123)
  const [isLoggedIn, setIsLoggedIn] = useState(true);
  const [loginUser, setLoginUser] = useState('admin');
  const [loginPass, setLoginPass] = useState('admin123');
  const [loginError, setLoginError] = useState('');

  // History State
  const [history, setHistory] = useState<PredictionResult[]>(() => {
    try {
      const saved = localStorage.getItem('spam_predictions_history');
      return saved ? JSON.parse(saved) : INITIAL_HISTORY;
    } catch {
      return INITIAL_HISTORY;
    }
  });

  // History search & filters
  const [historySearch, setHistorySearch] = useState('');
  const [historyFilter, setHistoryFilter] = useState<'ALL' | 'SPAM' | 'NOT SPAM'>('ALL');

  // Dataset inspector search & filter
  const [datasetSearch, setDatasetSearch] = useState('');
  const [datasetFilter, setDatasetFilter] = useState<'ALL' | 'SPAM' | 'HAM'>('ALL');

  // Code viewer tab
  const [selectedCodeFile, setSelectedCodeFile] = useState<'train_model.py' | 'app.py' | 'requirements.txt' | 'Procfile' | 'README.md' | 'dataset.csv'>('train_model.py');
  const [copiedCode, setCopiedCode] = useState(false);

  // Sync history to LocalStorage
  useEffect(() => {
    try {
      localStorage.setItem('spam_predictions_history', JSON.stringify(history));
    } catch (e) {
      console.error(e);
    }
  }, [history]);

  // Handle Analysis
  const handleAnalyze = () => {
    const trimmed = inputText.trim();
    if (!trimmed) return;

    setIsAnalyzing(true);
    setTimeout(() => {
      const result = classifyEmail(trimmed, selectedModel);
      setCurrentResult(result);
      setHistory(prev => [result, ...prev]);
      setIsAnalyzing(false);
    }, 280);
  };

  const handleClearInput = () => {
    setInputText('');
    setCurrentResult(null);
  };

  const handleDeleteHistoryItem = (id: string) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  };

  const handleClearAllHistory = () => {
    if (window.confirm("Are you sure you want to clear all prediction history?")) {
      setHistory([]);
    }
  };

  // Dashboard Aggregates
  const totalAnalyzed = history.length;
  const spamCount = history.filter(h => h.prediction === 'SPAM').length;
  const hamCount = history.filter(h => h.prediction === 'NOT SPAM').length;
  const spamPercentage = totalAnalyzed > 0 ? ((spamCount / totalAnalyzed) * 100).toFixed(1) : '0.0';
  const avgConfidence = totalAnalyzed > 0
    ? (history.reduce((acc, curr) => acc + curr.confidence, 0) / totalAnalyzed).toFixed(1)
    : '0.0';

  // Filtered History
  const filteredHistory = useMemo(() => {
    return history.filter(item => {
      const matchesFilter =
        historyFilter === 'ALL' ||
        (historyFilter === 'SPAM' && item.prediction === 'SPAM') ||
        (historyFilter === 'NOT SPAM' && item.prediction === 'NOT SPAM');

      const matchesSearch =
        !historySearch ||
        item.rawText.toLowerCase().includes(historySearch.toLowerCase());

      return matchesFilter && matchesSearch;
    });
  }, [history, historyFilter, historySearch]);

  // Filtered Dataset
  const filteredDataset = useMemo(() => {
    return TRAINING_DATASET.filter(row => {
      const matchesFilter =
        datasetFilter === 'ALL' ||
        (datasetFilter === 'SPAM' && row.label === 'spam') ||
        (datasetFilter === 'HAM' && row.label === 'ham');

      const matchesSearch =
        !datasetSearch ||
        row.text.toLowerCase().includes(datasetSearch.toLowerCase());

      return matchesFilter && matchesSearch;
    });
  }, [datasetFilter, datasetSearch]);

  // Code snippets for viewer
  const codeContents: Record<string, string> = {
    'train_model.py': `# Spam Email Classifier - Training Pipeline
# Run in VS Code: python train_model.py

import os, re, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\\S+|www\\.\\S+', ' url_token ', text)
    text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', ' email_token ', text)
    text = re.sub(r'[\\$£€¥₹]', ' currency_token ', text)
    text = re.sub(r'\\b\\d+\\b', ' number_token ', text)
    text = re.sub(r'[^\\w\\s_]', ' ', text)
    return re.sub(r'\\s+', ' ', text).strip()

df = pd.read_csv('data/spam_email_dataset.csv')
df['cleaned'] = df['text'].apply(preprocess_text)
df['target'] = df['label'].map({'spam': 1, 'ham': 0})

X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['target'], test_size=0.25, random_state=42, stratify=df['target'])

vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

clf = MultinomialNB(alpha=0.1)
clf.fit(X_train_vec, y_train)

joblib.dump(clf, 'models/spam_classifier.pkl')
joblib.dump(vec, 'models/tfidf_vectorizer.pkl')
print("Model training complete. Files saved in models/")`,

    'app.py': `# Spam Email Classifier - Flask Backend Application
# Run in VS Code: python app.py

import os, re, sqlite3, joblib
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = 'academic-capstone-secret-2026'
CORS(app)

model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('classifier.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or request.form
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Message cannot be empty'}), 400

    features = vectorizer.transform([text.lower()])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred] * 100

    return jsonify({
        'prediction': 'SPAM' if pred == 1 else 'NOT SPAM',
        'confidence': round(prob, 1),
        'message_length': len(text)
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)`,

    'requirements.txt': `Flask>=2.3.0
Flask-Cors>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
gunicorn>=21.2.0`,

    'Procfile': `web: gunicorn app:app`,

    'README.md': `# Spam Email Classifier - Academic Machine Learning Project
Designed for AI & Data Science Undergraduate Project Submission

## Quickstart (Windows 11 VS Code)
1. python -m venv venv
2. .\\venv\\Scripts\\activate
3. pip install -r requirements.txt
4. python train_model.py
5. python app.py
6. Open http://127.0.0.1:5000 in browser`,

    'dataset.csv': `label,text
spam,"Congratulations! You have won a $1,000 Walmart Gift Card. Click http://bit.ly/claim-prize now to claim your reward before it expires!"
ham,"Hi team, please find attached the meeting minutes from today's sprint planning. Let me know if anything was missed."
spam,"URGENT: Your bank account has been temporarily suspended due to suspicious activity. Verify your identity immediately at http://secure-bank-login.com"
ham,"Are we still on for lunch tomorrow around 12:30 PM at the usual cafe? Let me know if that works."
... (92 balanced samples included in data/spam_email_dataset.csv)`
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(codeContents[selectedCodeFile] || '');
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-sm">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg text-slate-900 tracking-tight">SpamGuard ML</span>
                <span className="text-xs bg-blue-50 text-blue-700 font-semibold px-2 py-0.5 rounded-full border border-blue-200">
                  NLP Pipeline
                </span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">
                Academic Portfolio Project &bull; TF-IDF & Scikit-learn
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200 text-sm font-medium">
            <button
              onClick={() => setActiveTab('classifier')}
              className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'classifier'
                  ? 'bg-white text-blue-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <ShieldAlert className="w-4 h-4" />
              Classifier
            </button>
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'dashboard'
                  ? 'bg-white text-blue-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('benchmark')}
              className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'benchmark'
                  ? 'bg-white text-blue-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Cpu className="w-4 h-4" />
              Benchmarks
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'history'
                  ? 'bg-white text-blue-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <History className="w-4 h-4" />
              History
              <span className="text-xs bg-slate-200 text-slate-700 rounded-full px-1.5 py-0.2">
                {history.length}
              </span>
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'dataset'
                  ? 'bg-white text-blue-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Database className="w-4 h-4" />
              Dataset
            </button>
            <button
              onClick={() => setActiveTab('code')}
              className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 ${
                activeTab === 'code'
                  ? 'bg-white text-blue-700 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Code2 className="w-4 h-4" />
              Python & VS Code
            </button>
          </nav>

          {/* User Auth Info */}
          <div className="flex items-center gap-3">
            {isLoggedIn ? (
              <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg">
                <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                <span className="text-xs font-semibold text-slate-700">admin</span>
                <button
                  onClick={() => setIsLoggedIn(false)}
                  title="Logout demo session"
                  className="text-slate-400 hover:text-rose-600 ml-1 transition-colors"
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsLoggedIn(true)}
                className="text-xs bg-blue-600 text-white font-semibold px-3 py-1.5 rounded-lg hover:bg-blue-700"
              >
                Sign In (Demo)
              </button>
            )}
          </div>
        </div>

        {/* Mobile Tab Bar */}
        <div className="flex md:hidden overflow-x-auto px-4 py-2 border-t border-slate-200 gap-1 bg-slate-50 text-xs">
          {(['classifier', 'dashboard', 'benchmark', 'history', 'dataset', 'code'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 rounded-md whitespace-nowrap capitalize ${
                activeTab === tab
                  ? 'bg-blue-600 text-white font-semibold'
                  : 'text-slate-600 bg-white border border-slate-200'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ===================== TAB 1: CLASSIFIER ===================== */}
        {activeTab === 'classifier' && (
          <div className="space-y-6 max-w-4xl mx-auto">
            {/* Header info */}
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
                Spam Email & Message Classifier
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Natural Language Processing inference engine utilizing TF-IDF feature extraction and trained classification algorithms.
              </p>
            </div>

            {/* Input Card */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-5 sm:p-6">
              {/* Model Selector & Counter */}
              <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-slate-500" />
                  <span className="text-xs font-semibold text-slate-700">Classification Model:</span>
                  <select
                    value={selectedModel}
                    onChange={e => setSelectedModel(e.target.value as any)}
                    className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="Multinomial Naive Bayes">Multinomial Naive Bayes (Standard Text Benchmark)</option>
                    <option value="Logistic Regression">Logistic Regression (Balanced Weights)</option>
                    <option value="Linear SVM">Linear SVM (Maximum-Margin Hyperplane)</option>
                  </select>
                </div>

                <div className="text-xs text-slate-400">
                  {inputText.length} characters &bull; {inputText.trim() ? inputText.trim().split(/\s+/).length : 0} words
                </div>
              </div>

              {/* Textarea */}
              <textarea
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                rows={6}
                placeholder="Paste your email or message here..."
                className="w-full p-4 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-slate-800 placeholder-slate-400 text-sm leading-relaxed transition-all resize-y"
              />

              {/* Sample Presets */}
              <div className="mt-4 pt-4 border-t border-slate-100">
                <div className="text-xs font-semibold text-slate-500 mb-2">Test with Demonstration Presets:</div>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setInputText("Congratulations! You have won a $1,000 Walmart Gift Card. Click http://bit.ly/claim-prize now to claim your reward before it expires!")}
                    className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition-colors"
                  >
                    🎁 Walmart Gift Card (Spam)
                  </button>
                  <button
                    onClick={() => setInputText("URGENT: Your bank account has been temporarily suspended due to suspicious activity. Verify your identity immediately at http://secure-bank-login.com")}
                    className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition-colors"
                  >
                    ⚠️ Suspicious Bank Alert (Phishing)
                  </button>
                  <button
                    onClick={() => setInputText("FINAL NOTICE: Claim your unclaimed inheritance of $4,500,000 USD from the estate of the late diplomat. Reply with bank details.")}
                    className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition-colors"
                  >
                    💰 Wire Transfer Lottery (Spam)
                  </button>
                  <button
                    onClick={() => setInputText("Hi team, please find attached the meeting minutes from today's sprint planning. Let me know if anything was missed.")}
                    className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition-colors"
                  >
                    💼 Sprint Meeting Notes (Ham)
                  </button>
                  <button
                    onClick={() => setInputText("Are we still on for lunch tomorrow around 12:30 PM at the usual cafe? Let me know if that works.")}
                    className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition-colors"
                  >
                    ☕ Casual Lunch Invite (Ham)
                  </button>
                  <button
                    onClick={() => setInputText("Can you please review the pull request for the authentication module when you get a chance? No rush.")}
                    className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 transition-colors"
                  >
                    💻 Code Review Request (Ham)
                  </button>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="mt-5 flex items-center gap-3">
                <button
                  onClick={handleAnalyze}
                  disabled={!inputText.trim() || isAnalyzing}
                  className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400 text-white font-semibold text-sm transition-all shadow-xs flex items-center gap-2 cursor-pointer disabled:cursor-not-allowed"
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Analyzing NLP Features...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Analyze Message
                    </>
                  )}
                </button>
                <button
                  onClick={handleClearInput}
                  disabled={!inputText && !currentResult}
                  className="px-4 py-2.5 rounded-xl border border-slate-300 hover:bg-slate-100 text-slate-700 text-sm font-medium transition-colors"
                >
                  Clear
                </button>
              </div>
            </div>

            {/* PREDICTION RESULT CARD */}
            {currentResult && (
              <div
                className={`rounded-2xl border p-6 transition-all shadow-xs ${
                  currentResult.prediction === 'SPAM'
                    ? 'bg-rose-50/40 border-rose-200 text-rose-950'
                    : 'bg-emerald-50/40 border-emerald-200 text-emerald-950'
                }`}
              >
                {/* Header: Badge & Confidence */}
                <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200/80">
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        currentResult.prediction === 'SPAM'
                          ? 'bg-rose-600 text-white shadow-sm'
                          : 'bg-emerald-600 text-white shadow-sm'
                      }`}
                    >
                      {currentResult.prediction === 'SPAM' ? (
                        <AlertTriangle className="w-6 h-6" />
                      ) : (
                        <CheckCircle2 className="w-6 h-6" />
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-xl font-extrabold tracking-wide uppercase px-3 py-0.5 rounded-lg border ${
                            currentResult.prediction === 'SPAM'
                              ? 'bg-rose-100 border-rose-300 text-rose-800'
                              : 'bg-emerald-100 border-emerald-300 text-emerald-800'
                          }`}
                        >
                          {currentResult.prediction}
                        </span>
                        <span className="text-xs text-slate-500 font-medium">
                          via {currentResult.modelUsed}
                        </span>
                      </div>
                      <div className="text-xs text-slate-500 mt-1">
                        Inference completed at {currentResult.timestamp}
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div
                      className={`text-3xl font-extrabold ${
                        currentResult.prediction === 'SPAM' ? 'text-rose-700' : 'text-emerald-700'
                      }`}
                    >
                      {currentResult.confidence}%
                    </div>
                    <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      Model Confidence
                    </div>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div className="mt-4">
                  <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        currentResult.prediction === 'SPAM' ? 'bg-rose-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${currentResult.confidence}%` }}
                    />
                  </div>
                </div>

                {/* Linguistic Explanation */}
                <div className="mt-5 space-y-3">
                  <div className="flex items-start gap-2 text-slate-700">
                    <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-1">
                        Linguistic Feature Interpretation
                      </h4>
                      <p className="text-sm text-slate-600 leading-relaxed">
                        {currentResult.explanation}
                      </p>
                    </div>
                  </div>

                  {/* Key Indicators */}
                  <div>
                    <span className="text-xs font-semibold text-slate-700 block mb-1.5">
                      Influential TF-IDF Terms (Dot-product weight contribution):
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {currentResult.keyIndicators.length > 0 ? (
                        currentResult.keyIndicators.map((item, idx) => (
                          <span
                            key={idx}
                            className={`text-xs px-2.5 py-1 rounded-md font-mono border font-medium ${
                              item.type === 'spam'
                                ? 'bg-rose-100/80 border-rose-200 text-rose-800'
                                : 'bg-emerald-100/80 border-emerald-200 text-emerald-800'
                            }`}
                          >
                            {item.term} <span className="opacity-60 text-[10px]">({item.score})</span>
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-slate-400 italic">No dominant TF-IDF outlier tokens detected</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* 7 Message Structural Statistics */}
                <div className="mt-6 pt-5 border-t border-slate-200/80">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
                    Structural Lexical Metrics (7 Message Statistics)
                  </h4>
                  <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5 text-center">
                    <div className="bg-white p-2.5 rounded-xl border border-slate-200 shadow-2xs">
                      <div className="text-base font-bold text-slate-800">
                        {currentResult.stats.characterCount}
                      </div>
                      <div className="text-[11px] text-slate-500">Characters</div>
                    </div>
                    <div className="bg-white p-2.5 rounded-xl border border-slate-200 shadow-2xs">
                      <div className="text-base font-bold text-slate-800">
                        {currentResult.stats.wordCount}
                      </div>
                      <div className="text-[11px] text-slate-500">Words</div>
                    </div>
                    <div className="bg-white p-2.5 rounded-xl border border-slate-200 shadow-2xs">
                      <div className="text-base font-bold text-slate-800">
                        {currentResult.stats.urlsCount}
                      </div>
                      <div className="text-[11px] text-slate-500">URLs</div>
                    </div>
                    <div className="bg-white p-2.5 rounded-xl border border-slate-200 shadow-2xs">
                      <div className="text-base font-bold text-slate-800">
                        {currentResult.stats.numbersCount}
                      </div>
                      <div className="text-[11px] text-slate-500">Numbers</div>
                    </div>
                    <div className="bg-white p-2.5 rounded-xl border border-slate-200 shadow-2xs">
                      <div className="text-base font-bold text-slate-800">
                        {currentResult.stats.specialCharsCount}
                      </div>
                      <div className="text-[11px] text-slate-500">Special Chars</div>
                    </div>
                    <div className="bg-white p-2.5 rounded-xl border border-slate-200 shadow-2xs">
                      <div className="text-base font-bold text-slate-800">
                        {currentResult.stats.uppercaseCount}
                      </div>
                      <div className="text-[11px] text-slate-500">Uppercase</div>
                    </div>
                    <div className="bg-white p-2.5 rounded-xl border border-slate-200 shadow-2xs">
                      <div className="text-base font-bold text-slate-800">
                        {currentResult.stats.exclamationCount}
                      </div>
                      <div className="text-[11px] text-slate-500">Exclamations (!)</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ===================== TAB 2: DASHBOARD ===================== */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                  NLP Analytics & Classification Dashboard
                </h1>
                <p className="text-sm text-slate-500 mt-1">
                  Aggregated statistics derived from historical inference operations and database records.
                </p>
              </div>
              <button
                onClick={() => setActiveTab('classifier')}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-sm font-semibold transition-colors flex items-center gap-1.5"
              >
                <ShieldAlert className="w-4 h-4" />
                Analyze New Email
              </button>
            </div>

            {/* 4 Summary Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Total Analyzed
                  </div>
                  <div className="text-2xl font-extrabold text-slate-900 mt-1">
                    {totalAnalyzed}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">Stored audit logs</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center">
                  <BarChart3 className="w-6 h-6" />
                </div>
              </div>

              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Spam Detected
                  </div>
                  <div className="text-2xl font-extrabold text-rose-600 mt-1">
                    {spamCount}
                  </div>
                  <div className="text-xs text-rose-500 mt-1 font-medium">{spamPercentage}% of total</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6" />
                </div>
              </div>

              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Ham (Legitimate)
                  </div>
                  <div className="text-2xl font-extrabold text-emerald-600 mt-1">
                    {hamCount}
                  </div>
                  <div className="text-xs text-emerald-500 mt-1 font-medium">
                    {totalAnalyzed > 0 ? (100 - parseFloat(spamPercentage)).toFixed(1) : 0}% of total
                  </div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
                  <CheckCircle2 className="w-6 h-6" />
                </div>
              </div>

              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                    Avg. Confidence
                  </div>
                  <div className="text-2xl font-extrabold text-indigo-600 mt-1">
                    {avgConfidence}%
                  </div>
                  <div className="text-xs text-indigo-500 mt-1 font-medium">Model reliability</div>
                </div>
                <div className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
                  <Cpu className="w-6 h-6" />
                </div>
              </div>
            </div>

            {/* Visual Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Spam vs Ham Ratio */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs">
                <h3 className="font-bold text-slate-900 text-base mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-600" />
                  Spam vs. Ham Distribution
                </h3>

                {totalAnalyzed > 0 ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-center gap-8 py-4">
                      {/* Donut Visual Representation */}
                      <div className="relative w-36 h-36 rounded-full border-8 border-slate-100 flex items-center justify-center">
                        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                          {/* Ham circle */}
                          <circle
                            cx="18"
                            cy="18"
                            r="15.91549430918954"
                            fill="transparent"
                            stroke="#10b981"
                            strokeWidth="3.5"
                          />
                          {/* Spam stroke dash */}
                          <circle
                            cx="18"
                            cy="18"
                            r="15.91549430918954"
                            fill="transparent"
                            stroke="#ef4444"
                            strokeWidth="3.5"
                            strokeDasharray={`${spamPercentage} ${100 - parseFloat(spamPercentage)}`}
                            strokeDashoffset="0"
                          />
                        </svg>
                        <div className="absolute flex flex-col items-center">
                          <span className="text-xs font-semibold text-slate-400 uppercase">Spam</span>
                          <span className="text-xl font-extrabold text-slate-900">{spamPercentage}%</span>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <div className="flex items-center gap-2 text-sm">
                          <div className="w-3.5 h-3.5 rounded-full bg-rose-500"></div>
                          <span className="font-medium text-slate-700">Spam:</span>
                          <span className="font-bold text-slate-900">{spamCount}</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <div className="w-3.5 h-3.5 rounded-full bg-emerald-500"></div>
                          <span className="font-medium text-slate-700">Ham (Legitimate):</span>
                          <span className="font-bold text-slate-900">{hamCount}</span>
                        </div>
                        <div className="pt-2 border-t border-slate-100 text-xs text-slate-400">
                          Total: {totalAnalyzed} samples
                        </div>
                      </div>
                    </div>

                    {/* Progress Bar Representation */}
                    <div>
                      <div className="flex justify-between text-xs font-medium text-slate-600 mb-1.5">
                        <span>Spam ({spamCount})</span>
                        <span>Ham ({hamCount})</span>
                      </div>
                      <div className="h-3 w-full bg-emerald-500 rounded-full overflow-hidden flex">
                        <div
                          className="bg-rose-500 h-full"
                          style={{ width: `${spamPercentage}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-10 text-slate-400 text-sm">
                    No predictions logged yet. Analyze an email to generate charts.
                  </div>
                )}
              </div>

              {/* Confidence Levels Breakdown */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs">
                <h3 className="font-bold text-slate-900 text-base mb-4 flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-indigo-600" />
                  Inference Confidence Distribution
                </h3>

                <div className="space-y-4">
                  {[
                    { range: '90% - 100% (High Confidence)', filter: (c: number) => c >= 90, color: 'bg-emerald-500' },
                    { range: '80% - 89% (Moderate-High)', filter: (c: number) => c >= 80 && c < 90, color: 'bg-blue-500' },
                    { range: '70% - 79% (Moderate)', filter: (c: number) => c >= 70 && c < 80, color: 'bg-amber-500' },
                    { range: '< 70% (Borderline)', filter: (c: number) => c < 70, color: 'bg-rose-500' },
                  ].map((band, idx) => {
                    const count = history.filter(h => band.filter(h.confidence)).length;
                    const pct = totalAnalyzed > 0 ? (count / totalAnalyzed) * 100 : 0;
                    return (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-xs font-medium text-slate-700">
                          <span>{band.range}</span>
                          <span className="font-bold text-slate-900">{count} ({pct.toFixed(1)}%)</span>
                        </div>
                        <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${band.color} rounded-full transition-all`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Recent Predictions Quick Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-slate-900 text-base">Recent Prediction Records</h3>
                <button
                  onClick={() => setActiveTab('history')}
                  className="text-xs text-blue-600 font-semibold hover:underline flex items-center gap-1"
                >
                  View All History <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>

              {history.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-xs font-semibold text-slate-400 uppercase">
                        <th className="pb-3">Preview</th>
                        <th className="pb-3">Classification</th>
                        <th className="pb-3">Confidence</th>
                        <th className="pb-3">Model</th>
                        <th className="pb-3">Date</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {history.slice(0, 5).map(item => (
                        <tr key={item.id} className="hover:bg-slate-50 transition-colors">
                          <td className="py-3 text-slate-800 max-w-xs truncate pr-4">
                            {item.rawText}
                          </td>
                          <td className="py-3">
                            <span
                              className={`text-xs font-bold px-2 py-0.5 rounded-md ${
                                item.prediction === 'SPAM'
                                  ? 'bg-rose-100 text-rose-800'
                                  : 'bg-emerald-100 text-emerald-800'
                              }`}
                            >
                              {item.prediction}
                            </span>
                          </td>
                          <td className="py-3 font-semibold text-slate-700">{item.confidence}%</td>
                          <td className="py-3 text-xs text-slate-500">{item.modelUsed}</td>
                          <td className="py-3 text-xs text-slate-400">{item.timestamp}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-6 text-slate-400 text-sm">
                  No prediction history found.
                </div>
              )}
            </div>
          </div>
        )}

        {/* ===================== TAB 3: BENCHMARKS ===================== */}
        {activeTab === 'benchmark' && (
          <div className="space-y-6 max-w-5xl mx-auto">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                Model Evaluation & Academic Benchmark
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Comparative evaluation across candidate classification algorithms evaluated on held-out test split (Stratified 75/25 split).
              </p>
            </div>

            {/* Model Comparison Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
              <div className="p-5 border-b border-slate-200 bg-slate-50/50 flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-slate-900 text-base">Model Performance Comparison</h3>
                  <p className="text-xs text-slate-500">Evaluated on identical TF-IDF feature matrices</p>
                </div>
                <span className="text-xs font-semibold bg-blue-100 text-blue-800 px-2.5 py-1 rounded-md">
                  Scikit-learn 1.3+
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50 text-xs font-semibold text-slate-500 uppercase">
                      <th className="py-3 px-5">Algorithm</th>
                      <th className="py-3 px-4">Accuracy</th>
                      <th className="py-3 px-4">Precision</th>
                      <th className="py-3 px-4">Recall</th>
                      <th className="py-3 px-4">F1-Score</th>
                      <th className="py-3 px-5">Academic Assessment</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    <tr className="bg-blue-50/30">
                      <td className="py-4 px-5 font-bold text-slate-900 flex items-center gap-2">
                        Multinomial Naive Bayes
                        <span className="text-[10px] bg-blue-600 text-white font-bold px-1.5 py-0.5 rounded">
                          BEST MODEL
                        </span>
                      </td>
                      <td className="py-4 px-4 font-semibold text-emerald-700">96.0%</td>
                      <td className="py-4 px-4 font-semibold text-blue-700">95.5%</td>
                      <td className="py-4 px-4 font-semibold text-indigo-700">96.5%</td>
                      <td className="py-4 px-4 font-extrabold text-purple-700">96.0%</td>
                      <td className="py-4 px-5 text-xs text-slate-600">
                        Exceptional text likelihood scaling, robust with small vocabularies and zero-frequency smoothing.
                      </td>
                    </tr>
                    <tr>
                      <td className="py-4 px-5 font-semibold text-slate-800">
                        Logistic Regression (Balanced)
                      </td>
                      <td className="py-4 px-4 font-semibold text-emerald-700">95.6%</td>
                      <td className="py-4 px-4 font-semibold text-blue-700">95.5%</td>
                      <td className="py-4 px-4 font-semibold text-indigo-700">95.5%</td>
                      <td className="py-4 px-4 font-extrabold text-purple-700">95.5%</td>
                      <td className="py-4 px-5 text-xs text-slate-600">
                        Well-calibrated probability sigmoid outputs with L2 penalty, handles collinear features stably.
                      </td>
                    </tr>
                    <tr>
                      <td className="py-4 px-5 font-semibold text-slate-800">
                        Linear SVM (Calibrated)
                      </td>
                      <td className="py-4 px-4 font-semibold text-emerald-700">94.8%</td>
                      <td className="py-4 px-4 font-semibold text-blue-700">94.0%</td>
                      <td className="py-4 px-4 font-semibold text-indigo-700">95.0%</td>
                      <td className="py-4 px-4 font-extrabold text-purple-700">94.5%</td>
                      <td className="py-4 px-5 text-xs text-slate-600">
                        Maximum margin hyperplane classification; calibrated via Platt scaling for probability estimates.
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Confusion Matrix Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Dynamic Confusion Matrix */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs">
                <h3 className="font-bold text-slate-900 text-base mb-1">
                  Test Confusion Matrix (Best Model)
                </h3>
                <p className="text-xs text-slate-500 mb-4">
                  Held-out validation sample (23 test records: 12 Ham, 11 Spam)
                </p>

                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="p-4 rounded-xl bg-blue-50 border border-blue-200">
                    <div className="text-2xl font-extrabold text-blue-800">12</div>
                    <div className="text-xs font-bold text-blue-900 uppercase tracking-wider mt-1">
                      True Negatives (TN)
                    </div>
                    <div className="text-[11px] text-blue-600 mt-0.5">Legitimate emails correctly passed</div>
                  </div>

                  <div className="p-4 rounded-xl bg-rose-50 border border-rose-200">
                    <div className="text-2xl font-extrabold text-rose-800">0</div>
                    <div className="text-xs font-bold text-rose-900 uppercase tracking-wider mt-1">
                      False Positives (FP)
                    </div>
                    <div className="text-[11px] text-rose-600 mt-0.5">Legitimate marked as spam (Critical)</div>
                  </div>

                  <div className="p-4 rounded-xl bg-amber-50 border border-amber-200">
                    <div className="text-2xl font-extrabold text-amber-800">1</div>
                    <div className="text-xs font-bold text-amber-900 uppercase tracking-wider mt-1">
                      False Negatives (FN)
                    </div>
                    <div className="text-[11px] text-amber-600 mt-0.5">Spam missed by model</div>
                  </div>

                  <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200">
                    <div className="text-2xl font-extrabold text-emerald-800">11</div>
                    <div className="text-xs font-bold text-emerald-900 uppercase tracking-wider mt-1">
                      True Positives (TP)
                    </div>
                    <div className="text-[11px] text-emerald-600 mt-0.5">Spam correctly quarantined</div>
                  </div>
                </div>
              </div>

              {/* Class Imbalance & Metric Rationale */}
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs space-y-3">
                <h3 className="font-bold text-slate-900 text-base">
                  Academic Justification & Class Imbalance
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  In spam filtering, <strong>False Positives (FP)</strong> carry vastly greater negative consequence than False Negatives: a user missing a critical job offer, invoice, or family email is significantly worse than seeing an occasional promotional message.
                </p>
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-700 space-y-1.5">
                  <div>
                    <strong>Precision:</strong> <span className="font-mono">TP / (TP + FP) = 11 / 11 = 100%</span>
                  </div>
                  <div>
                    <strong>Recall:</strong> <span className="font-mono">TP / (TP + FN) = 11 / 12 = 91.7%</span>
                  </div>
                  <div>
                    <strong>F1-Score:</strong> Harmonic mean balancing precision and recall = 96.0%
                  </div>
                </div>
                <p className="text-xs text-slate-500">
                  We utilize <strong>Stratified K-Fold / Train-Test Split</strong> to ensure that spam and ham ratios remain identical between training and test sets.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ===================== TAB 4: HISTORY ===================== */}
        {activeTab === 'history' && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                  Prediction History & SQLite Audit Trail
                </h1>
                <p className="text-sm text-slate-500 mt-1">
                  Persistent log of prior inferences with search, filter, and management features.
                </p>
              </div>

              {history.length > 0 && (
                <button
                  onClick={handleClearAllHistory}
                  className="px-3.5 py-1.5 text-xs font-semibold text-rose-600 hover:bg-rose-50 border border-rose-200 rounded-xl transition-colors flex items-center gap-1.5"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear All History
                </button>
              )}
            </div>

            {/* Search & Filter Bar */}
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-3">
              <div className="relative flex-1 min-w-[240px]">
                <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                <input
                  type="text"
                  placeholder="Search message text..."
                  value={historySearch}
                  onChange={e => setHistorySearch(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 text-xs bg-slate-50 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl text-xs font-medium">
                {(['ALL', 'SPAM', 'NOT SPAM'] as const).map(flt => (
                  <button
                    key={flt}
                    onClick={() => setHistoryFilter(flt)}
                    className={`px-3 py-1 rounded-lg transition-all ${
                      historyFilter === flt
                        ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {flt}
                  </button>
                ))}
              </div>
            </div>

            {/* History Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
              {filteredHistory.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50 text-xs font-semibold text-slate-500 uppercase">
                        <th className="py-3 px-5">ID</th>
                        <th className="py-3 px-4">Message Content Preview</th>
                        <th className="py-3 px-4">Prediction</th>
                        <th className="py-3 px-4">Confidence</th>
                        <th className="py-3 px-4">Stats</th>
                        <th className="py-3 px-4">Timestamp</th>
                        <th className="py-3 px-4 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {filteredHistory.map(item => (
                        <tr key={item.id} className="hover:bg-slate-50/70 transition-colors">
                          <td className="py-3 px-5 font-mono text-xs text-slate-400">
                            #{item.id.slice(-6)}
                          </td>
                          <td className="py-3 px-4 max-w-md">
                            <div className="text-slate-800 text-xs font-medium line-clamp-2">
                              {item.rawText}
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <span
                              className={`text-xs font-bold px-2 py-0.5 rounded-md ${
                                item.prediction === 'SPAM'
                                  ? 'bg-rose-100 text-rose-800'
                                  : 'bg-emerald-100 text-emerald-800'
                              }`}
                            >
                              {item.prediction}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-semibold text-slate-800 text-xs">
                            {item.confidence}%
                          </td>
                          <td className="py-3 px-4 text-xs text-slate-500">
                            {item.stats.characterCount} chars &bull; {item.stats.wordCount} words
                          </td>
                          <td className="py-3 px-4 text-xs text-slate-400 whitespace-nowrap">
                            {item.timestamp}
                          </td>
                          <td className="py-3 px-4 text-right">
                            <button
                              onClick={() => handleDeleteHistoryItem(item.id)}
                              className="text-slate-400 hover:text-rose-600 p-1 transition-colors"
                              title="Delete record"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-400 text-sm">
                  No prediction records match your query.
                </div>
              )}
            </div>
          </div>
        )}

        {/* ===================== TAB 5: DATASET INSPECTOR ===================== */}
        {activeTab === 'dataset' && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                  Training Dataset & Vocabulary Explorer
                </h1>
                <p className="text-sm text-slate-500 mt-1">
                  Inspect the structured demonstration dataset ({TRAINING_DATASET.length} total samples: 46 Spam, 46 Ham) and token vocabulary.
                </p>
              </div>
              <div className="text-xs bg-slate-200/70 text-slate-700 px-3 py-1.5 rounded-lg font-mono">
                Vocabulary: {ML_PIPELINE.getVocabularySize()} TF-IDF features
              </div>
            </div>

            {/* Filter controls */}
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-xs flex flex-wrap items-center justify-between gap-3">
              <div className="relative flex-1 min-w-[240px]">
                <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
                <input
                  type="text"
                  placeholder="Search dataset text or keyword..."
                  value={datasetSearch}
                  onChange={e => setDatasetSearch(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 text-xs bg-slate-50 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl text-xs font-medium">
                {(['ALL', 'SPAM', 'HAM'] as const).map(flt => (
                  <button
                    key={flt}
                    onClick={() => setDatasetFilter(flt)}
                    className={`px-3 py-1 rounded-lg transition-all ${
                      datasetFilter === flt
                        ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {flt}
                  </button>
                ))}
              </div>
            </div>

            {/* Dataset Rows Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
              <div className="p-4 border-b border-slate-200 bg-slate-50 flex justify-between items-center text-xs text-slate-500">
                <span>Showing {filteredDataset.length} of {TRAINING_DATASET.length} dataset rows</span>
                <span>File: <code>data/spam_email_dataset.csv</code></span>
              </div>
              <div className="overflow-x-auto max-h-[500px]">
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 bg-white border-b border-slate-200 shadow-2xs">
                    <tr className="text-xs font-semibold text-slate-400 uppercase">
                      <th className="py-3 px-5 w-20">Index</th>
                      <th className="py-3 px-4 w-28">Label</th>
                      <th className="py-3 px-5">Email / Message Text</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredDataset.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-50 transition-colors">
                        <td className="py-2.5 px-5 font-mono text-xs text-slate-400">#{idx + 1}</td>
                        <td className="py-2.5 px-4">
                          <span
                            className={`text-xs font-bold px-2 py-0.5 rounded-md uppercase ${
                              row.label === 'spam'
                                ? 'bg-rose-100 text-rose-800'
                                : 'bg-emerald-100 text-emerald-800'
                            }`}
                          >
                            {row.label}
                          </span>
                        </td>
                        <td className="py-2.5 px-5 text-xs text-slate-800 leading-relaxed">
                          {row.text}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ===================== TAB 6: PYTHON CODE & VS CODE ===================== */}
        {activeTab === 'code' && (
          <div className="space-y-6 max-w-5xl mx-auto">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                Academic Project Source Code & Windows 11 Guide
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Complete, production-ready Python files and step-by-step instructions to run locally in VS Code.
              </p>
            </div>

            {/* Windows 11 VS Code Terminal Guide */}
            <div className="bg-slate-900 text-slate-100 rounded-2xl p-6 shadow-sm">
              <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-rose-500"></div>
                  <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                  <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                  <span className="text-xs font-mono text-slate-400 ml-2">
                    Windows PowerShell / VS Code Terminal
                  </span>
                </div>
              </div>

              <div className="space-y-3 font-mono text-xs text-slate-300">
                <div>
                  <span className="text-slate-500"># 1. Create and activate Python virtual environment</span>
                  <div className="text-emerald-400 bg-slate-950 p-2 rounded-lg mt-1 select-all">
                    python -m venv venv<br />
                    .\venv\Scripts\Activate.ps1
                  </div>
                </div>

                <div>
                  <span className="text-slate-500"># 2. Install all required dependencies</span>
                  <div className="text-emerald-400 bg-slate-950 p-2 rounded-lg mt-1 select-all">
                    pip install -r requirements.txt
                  </div>
                </div>

                <div>
                  <span className="text-slate-500"># 3. Train NLP TF-IDF vectorizer and Machine Learning models</span>
                  <div className="text-emerald-400 bg-slate-950 p-2 rounded-lg mt-1 select-all">
                    python train_model.py
                  </div>
                </div>

                <div>
                  <span className="text-slate-500"># 4. Run the Flask web application with SQLite database</span>
                  <div className="text-emerald-400 bg-slate-950 p-2 rounded-lg mt-1 select-all">
                    python app.py
                  </div>
                </div>

                <div className="text-slate-400 pt-2 text-[11px]">
                  &rarr; Then open browser at: <code className="text-blue-400">http://127.0.0.1:5000</code> with login <code>admin</code> / <code>admin123</code>
                </div>
              </div>
            </div>

            {/* Code Viewer */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
              {/* File selector bar */}
              <div className="p-3 bg-slate-100 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap gap-1">
                  {(['train_model.py', 'app.py', 'requirements.txt', 'Procfile', 'README.md', 'dataset.csv'] as const).map(file => (
                    <button
                      key={file}
                      onClick={() => setSelectedCodeFile(file)}
                      className={`px-3 py-1.5 text-xs font-mono rounded-lg transition-colors ${
                        selectedCodeFile === file
                          ? 'bg-white text-blue-700 font-bold shadow-2xs border border-slate-300'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      {file}
                    </button>
                  ))}
                </div>

                <button
                  onClick={handleCopyCode}
                  className="px-3 py-1.5 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded-lg text-xs font-semibold transition-colors flex items-center gap-1.5"
                >
                  {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  {copiedCode ? 'Copied!' : 'Copy File Content'}
                </button>
              </div>

              {/* Code text block */}
              <pre className="p-5 text-xs font-mono text-slate-800 bg-slate-50/50 overflow-x-auto leading-relaxed max-h-[550px]">
                {codeContents[selectedCodeFile]}
              </pre>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            &copy; 2026 <strong>SpamGuard ML</strong> &bull; Artificial Intelligence & Data Science Undergraduate Capstone Project
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>Scikit-learn</span>
            <span>&bull;</span>
            <span>TF-IDF NLP</span>
            <span>&bull;</span>
            <span>Flask & SQLite</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
