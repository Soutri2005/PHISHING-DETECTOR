# 🛡️ Phishing Message Detector — Overfitting Study

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest Passed](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](tests/)

An end-to-end Machine Learning cybersecurity application and empirical research laboratory evaluating text classification, decision boundaries, model complexity, and regularization techniques on real-world phishing email data.

---

## 🎯 Project Overview & Research Question

In natural language processing and spam/phishing classification, models often face extreme dimensional sparsity (thousands of TF-IDF word features). A central engineering challenge is preventing models from **memorizing noise** (specific employee names, idiosyncratic URLs, temporary timestamps) while maintaining robust generalization on unseen zero-day attacks.

### 🔬 Primary Research Question:
> *"How does model complexity affect phishing message detection, and which techniques help reduce overfitting while maintaining good performance on unseen messages?"*

### 🔑 Key Engineering Highlights:
- **Zero-Leakage Architecture:** Strict feature fit isolation. The `TfidfVectorizer` vocabulary and IDF weights are computed **strictly on training partitions**.
- **Real Corpus:** Trained on 17,522 clean, deduplicated emails from the Hugging Face Phishing Email corpus.
- **4 Real Classifiers:** Multinomial Naive Bayes, Logistic Regression ($L_1 / L_2$), Decision Trees, and Random Forests.
- **Empirical Overfitting Lab:** Dynamic experiments demonstrating tree capacity sweeps, regularization curves ($C$ parameter), and sample-size learning curves.
- **Explainability & Heuristics:** Keyword log-odds decomposition paired with cybersecurity heuristics (IP detection, urgency matching, credential harvesting flags).

---

## 🏗️ Technology Stack

- **Language:** Python 3
- **User Interface:** Streamlit (Cybersecurity Dark Navy & Teal UI)
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Pipelines, TF-IDF, Metrics, Tree inspection)
- **Serialization:** Joblib
- **Visualization:** Altair, Matplotlib, Seaborn
- **Automated Testing:** Pytest

---

## 📁 Repository Structure

```
phishing-overfitting-study/
├── app.py                          # Streamlit Entry Point (Page 1: Project Overview)
├── requirements.txt                # Package dependencies
├── README.md                       # Comprehensive documentation
├── .gitignore                      # Git exclusion rules
├── data/
│   ├── README.md                   # Data provenance & schema
│   ├── raw/                        # Downloaded raw dataset
│   └── processed/                  # Cleaned, stratified splits (train, val, test)
├── models/
│   ├── best_model.joblib           # Top generalizer (Logistic Regression)
│   ├── logistic_regression.joblib  # Fitted LR pipeline
│   ├── naive_bayes.joblib          # Fitted MNB pipeline
│   ├── decision_tree.joblib        # Fitted DT pipeline
│   └── random_forest.joblib        # Fitted RF pipeline
├── src/
│   ├── __init__.py
│   ├── download_dataset.py         # Automated dataset retriever
│   ├── data_loader.py              # Cleaning, deduplication, and stratified split
│   ├── preprocessing.py            # Text cleaning, regex heuristics, TF-IDF pipeline
│   ├── train.py                    # Multi-model training and evaluation harness
│   ├── evaluate.py                 # Classification & generalization gap metrics
│   ├── experiments.py              # Overfitting experiments (depth, C, learning curves)
│   └── predict.py                  # Inference pipeline with log-odds explainability
├── pages/
│   ├── 1_Phishing_Detector.py      # Real-time inference & heuristic analysis
│   ├── 2_Model_Comparison.py       # Benchmark table, charts, confusion matrices
│   ├── 3_Overfitting_Lab.py        # Interactive tree depth & regularization lab
│   ├── 4_Error_Analysis.py         # False positive/negative inspection & token weights
│   └── 5_Methodology.py            # Research protocols, leakage audit, findings
├── reports/
│   ├── data_preparation_report.json
│   ├── training_results.json
│   ├── dt_complexity_experiments.csv
│   ├── lr_regularization_experiments.csv
│   ├── learning_curve_experiments.csv
│   └── figures/
│       ├── dt_complexity_curve.png
│       ├── lr_regularization_curve.png
│       └── learning_curve.png
└── tests/
    ├── test_data_loader.py         # Data splitting & leakage tests
    ├── test_preprocessing.py       # Normalization & heuristics tests
    ├── test_training.py            # Metric calculation unit tests
    └── test_prediction.py          # Inference pipeline smoke tests
```

---

## 🚀 Quickstart & Reproduction Workflow

Automated end-to-end execution from raw data to working application:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and Prepare the Real Dataset
Downloads `Phishing_Email.csv` from Hugging Face, purges 1,109 exact duplicates, performs stratified 70/15/15 splitting, and verifies 0 cross-partition leakage:
```bash
python -m src.data_loader
```

### 3. Train All Four Baseline Models
Fits all pipelines, computes generalization gaps, evaluates the best model on the untouched test partition, and serializes joblib artifacts:
```bash
python -m src.train
```

### 4. Execute Overfitting Experiments
Runs Decision Tree depth sweeps, Logistic Regression $L_1 / L_2$ regularization sweeps, and empirical learning curves:
```bash
python -m src.experiments
```

### 5. Run Automated Tests
```bash
pytest tests/ -v
```

### 6. Launch the Web Application
```bash
streamlit run app.py
```

---

## 📊 Empirical Results & Model Benchmark

Trained on 12,264 training samples, evaluated on 2,629 validation samples and 2,629 held-out test samples (37.35% Phishing base rate):

| Model Architecture | Train Acc | Val Acc | Generalization Gap | Val Precision | Val Recall | Val F1-Score | Training Time | Overfitting Severity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Multinomial Naive Bayes** | 96.59% | 95.89% | **0.70%** | 98.42% | 90.87% | 94.50% | 22.5s | Negligible (Linear) |
| **Logistic Regression ($L_2, C=1.0$)** 🏆 | 98.61% | **97.79%** | **0.82%** | **98.24%** | **95.85%** | **97.03%** | 24.5s | **Optimal Balance** |
| **Decision Tree (Unconstrained)** | **99.99%** | 93.46% | **6.53%** | 90.96% | 91.78% | 91.37% | 27.6s | **Severe Overfitting** |
| **Random Forest (100 Trees)** | **99.99%** | 96.88% | **3.11%** | 97.46% | 94.35% | 95.88% | 108.8s | Moderate Gap |

### 🔒 Untouched Final Test Set Evaluation (Selected Model: Logistic Regression):
- **Test Accuracy:** **97.76%**
- **Test Precision:** **98.14%**
- **Test Recall (Phishing Catch Rate):** **95.83%**
- **Test F1-Score:** **96.97%**
- **Generalization Gap (Train vs Test):** **0.85%**

---

## 🧪 Overfitting Study Findings

### A. Decision Tree Depth Sweep:
| Max Depth Setting | Actual Depth | Total Leaves | Train Accuracy | Validation Accuracy | Generalization Gap | Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `max_depth = 2` | 2 | 4 | 62.66% | 62.65% | 0.01% | High Bias (Underfitting) |
| `max_depth = 5` | 5 | 17 | 74.54% | 72.65% | 1.89% | Developing Capacity |
| `max_depth = 10` | 10 | 79 | 85.82% | 83.42% | 2.40% | Moderate Capacity |
| `max_depth = 20` | 20 | 211 | 93.22% | 89.31% | 3.91% | Growing Variance |
| `max_depth = 30` | 30 | 268 | 96.04% | 91.67% | 4.37% | Noticeable Overfitting |
| `max_depth = None` | **261** | **564** | **99.99%** | **93.46%** | **6.53%** | **Severe Overfitting** |

**Scientific Takeaway:** In sparse text data, an unconstrained tree constructs 261 split levels and 564 leaves, effectively isolating individual idiosyncratic email fragments into dedicated leaves. Setting `min_samples_split >= 10` or capping depth at 15–20 prevents this divergence.

### B. Regularization Strength ($C$) & $L_1$ Sparsity:
- **Underfitting Region ($C \le 0.01$):** Excess penalty crushes feature weights, dropping accuracy to baseline majority class (~62.6%).
- **Optimal Region ($C = 1.0$):** Balances log-loss minimization with weight shrinkage, achieving 97.79% validation accuracy with minimal gap (0.82%).
- **$L_1$ Feature Selection (Lasso):** At $C=0.1$, $L_1$ drives non-zero coefficients down from 5,000 to just **73 active words** while maintaining 89.58% accuracy. At $C=1.0$, 420 active words achieve 96.88% accuracy.

### C. Empirical Learning Curves:
- At **5% training samples (613 emails)**: Generalization gap is **8.03%** (Train 98.86%, Val 90.83%).
- At **50% training samples (6,132 emails)**: Gap shrinks to **1.27%**.
- At **100% training samples (12,264 emails)**: Gap reaches asymptotic limit of **0.82%**, demonstrating that variance is fully controlled.

---

## 🔎 Application UI Pages

1. **Project Overview (`app.py`):** Executive summary, core telemetry, workflow pipeline diagram, and conceptual definitions.
2. **Live Phishing Detector (`pages/1_Phishing_Detector.py`):** Real-time text classifier with probability meters, quick-load test presets, heuristic risk indexes, defanged URL previews, and token log-odds explainability.
3. **Model Comparison (`pages/2_Model_Comparison.py`):** Side-by-side benchmark table, accuracy vs gap charts, confusion matrices, and test-set verification.
4. **Interactive Overfitting Lab (`pages/3_Overfitting_Lab.py`):** Real-time hyperparameters sandbox where students can tune tree depths and $C$ values to generate custom generalization curves on demand.
5. **Error Analysis (`pages/4_Error_Analysis.py`):** Inspection of false positives, false negatives, top phishing/legitimate n-grams, and cybersecurity failure modes.
6. **Methodology & Findings (`pages/5_Methodology.py`):** Complete research audit, zero-leakage verification, mathematical proofs, and limitation analysis.

---

## 🔒 Threat Modeling & Limitations

1. **Header Ignorance:** Body-only classification cannot inspect SMTP transmission hops, SPF/DKIM authentication, or DMARC records.
2. **Adversarial Obfuscation:** Attackers can inject zero-width unicode spaces or homoglyphs (`pаypаl` using Cyrillic `а`) to bypass TF-IDF tokenization.
3. **Image-Based Phishing (Quishing):** Text-only NLP receives zero signal when the phishing lure is enclosed within an image attachment or QR code.
4. **Future Extensions:** Implement subword BPE tokenization, DistilBERT embeddings, OCR image extraction, and WHOIS domain age telemetry.

---

## 📜 Dataset Citation & License

- **Corpus:** [Hugging Face `zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
- **License:** Open Research / Public Domain
- **Attribution:** Mirror of the Kaggle Phishing Email Detection corpus containing verified real-world benign and fraudulent communications.
