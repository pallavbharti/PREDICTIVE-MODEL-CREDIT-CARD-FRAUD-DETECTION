# Predictive Model — Credit Card Fraud Detection

**Subject:** Probability & Statistics — Final Project  
**Institution:** UPES, School of Computer Science  
**Program:** B.Tech CSE, VI Semester  
**Authors:** Ishika Mehrotra | Shivanya Bharadwaj | Pallav Bharti

---

## What Does This Project Do?

This project builds a predictive system that answers three questions about any given credit card transaction:

| # | Question | Type | Output |
|---|----------|------|--------|
| 1 | Is this transaction fraudulent or normal? | Binary Classification | `0` (Normal) or `1` (Fraud) |
| 2 | What is the probability that it is fraud? | Probability Scoring | A score between `0.0` and `1.0` |
| 3 | If it turns out to be fraud, how much money could be lost? | Financial Loss Estimation | Expected loss in dollars ($) |

We use two machine learning models — **Gaussian Naive Bayes** and **Logistic Regression** — and compare their performance side by side.

---

## Why Credit Card Fraud Detection?

Every year, credit card fraud causes billions of dollars in losses globally. The tricky part is that fraudulent transactions are extremely rare — in our dataset, only **0.17%** of transactions are fraud. That makes this a classic imbalanced classification problem, where a naive model could just predict "not fraud" every time and still be 99.8% accurate, while catching zero actual fraud. Our project tackles this head-on using balanced class weighting and probability-based evaluation metrics.

---

## Dataset

We use the well-known **Credit Card Fraud Detection** dataset originally from Kaggle, containing real transactions made by European cardholders in September 2013.

- **Total transactions:** 284,807
- **Fraudulent transactions:** 492 (just 0.172%)
- **Features:** 30 input features
  - `V1` to `V28` — these are PCA-transformed features (original feature names are hidden for privacy)
  - `Amount` — the transaction amount in dollars
  - `Time` — seconds elapsed since the first transaction in the dataset
- **Target column:** `Class` — where `0` = Normal and `1` = Fraud

The script automatically downloads the dataset from a public URL. If the download fails, it falls back to a local `creditcard.csv` file.

---

## How the Code Works — Step by Step

### Step 1: Data Loading

The script loads the CSV dataset, separates it into fraud and normal subsets, and prints a quick summary (row count, column count, class distribution).

### Step 2: Feature Preparation

All 30 features (`V1`–`V28`, `Amount`, `Time`) are used. Since the PCA features are already scaled, only `Amount` and `Time` are standardized using `StandardScaler` to bring them to a comparable range. The data is then split into **80% training** and **20% testing** sets using stratified sampling (so both sets maintain the same fraud ratio).

### Step 3: Model Training

Two classifiers are trained on the training data:

1. **Gaussian Naive Bayes (GNB)** — A probabilistic classifier that assumes features follow a normal distribution. It naturally outputs probability estimates, which makes it a great fit for fraud scoring.

2. **Logistic Regression (LR)** — A linear model with `class_weight='balanced'`, which internally adjusts for the fact that fraud cases are rare. This prevents the model from ignoring the minority class.

### Step 4: Three Predictions

This is the core of the project. Each prediction builds on the previous one:

**Prediction 1 — Fraud or Normal (Binary Classification)**  
Both models classify each test transaction as fraud (`1`) or normal (`0`). We evaluate them using Accuracy, F1-Score, ROC-AUC, and a full Classification Report. Confusion matrices are plotted side by side to visually compare how many frauds each model catches versus how many it misses.

**Prediction 2 — Fraud Probability Score**  
Instead of a hard yes/no answer, the models output a probability score between 0.0 and 1.0 for each transaction. For example, a score of `0.87` means the model thinks there is an 87% chance the transaction is fraudulent. We visualize this with ROC curves (comparing both models) and a histogram showing how fraud and normal transactions are distributed across probability scores.

**Prediction 3 — Expected Financial Loss**  
This is where statistics comes in. For each transaction, we estimate the expected financial loss using the formula:

```
Expected Loss = P(Fraud | transaction) × Mean Fraud Amount
```

The fraud probability comes from the Logistic Regression model, and the mean fraud amount is calculated from the training data. We also categorize historical fraud transactions into loss ranges ($0–$100, $100–$500, $500–$2000, $2000+) and compute the probability of each range. Three charts are generated: a loss range bar chart, a fraud amount distribution with normal curve fit, and a scatter plot of fraud probability vs. expected loss.

### Step 5: Live Demo

The script picks one real fraud transaction and one real normal transaction from the test set and runs both models on them in real-time. It prints the fraud probability, the predicted label, the expected financial loss, and a risk level tag (HIGH or LOW) for each.

---

## Output & Visualizations

The script generates three chart files, each saved at 150 DPI:

| File | What It Shows |
|------|---------------|
| `pred1_classification.png` | Side-by-side confusion matrices for Naive Bayes and Logistic Regression |
| `pred2_probability_score.png` | ROC curves for both models + fraud probability distribution histogram |
| `pred3_financial_loss.png` | Loss range probabilities + fraud amount distribution + fraud probability vs. expected loss scatter |

All results and metrics are also printed to the console in a structured, readable format.

---

## Libraries Used

| Library | Version | Role in This Project |
|---------|---------|----------------------|
| `pandas` | — | Loading and manipulating the dataset |
| `numpy` | — | Numerical operations, array handling |
| `matplotlib` | — | Creating all charts and visualizations |
| `seaborn` | — | Heatmaps for confusion matrices |
| `scipy` | — | Normal distribution fitting for loss analysis |
| `scikit-learn` | — | Model training, evaluation metrics, preprocessing |

---

## How to Run

**Prerequisites:** Python 3.8 or above and an active internet connection (for dataset download).

1. Install the required packages:
   ```
   pip install pandas numpy matplotlib seaborn scipy scikit-learn
   ```

2. Run the script:
   ```
   python predictive_model.py
   ```

3. The script will automatically download the dataset, train both models, run all three predictions, generate charts, and print a complete summary to the terminal.

> If the download URL does not work, download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place the `creditcard.csv` file in the same folder as the script.

---

## Project Structure

```
Stats/
│
├── predictive_model.py          # Complete source code (single-file project)
├── creditcard.csv               # Dataset (auto-downloaded or placed manually)
├── README.md                    # Project documentation (this file)
│
├── pred1_classification.png     # Generated — Confusion matrix comparison
├── pred2_probability_score.png  # Generated — ROC curves & probability distributions
└── pred3_financial_loss.png     # Generated — Financial loss analysis charts
```

---

## Key Results (Summary)

- **Logistic Regression** outperforms Naive Bayes overall, achieving an ROC-AUC of approximately **0.97**, meaning it can distinguish fraud from normal transactions with high reliability.
- The probability scoring system allows real-world applications like setting custom thresholds — a bank could flag any transaction with a fraud probability above 30% for manual review, rather than using the default 50% cutoff.
- The expected financial loss model gives a dollar-value estimate for each transaction, which is useful for insurance and risk assessment contexts.

---

## Limitations

- The dataset is from 2013, so the fraud patterns may not reflect current tactics.
- We only use two models. Ensemble methods like Random Forest or XGBoost could potentially improve recall.
- The financial loss prediction assumes a constant mean fraud amount, which is a simplification.
- The PCA-transformed features (V1–V28) are not interpretable, so we cannot explain which specific transaction attributes contribute most to a fraud prediction.

---

## References

- Dataset: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Scikit-learn documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- Original dataset paper: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.

---

*UPES, School of Computer Science — B.Tech VI Semester, 2026*
