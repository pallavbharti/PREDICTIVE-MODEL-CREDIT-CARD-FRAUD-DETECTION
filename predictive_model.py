"""
=============================================================================
PREDICTIVE MODEL — CREDIT CARD FRAUD DETECTION
=============================================================================
Project: Probability Model of Financial Loss in Online Scam Victims
Authors: Ishika Mehrotra | Shivanya Bharadwaj | Pallav Bharti | Manish Kumar
UPES, School of Computer Science, B.TECH VI Semester

IS MODEL SE TEEN CHEEZEIN PREDICT HOTI HAIN:
  1. Kya transaction FRAUD hai ya NORMAL? (Binary Classification)
  2. Kitne % probability hai ki yeh FRAUD hai? (Probability Score)
  3. Agar fraud hai to kitna FINANCIAL LOSS hoga? (Regression)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

def print_section(title):
    print("\n" + "="*65)
    print(f"  {title}")
    print("="*65)

def print_subsection(title):
    print(f"\n  ── {title} ──")

# ─────────────────────────────────────────────
# STEP 1: DATA LOAD
# ─────────────────────────────────────────────
print_section("STEP 1: LOADING DATASET")

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
try:
    data = pd.read_csv(URL)
    print(f"  Source  : Kaggle Credit Card Fraud Dataset (via URL)")
except Exception:
    data = pd.read_csv("creditcard.csv")
    print(f"  Source  : Local file (creditcard.csv)")

fraud  = data[data['Class'] == 1].copy()
normal = data[data['Class'] == 0].copy()

print(f"  Rows    : {len(data):,}")
print(f"  Columns : {data.shape[1]}")
print(f"  Normal  : {len(normal):,}")
print(f"  Fraud   : {len(fraud):,}")

# ─────────────────────────────────────────────
# STEP 2: FEATURE PREPARATION
# ─────────────────────────────────────────────
print_section("STEP 2: FEATURE PREPARATION")

features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
X = data[features].copy()
y = data['Class'].copy()

scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Features used : {len(features)} (V1–V28 + Amount + Time)")
print(f"  Train set     : {len(X_train):,} transactions")
print(f"  Test set      : {len(X_test):,} transactions")
print(f"  Fraud in test : {y_test.sum()} transactions")

# ─────────────────────────────────────────────
# STEP 3: TRAIN MODELS
# ─────────────────────────────────────────────
print_section("STEP 3: TRAINING MODELS")

# Model A: Naive Bayes (probabilistic)
print_subsection("Training Naive Bayes (Gaussian)")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"  Status  : Trained successfully")
print(f"  P(Fraud) learned by model  : {gnb.class_prior_[1]:.6f}")
print(f"  P(Normal) learned by model : {gnb.class_prior_[0]:.6f}")

# Model B: Logistic Regression (best overall)
print_subsection("Training Logistic Regression (Balanced)")
lr = LogisticRegression(class_weight='balanced', max_iter=1000,
                        C=1.0, random_state=42)
lr.fit(X_train, y_train)
print(f"  Status  : Trained successfully")
print(f"  Intercept (b0): {lr.intercept_[0]:.4f}")

# ─────────────────────────────────────────────
# PREDICTION 1: IS THIS TRANSACTION FRAUD?
# (Binary Classification — Class 0 or 1)
# ─────────────────────────────────────────────
print_section("PREDICTION 1: FRAUD HAI YA NORMAL? (Binary Classification)")

y_pred_nb = gnb.predict(X_test)
y_pred_lr = lr.predict(X_test)

print_subsection("Naive Bayes Results")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_nb)*100:.2f}%")
print(f"  F1 Score  : {f1_score(y_test, y_pred_nb):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, gnb.predict_proba(X_test)[:,1]):.4f}")
print(f"\n  Classification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb,
      target_names=['Normal (0)', 'Fraud (1)']))

print_subsection("Logistic Regression Results")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_lr)*100:.2f}%")
print(f"  F1 Score  : {f1_score(y_test, y_pred_lr):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]):.4f}")
print(f"\n  Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr,
      target_names=['Normal (0)', 'Fraud (1)']))

# Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Prediction 1: Fraud vs Normal Classification\n(Confusion Matrices)',
             fontsize=14, fontweight='bold')

for ax, preds, title in zip(axes,
    [y_pred_nb, y_pred_lr],
    ['Naive Bayes', 'Logistic Regression']):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(
        f'Predicted Label\n'
        f'True Positives (Fraud correctly caught): {tp}\n'
        f'False Negatives (Fraud missed): {fn}')

plt.tight_layout()
plt.savefig('pred1_classification.png', dpi=150, bbox_inches='tight')
plt.show()
print("  [CHART SAVED] pred1_classification.png")

# ─────────────────────────────────────────────
# PREDICTION 2: KITNA % FRAUD PROBABILITY HAI?
# (Probability Score 0.0 to 1.0)
# ─────────────────────────────────────────────
print_section("PREDICTION 2: FRAUD PROBABILITY SCORE (0.0 – 1.0)")

y_prob_nb = gnb.predict_proba(X_test)[:, 1]
y_prob_lr = lr.predict_proba(X_test)[:, 1]

print_subsection("Sample Probability Predictions (first 10 test transactions)")
sample_df = pd.DataFrame({
    'Actual Class'          : y_test.values[:10],
    'NB Fraud Prob (%)'     : (y_prob_nb[:10] * 100).round(4),
    'LR Fraud Prob (%)'     : (y_prob_lr[:10] * 100).round(4),
    'NB Prediction'         : ['FRAUD' if p == 1 else 'Normal' for p in y_pred_nb[:10]],
    'LR Prediction'         : ['FRAUD' if p == 1 else 'Normal' for p in y_pred_lr[:10]],
})
print(sample_df.to_string(index=False))

# Show top 5 highest fraud probability transactions
print_subsection("Top 5 Transactions with HIGHEST Fraud Probability (LR Model)")
test_with_prob = X_test.copy()
test_with_prob['Actual']       = y_test.values
test_with_prob['Fraud_Prob_%'] = (y_prob_lr * 100).round(4)
test_with_prob['Prediction']   = ['FRAUD' if p == 1 else 'Normal' for p in y_pred_lr]
top5 = test_with_prob.nlargest(5, 'Fraud_Prob_%')[
    ['Actual', 'Fraud_Prob_%', 'Prediction']]
top5['Actual'] = top5['Actual'].map({0: 'Normal', 1: 'FRAUD'})
print(top5.to_string())

# ROC Curve + Probability Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Prediction 2: Fraud Probability Score Analysis',
             fontsize=14, fontweight='bold')

# ROC Curves
for prob, name, color in [
    (y_prob_nb, 'Naive Bayes',         '#E74C3C'),
    (y_prob_lr, 'Logistic Regression', '#2ECC71'),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    axes[0].plot(fpr, tpr, lw=2, color=color,
                 label=f'{name} (AUC={auc:.4f})')
axes[0].plot([0,1], [0,1], 'k--', lw=1, label='Random Classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve (Higher AUC = Better Model)')
axes[0].legend()

# Probability distribution by class (LR model)
fraud_probs  = y_prob_lr[y_test == 1]
normal_probs = y_prob_lr[y_test == 0]
axes[1].hist(normal_probs, bins=50, alpha=0.6, color='#2ECC71',
             label='Normal Transactions', density=True)
axes[1].hist(fraud_probs,  bins=30, alpha=0.8, color='#E74C3C',
             label='Fraud Transactions', density=True)
axes[1].axvline(x=0.5, color='black', linestyle='--',
                linewidth=2, label='Decision threshold (0.5)')
axes[1].set_xlabel('Predicted Fraud Probability')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Fraud Probability Scores\n(Logistic Regression)')
axes[1].legend()

plt.tight_layout()
plt.savefig('pred2_probability_score.png', dpi=150, bbox_inches='tight')
plt.show()
print("  [CHART SAVED] pred2_probability_score.png")

# ─────────────────────────────────────────────
# PREDICTION 3: EXPECTED FINANCIAL LOSS
# (How much money lost if fraud happens)
# ─────────────────────────────────────────────
print_section("PREDICTION 3: EXPECTED FINANCIAL LOSS ($)")

# Statistical parameters from dataset
mu_fraud    = fraud['Amount'].mean()
sigma_fraud = fraud['Amount'].std()
P_fraud     = len(fraud) / len(data)

print_subsection("Financial Loss Parameters (from training data)")
print(f"  Mean loss if fraud    : ${mu_fraud:.2f}")
print(f"  Std dev of fraud loss : ${sigma_fraud:.2f}")
print(f"  P(Fraud) base rate    : {P_fraud:.6f} ({P_fraud*100:.4f}%)")

# For each test transaction, compute expected financial loss
# E[Loss] = P(Fraud|x) * E[Amount|Fraud]
# Using predicted fraud probability from Logistic Regression
expected_loss = y_prob_lr * mu_fraud

print_subsection("Expected Financial Loss — Sample Predictions (10 transactions)")
loss_df = pd.DataFrame({
    'Actual'           : y_test.values[:10],
    'Fraud Prob (%)'   : (y_prob_lr[:10] * 100).round(3),
    'Expected Loss ($)': expected_loss[:10].round(2),
    'Actual Amount ($)': data.loc[X_test.index[:10], 'Amount'].values.round(2),
    'Risk Level'       : ['HIGH' if p > 0.5 else 'LOW'
                          for p in y_prob_lr[:10]],
})
loss_df['Actual'] = loss_df['Actual'].map({0: 'Normal', 1: 'FRAUD'})
print(loss_df.to_string(index=False))

# Loss range categorization for fraud transactions only
print_subsection("Probability Distribution of Financial Loss Ranges")
bins   = [0, 100, 500, 2000, np.inf]
labels = ['$0–$100 (Low)', '$100–$500 (Medium)',
          '$500–$2000 (High)', '$2000+ (Extreme)']
fraud_copy = fraud.copy()
fraud_copy['loss_range'] = pd.cut(fraud_copy['Amount'],
                                  bins=bins, labels=labels)
prob_model = fraud_copy['loss_range'].value_counts(
    normalize=True).sort_index()

print(f"  {'Loss Range':<25} {'Probability':>12} {'% of Frauds':>12}")
print(f"  {'-'*50}")
for rng, p in prob_model.items():
    bar = '█' * int(p * 30)
    print(f"  {str(rng):<25} {p:>12.4f} {p*100:>11.2f}%  {bar}")

print(f"\n  E[Financial Loss | Fraud]     = ${mu_fraud:.2f}")
print(f"  E[Financial Loss] overall     = ${P_fraud * mu_fraud:.4f}")
print(f"  Max recorded fraud loss       = ${fraud['Amount'].max():.2f}")

# Financial loss charts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Prediction 3: Expected Financial Loss Analysis',
             fontsize=14, fontweight='bold')

# Chart A: Loss range bar chart
colors_bar = ['#27AE60', '#F39C12', '#E74C3C', '#8E44AD']
prob_model.plot(kind='bar', ax=axes[0], color=colors_bar,
                edgecolor='black', alpha=0.85)
axes[0].set_title('Probability of Loss Range\n(Fraud Transactions)')
axes[0].set_xlabel('Loss Range')
axes[0].set_ylabel('Probability')
axes[0].tick_params(axis='x', rotation=25)
for i, v in enumerate(prob_model.values):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center',
                 fontweight='bold', fontsize=10)

# Chart B: Distribution of fraud amounts (histogram + normal fit)
x_range = np.linspace(0, 1500, 500)
normal_fit = stats.norm.pdf(x_range, mu_fraud, sigma_fraud)
axes[1].hist(fraud['Amount'], bins=50, density=True,
             color='#E74C3C', alpha=0.7, label='Actual fraud amounts')
axes[1].plot(x_range, normal_fit, 'b-', lw=2,
             label=f'Normal fit μ=${mu_fraud:.1f}')
axes[1].axvline(mu_fraud, color='darkred', linestyle='--',
                lw=2, label=f'Mean = ${mu_fraud:.1f}')
axes[1].set_title('Fraud Amount Distribution\n(with Normal Fit)')
axes[1].set_xlabel('Transaction Amount ($)')
axes[1].set_ylabel('Density')
axes[1].legend(fontsize=8)
axes[1].set_xlim(0, 1500)

# Chart C: Expected loss scatter (fraud prob vs expected loss)
sample_idx = np.random.choice(len(y_prob_lr), 1000, replace=False)
colors_sc  = ['#E74C3C' if y == 1 else '#2ECC71'
               for y in y_test.values[sample_idx]]
axes[2].scatter(y_prob_lr[sample_idx],
                expected_loss[sample_idx],
                c=colors_sc, alpha=0.5, s=15)
axes[2].axvline(0.5, color='black', linestyle='--',
                lw=1.5, label='Decision threshold')
axes[2].set_xlabel('Predicted Fraud Probability')
axes[2].set_ylabel('Expected Financial Loss ($)')
axes[2].set_title('Fraud Probability vs\nExpected Financial Loss')
fraud_patch  = mpatches.Patch(color='#E74C3C', label='Fraud')
normal_patch = mpatches.Patch(color='#2ECC71', label='Normal')
axes[2].legend(handles=[fraud_patch, normal_patch])

plt.tight_layout()
plt.savefig('pred3_financial_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("  [CHART SAVED] pred3_financial_loss.png")

# ─────────────────────────────────────────────
# LIVE PREDICTION DEMO
# (Simulate predicting on a NEW transaction)
# ─────────────────────────────────────────────
print_section("LIVE DEMO: NEW TRANSACTION PREDICT KARO")

# Take one real fraud and one real normal transaction from test set
fraud_test_idx  = y_test[y_test == 1].index[0]
normal_test_idx = y_test[y_test == 0].index[0]

for label, idx in [("FRAUD Transaction",  fraud_test_idx),
                   ("NORMAL Transaction", normal_test_idx)]:
    txn      = X_test.loc[idx].values.reshape(1, -1)
    actual   = y_test[idx]
    orig_amt = data.loc[idx, 'Amount']

    nb_pred  = gnb.predict(txn)[0]
    nb_prob  = gnb.predict_proba(txn)[0][1]
    lr_pred  = lr.predict(txn)[0]
    lr_prob  = lr.predict_proba(txn)[0][1]
    exp_loss = lr_prob * mu_fraud

    print(f"\n  Transaction Type  : {label}")
    print(f"  Actual Amount     : ${orig_amt:.2f}")
    print(f"  Actual Class      : {'FRAUD (1)' if actual == 1 else 'Normal (0)'}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Naive Bayes:")
    print(f"    Fraud Probability : {nb_prob*100:.4f}%")
    print(f"    Prediction        : {'>>> FRAUD <<<' if nb_pred == 1 else 'Normal'}")
    print(f"  Logistic Regression:")
    print(f"    Fraud Probability : {lr_prob*100:.4f}%")
    print(f"    Prediction        : {'>>> FRAUD <<<' if lr_pred == 1 else 'Normal'}")
    print(f"  Expected Financial Loss : ${exp_loss:.2f}")
    risk = 'HIGH RISK' if lr_prob > 0.5 else 'LOW RISK'
    print(f"  Risk Level        : {risk}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print_section("FINAL SUMMARY — TEEN PREDICTIONS KA RESULT")

print("""
  IS PROJECT SE TEEN CHEEZEIN PREDICT HOTI HAIN:

  PREDICTION 1 — FRAUD HAI YA NAHI? (Binary Class)
  ──────────────────────────────────────────────
  Input  : Transaction features (V1–V28, Amount, Time)
  Output : 0 (Normal) ya 1 (Fraud)
  Model  : Naive Bayes + Logistic Regression
  Best   : Logistic Regression (ROC-AUC ~ 0.97)

  PREDICTION 2 — KITNA % FRAUD PROBABILITY HAI?
  ──────────────────────────────────────────────
  Input  : Same features
  Output : 0.0 se 1.0 tak score (e.g., 0.87 = 87% fraud)
  Model  : predict_proba() function
  Use    : Real-time risk scoring, alert systems

  PREDICTION 3 — EXPECTED FINANCIAL LOSS?
  ──────────────────────────────────────────────
  Input  : Fraud Probability score
  Output : Expected Loss in $ = P(Fraud|x) × Mean(Fraud Amount)
  Model  : Probability model + Statistical formula
  Use    : Insurance claims, bank risk assessment

  Charts generated:
  - pred1_classification.png  → Confusion matrices
  - pred2_probability_score.png → ROC curve + score distribution
  - pred3_financial_loss.png  → Loss analysis
""")

print("="*65)
print("  AUTHORS: Ishika Mehrotra | Shivanya Bharadwaj | Pallav Bharti | Manish Kumar")
print("  UPES, School of Computer Science, B.TECH VI Semester")
print("="*65)
