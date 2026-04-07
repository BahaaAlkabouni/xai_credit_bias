# Bias and Fairness in Credit Scoring Models
**XAI Final Assignment — Explainable AI (SOW-BKI266)**  
Author: Bahaa Alkabouni (S1117232) — Radboud University, 2026


## Overview

This project investigates age-related bias in a machine learning credit scoring model using two XAI methods:

| Method | Type | Coverage |
|--------|------|----------|
| **LIME** (Local Interpretable Model-agnostic Explanations) | Post-hoc, local | Covered in lectures |
| **DiCE** (Diverse Counterfactual Explanations) | Post-hoc, local | Novel (not in lectures) |

A **Random Forest** classifier is trained on the [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) to predict credit risk. Both XAI methods are then used to audit the model for age bias, and their stability is tested under perturbation.

---

## Research Question

> *How effectively do LIME and counterfactual explanations expose age bias in credit scoring models, and how stable are these explanations when minor changes are applied to the input data?*

---

## Repository Structure

```
xai_credit_bias/
├── main.py            
├── requirements.txt   
├── README.md          
└── figures/           
    ├── feature_importance.png
    ├── lime_explanation.png
    ├── lime_stability.png
    ├── lime_age_groups.png
    ├── dice_counterfactuals.png
    └── dice_age_perturbation.png
```

---

## Installation

**Python 3.9+ recommended.**

```bash
# 1. Clone the repository
git clone https://github.com/<BahaaAlkabouni>/xai_credit_bias.git
cd xai_credit_bias

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

The script will:
1. Download the German Credit dataset automatically from UCI
2. Train a Random Forest classifier
3. Generate and save all figures to `./figures/`
4. Print evaluation metrics and key findings to the console

No manual data download is required.

---

## Generated Figures

| Figure | Description |
|--------|-------------|
| `feature_importance.png` | Global RF feature importance (age highlighted in red) |
| `lime_explanation.png` | LIME explanation for a single test instance |
| `lime_stability.png` | LIME stability: variance of weights across 20 repeated runs |
| `lime_age_groups.png` | Comparison of LIME age weight: young vs older applicants |
| `dice_counterfactuals.png` | DiCE counterfactuals: what changes flip Bad→Good Credit |
| `dice_age_perturbation.png` | Age perturbation test: at what age does prediction flip? |

---

## Dataset

**Statlog (German Credit Data)** — UCI Machine Learning Repository  
Hofmann, H. (1994). 1000 instances, 20 features, binary classification (Good/Bad credit risk).  
Downloaded automatically via: `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data`

---

## Dependencies

```
numpy, pandas, scikit-learn, matplotlib, lime, dice-ml
```

See `requirements.txt` for version details.

---

## References

- Ribeiro et al. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier.* KDD.
- Mothilal et al. (2020). *Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations.* FAccT.
- Mehrabi et al. (2021). *A Survey on Bias and Fairness in Machine Learning.* ACM CSUR.
- Hofmann, H. (1994). Statlog German Credit Data. UCI ML Repository.
