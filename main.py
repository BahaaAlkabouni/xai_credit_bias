import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import lime
import lime.lime_tabular

import dice_ml
from dice_ml import Dice

import os
os.makedirs("figures", exist_ok=True)


# STEP 1: LOAD DATA
print("Loading dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

col_names = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "credit_risk"
]

df = pd.read_csv(url, sep=" ", header=None, names=col_names)

# Recode target: 1 = Good credit -> 0, 2 = Bad credit -> 1
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

print(f"Dataset shape: {df.shape}")
print(f"Bad credit cases: {df['credit_risk'].sum()} / {len(df)}")


# STEP 2: PREPARE FEATURES

y = df["credit_risk"]

categorical_cols = [
    "checking_account", "credit_history", "purpose",
    "savings_account", "employment", "personal_status",
    "other_debtors", "property", "other_installments",
    "housing", "job", "telephone", "foreign_worker"
]

numeric_cols = [
    "duration", "credit_amount", "installment_rate",
    "age", "residence_since", "existing_credits", "num_dependents"
]

# One-hot encode categorical columns
X = pd.get_dummies(df[categorical_cols + numeric_cols],
                   columns=categorical_cols,
                   drop_first=True)

feature_names = list(X.columns)

# Find which column indices are categorical (needed by LIME)
cat_indices = [i for i, name in enumerate(feature_names)
               if any(name.startswith(c + "_") for c in categorical_cols)]


# STEP 3: TRAIN / TEST SPLIT AND MODEL TRAINING

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\nTraining Random Forest...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.values, y_train)

y_pred = model.predict(X_test.values)
print(classification_report(y_test, y_pred,
                             target_names=["Good Credit", "Bad Credit"]))


# STEP 4: GLOBAL FEATURE IMPORTANCE

print("Plotting feature importance...")

importances = model.feature_importances_
top_n = 12
top_indices = np.argsort(importances)[-top_n:]
top_names   = [feature_names[i] for i in top_indices]
top_values  = importances[top_indices]

# Colour the age bar red so it stands out
colors = ["#e74c3c" if "age" in name else "#3498db" for name in top_names]

plt.figure(figsize=(10, 6))
plt.barh(top_names, top_values, color=colors)
plt.xlabel("Feature Importance")
plt.title("Random Forest — Global Feature Importance\n(red = age feature)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()
print("  Saved: figures/feature_importance.png")


# STEP 5: LIME — EXPLAIN ONE PREDICTION
print("\nRunning LIME explanation...")

# Set up the LIME explainer using training data
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["Good Credit", "Bad Credit"],
    categorical_features=cat_indices,
    mode="classification",
    random_state=42
)

# Pick the first test instance to explain
instance       = X_test.values[0]
predicted_class = model.predict([instance])[0]
predicted_prob  = model.predict_proba([instance])[0][predicted_class]

# Generate LIME explanation
lime_exp = lime_explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=8,
    num_samples=5000
)

# Plot the explanation
feature_weights = lime_exp.as_list()
labels  = [fw[0] for fw in feature_weights]
weights = [fw[1] for fw in feature_weights]
colors  = ["#e74c3c" if w < 0 else "#2ecc71" for w in weights]

plt.figure(figsize=(10, 5))
plt.barh(labels[::-1], weights[::-1], color=colors[::-1])
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("LIME Feature Weight")
plt.title(f"LIME Explanation — Instance #0\n"
          f"Predicted: {'Bad Credit' if predicted_class == 1 else 'Good Credit'} "
          f"(prob = {predicted_prob:.2f})")
plt.tight_layout()
plt.savefig("figures/lime_explanation.png", dpi=150)
plt.close()
print("  Saved: figures/lime_explanation.png")


# STEP 6: LIME — STABILITY TEST (run 20 times, measure variance)

print("Running LIME stability test (20 runs)...")

all_weights = {}

for run in range(20):
    exp = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=8,
        num_samples=5000
    )
    for feat, weight in exp.as_list():
        all_weights.setdefault(feat, []).append(weight)

# Calculate mean and standard deviation for each feature
means = {f: np.mean(w) for f, w in all_weights.items()}
stds  = {f: np.std(w)  for f, w in all_weights.items()}

# Sort by absolute mean weight and take top 8
top_feats = sorted(means.keys(), key=lambda f: abs(means[f]), reverse=True)[:8]
colors    = ["#e74c3c" if means[f] < 0 else "#2ecc71" for f in top_feats]

plt.figure(figsize=(10, 5))
plt.barh(top_feats, [means[f] for f in top_feats],
         xerr=[stds[f] for f in top_feats],
         color=colors,
         error_kw={"ecolor": "black", "capsize": 4})
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Mean LIME Weight +/- Std Dev (20 runs)")
plt.title("LIME Stability Test — Instance #0\n"
          "Error bars show how much the explanation changes across runs")
plt.tight_layout()
plt.savefig("figures/lime_stability.png", dpi=150)
plt.close()
print("  Saved: figures/lime_stability.png")

# Print the age feature variance
for feat in top_feats:
    if "age" in feat.lower():
        print(f"  Age — mean: {means[feat]:.4f}, std: {stds[feat]:.4f}")


# ============================================================
# STEP 7: LIME — COMPARE AGE INFLUENCE: YOUNG vs OLDER
# ============================================================

print("\nComparing LIME age weights: young vs older applicants...")

young_weights = []
older_weights = []

# Go through 50 test instances and collect the LIME weight of 'age'
for i in range(min(50, len(X_test))):
    row = X_test.values[i]
    age = X_test.iloc[i]["age"]

    exp = lime_explainer.explain_instance(
        data_row=row,
        predict_fn=model.predict_proba,
        num_features=8,
        num_samples=3000
    )

    for feat, weight in exp.as_list():
        if "age" in feat.lower():
            if age < 30:
                young_weights.append(weight)
            else:
                older_weights.append(weight)

# Box plot comparing the two groups
plt.figure(figsize=(7, 5))
plt.boxplot(
    [young_weights, older_weights],
    labels=["Young (age < 30)", "Older (age >= 30)"],
    patch_artist=True,
    boxprops={"facecolor": "#3498db", "alpha": 0.7}
)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.ylabel("LIME weight of age feature")
plt.title("Does age hurt young applicants more?\n"
          "Negative weight = pushes toward 'Bad Credit'")
plt.tight_layout()
plt.savefig("figures/lime_age_groups.png", dpi=150)
plt.close()
print("  Saved: figures/lime_age_groups.png")

print(f"  Young — mean age weight: {np.mean(young_weights):.4f}")
print(f"  Older — mean age weight: {np.mean(older_weights):.4f}")


# ============================================================
# STEP 8: DiCE — COUNTERFACTUAL EXPLANATIONS
# ============================================================

print("\nRunning DiCE counterfactual explanations...")

# DiCE needs the training data with the target column included
train_df_for_dice = X_train.copy()
train_df_for_dice["credit_risk"] = y_train.values

# Tell DiCE which features are continuous
dice_data = dice_ml.Data(
    dataframe=train_df_for_dice,
    continuous_features=numeric_cols,
    outcome_name="credit_risk"
)

# Wrap the trained model for DiCE
dice_model = dice_ml.Model(model=model, backend="sklearn")

# Create the DiCE explainer
dice_explainer = Dice(dice_data, dice_model, method="random")

# Find a test instance predicted as Bad Credit to explain
bad_credit_instances = X_test[model.predict(X_test.values) == 1]
query = bad_credit_instances.iloc[[0]]

# Generate 4 counterfactuals: what needs to change to get Good Credit?
cf_result = dice_explainer.generate_counterfactuals(
    query,
    total_CFs=4,
    desired_class="opposite",
    random_seed=42
)

cf_df  = cf_result.cf_examples_list[0].final_cfs_df
orig   = query[numeric_cols].values[0]
cf_vals = cf_df[numeric_cols].values

print("\n  Original instance:")
for name, val in zip(numeric_cols, orig):
    print(f"    {name}: {val:.1f}")

print("\n  Counterfactuals (Good Credit predictions):")
print(cf_df[numeric_cols + ["credit_risk"]].to_string(index=False))

# Bar chart: how much does each feature need to change on average?
avg_change = np.mean(cf_vals - orig, axis=0)
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in avg_change]

plt.figure(figsize=(8, 5))
plt.bar(numeric_cols, avg_change, color=colors)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Average change from original")
plt.title("DiCE Counterfactuals — What Needs to Change?\n"
          "Green = needs to increase, Red = needs to decrease")
plt.tight_layout()
plt.savefig("figures/dice_counterfactuals.png", dpi=150)
plt.close()
print("  Saved: figures/dice_counterfactuals.png")


# ============================================================
# STEP 9: DiCE — AGE PERTURBATION TEST
# ============================================================

print("\nRunning age perturbation test...")

# Find a young applicant predicted as Bad Credit
young_bad = X_test[(model.predict(X_test.values) == 1) & (X_test["age"] < 30)]

if young_bad.empty:
    print("  No young bad-credit applicants found in test set.")
else:
    test_row = young_bad.iloc[[0]].copy()
    base_age = int(test_row["age"].values[0])

    ages  = list(range(base_age, 65))
    probs = []

    # Change only the age and re-predict — everything else stays the same
    for age in ages:
        temp = test_row.copy()
        temp["age"] = age
        prob_bad = model.predict_proba(temp.values)[0][1]
        probs.append(prob_bad)

    # Find the age where prediction flips to Good Credit
    flip_age = None
    for age, prob in zip(ages, probs):
        if prob < 0.5:
            flip_age = age
            break

    plt.figure(figsize=(10, 5))
    plt.plot(ages, probs, color="#e74c3c", linewidth=2, label="P(Bad Credit)")
    plt.axhline(0.5, color="black", linewidth=1, linestyle="--",
                label="Decision boundary (0.5)")

    if flip_age:
        plt.axvline(flip_age, color="#2ecc71", linewidth=2, linestyle=":",
                    label=f"Prediction flips at age {flip_age}")

    plt.xlabel("Applicant Age (everything else unchanged)")
    plt.ylabel("P(Bad Credit)")
    plt.title("Age Perturbation Test\n"
              "How does changing age alone affect the credit risk prediction?")
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/dice_age_perturbation.png", dpi=150)
    plt.close()
    print("  Saved: figures/dice_age_perturbation.png")
    print(f"  Base age: {base_age} | Prediction flips at age: {flip_age}")


# ============================================================
# DONE
# ============================================================

print("\nAll done! All figures saved in ./figures/")
