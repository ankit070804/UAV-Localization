import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# LOAD DATASET
# ============================================

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

df = pd.read_csv("training_dataset_all.csv")

print("Dataset Shape :", df.shape)

print("\nColumns:\n")
for col in df.columns:
    print(col)

# ============================================
# TARGET COLUMN
# ============================================

TARGET = "localization_error"

# ============================================
# DROP ONLY EXISTING COLUMNS
# ============================================

DROP_COLUMNS = [
    "frame",
    "sequence",
    "ground_truth_distance",
    "estimated_distance",
    "translation_distance",
    "matches",
    TARGET
]

DROP_COLUMNS = [
    col for col in DROP_COLUMNS
    if col in df.columns
]

print("\nDropped Columns:")
print(DROP_COLUMNS)

# ============================================
# FEATURES / LABELS
# ============================================

X = df.drop(columns=DROP_COLUMNS)

y = df[TARGET]

print("\nNumber of Features :", X.shape[1])

# ============================================
# LOAD MODEL
# ============================================

print("\nLoading trained model...")

model = joblib.load("rf_localization.pkl")

print("Done!")

# ============================================
# PREDICTION
# ============================================

pred = model.predict(X)

# ============================================
# METRICS
# ============================================

mae = mean_absolute_error(y, pred)

rmse = mean_squared_error(
    y,
    pred
) ** 0.5

r2 = r2_score(
    y,
    pred
)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

# ============================================
# SAVE RESULTS
# ============================================

results = df.copy()

results["predicted_error"] = pred

results["absolute_error"] = (
    results[TARGET] - pred
).abs()

results.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nSaved : evaluation_results.csv")

# ============================================
# ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(7,7))

plt.scatter(
    y,
    pred,
    alpha=0.7
)

minimum = min(y.min(), pred.min())
maximum = max(y.max(), pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    'r--'
)

plt.xlabel("Actual Localization Error")

plt.ylabel("Predicted Localization Error")

plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ============================================
# ERROR HISTOGRAM
# ============================================

plt.figure(figsize=(7,5))

plt.hist(
    results["absolute_error"],
    bins=20
)

plt.xlabel("Absolute Error (m)")

plt.ylabel("Frequency")

plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    "error_distribution.png",
    dpi=300
)

plt.show()

# ============================================
# ERROR PER SAMPLE
# ============================================

plt.figure(figsize=(12,5))

plt.plot(
    results["absolute_error"].values
)

plt.xlabel("Frame")

plt.ylabel("Absolute Error")

plt.title("Localization Error per Sample")

plt.tight_layout()

plt.savefig(
    "sample_error.png",
    dpi=300
)

plt.show()

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "evaluation_feature_importance.csv",
    index=False
)

plt.figure(figsize=(10,8))

top20 = importance.head(20)

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Top 20 Feature Importance")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("actual_vs_predicted.png")
print("error_distribution.png")
print("sample_error.png")
print("feature_importance.png")
print("evaluation_feature_importance.csv")

print("\n" + "=" * 50)
print("Evaluation Completed Successfully")
print("=" * 50)