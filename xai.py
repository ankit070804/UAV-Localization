import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==========================================
# Load Model
# ==========================================

model = joblib.load("rf_localization.pkl")

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("training_dataset.csv")

TARGET = "localization_error"

DROP_COLUMNS = [
    "frame",
    "ground_truth_distance",
    "estimated_distance",
    TARGET
]

X = df.drop(columns=DROP_COLUMNS)

print("Samples :", len(X))
print("Features:", X.shape[1])

# ==========================================
# SHAP
# ==========================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

print("\nSHAP values computed successfully!")

# ==========================================
# BAR PLOT
# ==========================================

plt.figure(figsize=(12,6))

shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# BEESWARM
# ==========================================

plt.figure(figsize=(12,6))

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print("shap_bar.png")
print("shap_summary.png")