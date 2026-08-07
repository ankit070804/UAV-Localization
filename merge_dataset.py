import pandas as pd

# ==========================
# Load files
# ==========================

features = pd.read_csv("features.csv")
labels = pd.read_csv("localization_labels.csv")

# ==========================
# Merge on frame number
# ==========================

dataset = pd.merge(
    features,
    labels,
    on="frame",
    how="inner"
)

# ==========================
# Save
# ==========================

dataset.to_csv(
    "training_dataset.csv",
    index=False
)

print("\n=================================")
print("Training Dataset Created")
print("=================================\n")

print("Shape :", dataset.shape)

print("\nColumns:\n")
print(dataset.columns.tolist())

print("\nFirst Five Rows\n")
print(dataset.head())