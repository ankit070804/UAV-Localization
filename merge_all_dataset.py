import pandas as pd

# ============================================
# LOAD FILES
# ============================================

features = pd.read_csv("all_features.csv")
labels = pd.read_csv("all_localization_labels.csv")

print("=" * 50)
print("Loaded Files")
print("=" * 50)

print("Features :", features.shape)
print("Labels   :", labels.shape)

# ============================================
# MERGE
# ============================================

training = pd.merge(
    features,
    labels,
    on=["sequence", "frame"],
    how="inner"
)

print("\nMerged Shape :", training.shape)

# ============================================
# REMOVE DUPLICATE COLUMNS (if any)
# ============================================

training = training.loc[:, ~training.columns.duplicated()]

# ============================================
# CHECK FOR MISSING VALUES
# ============================================

print("\nMissing Values")

missing = training.isnull().sum()

print(missing[missing > 0])

training.fillna(0, inplace=True)

# ============================================
# SAVE
# ============================================

training.to_csv(
    "training_dataset_all.csv",
    index=False
)

print("\n========================================")
print("Training Dataset Created Successfully")
print("========================================")

print("\nShape :", training.shape)

print("\nColumns:\n")
print(training.columns.tolist())

print("\nFirst Five Rows\n")
print(training.head())