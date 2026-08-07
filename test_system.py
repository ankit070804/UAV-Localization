import pandas as pd
import joblib
import shap

from dataset_loader import TartanAirDatasetLoader
from feature_extractor import FeatureExtractor
from semantic_extractor import SemanticExtractor
from depth_features import DepthFeatureExtractor

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("rf_localization.pkl")
feature_names = joblib.load("feature_names.pkl")
explainer = shap.TreeExplainer(model)

# =====================================================
# DATASET
# =====================================================

dataset = TartanAirDatasetLoader(
    r"E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy\P006"
)

feature_extractor = FeatureExtractor()
semantic_extractor = SemanticExtractor()
depth_extractor = DepthFeatureExtractor()

# =====================================================
# SELECT FRAME
# =====================================================

frame = 20
sample = dataset.get_sample(frame)

# =====================================================
# FEATURE EXTRACTION
# =====================================================

features = {}

features.update(feature_extractor.extract(sample["rgb"]))
features.update(semantic_extractor.extract(sample["segmentation"]))
features.update(depth_extractor.extract(sample["depth"]))

# ---------------- Motion ----------------

try:

    pose1 = dataset.poses[frame]
    pose2 = dataset.poses[frame + 1]

    features["dx"] = pose2[0] - pose1[0]
    features["dy"] = pose2[1] - pose1[1]
    features["dz"] = pose2[2] - pose1[2]

except:

    features["dx"] = 0
    features["dy"] = 0
    features["dz"] = 0

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame([features])

for col in feature_names:

    if col not in df.columns:
        df[col] = 0

df = df[feature_names]

# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(df)[0]

print("\n====================================================")
print(" UAV LOCALIZATION EXPLANATION SYSTEM")
print("====================================================")

print(f"\nPredicted Localization Error : {prediction:.3f} meters")

# =====================================================
# SHAP
# =====================================================

shap_values = explainer.shap_values(df)

importance = pd.DataFrame({

    "Feature": feature_names,
    "Value": df.iloc[0].values,
    "SHAP": shap_values[0]

})

importance["ABS"] = importance["SHAP"].abs()
importance = importance.sort_values("ABS", ascending=False)

top = importance.head(5)

print("\nTop Influencing Factors\n")
print(top[["Feature", "Value", "SHAP"]])

# =====================================================
# XAI REASONING
# =====================================================

from reasoning_engine import explain
from llm_engine import generate_report

facts, recommendations = explain(top)

print("\n====================================================")
print("Verified XAI Reasoning")
print("====================================================\n")

for fact in facts:

    print(f"• {fact['reason']}")
    print(f"  Effect : {fact['effect']}")
    print()

# =====================================================
# LLM REPORT
# =====================================================

print("\n====================================================")
print("LLM Generated Report")
print("====================================================\n")

report = generate_report(
    prediction,
    facts,
    recommendations
)

print(report)

print("\n====================================================")
print("Analysis Completed Successfully")
print("====================================================")