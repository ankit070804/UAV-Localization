import joblib
import shap
import pandas as pd

from dataset_loader import TartanAirDatasetLoader
from feature_extractor import FeatureExtractor
from semantic_extractor import SemanticExtractor
from depth_features import DepthFeatureExtractor
from pose_analyser import PoseAnalyzer

# --------------------------------------------

DATASET_PATH = r"E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy\P000"

# --------------------------------------------

dataset = TartanAirDatasetLoader(DATASET_PATH)

feature_extractor = FeatureExtractor()
semantic_extractor = SemanticExtractor()
depth_extractor = DepthFeatureExtractor()
pose_analyzer = PoseAnalyzer()

# --------------------------------------------

model = joblib.load("rf_localization.pkl")

feature_names = joblib.load("feature_names.pkl")

explainer = shap.TreeExplainer(model)

# --------------------------------------------

frame = 20

sample = dataset.get_sample(frame)

features = {}

features.update(feature_extractor.extract(sample["rgb"]))

features.update(
    semantic_extractor.extract(sample["segmentation"])
)

features.update(
    depth_extractor.extract(sample["depth"])
)

pose1 = dataset.get_sample(frame)["pose"]
pose2 = dataset.get_sample(frame+1)["pose"]

features.update(
    pose_analyzer.motion_features(pose1, pose2)
)

features["matches"] = 1800

# --------------------------------------------

row = pd.DataFrame([features])

for col in feature_names:

    if col not in row.columns:

        row[col] = 0

row = row[feature_names]

prediction = model.predict(row)[0]

print()

print("="*50)
print(" UAV-XAI SYSTEM")
print("="*50)

print()

print(f"Predicted Localization Error : {prediction:.3f} m")

# --------------------------------------------

values = explainer.shap_values(row)

if isinstance(values, list):
    values = values[0]

shap_values = values[0]

importance = pd.DataFrame({

    "Feature":feature_names,
    "Value":row.iloc[0].values,
    "SHAP":shap_values,
    "Importance":abs(shap_values)

})

importance = importance.sort_values(
    "Importance",
    ascending=False
).head(5)

print()

print("Top Reasons")

print()

print(
importance[
["Feature","Value","SHAP"]
]
)

print()

print("="*50)
print("Explanation")
print("="*50)

mapping = {

28:"Furniture",
229:"Shelf",
180:"Carpet",
157:"Wall",
199:"Door",
74:"Window",
87:"Ceiling",
64:"Floor"

}

for _,r in importance.iterrows():

    f = r["Feature"]

    if f.startswith("class_"):

        cid = int(f.split("_")[1])

        name = mapping.get(cid,f"Class {cid}")

        print(
f"• Large percentage of {name} influenced localization."
)

    elif f=="contrast":

        print(
"• Low image contrast reduced reliable feature detection."
)

    elif f=="brightness":

        print(
"• Poor illumination reduced localization quality."
)

    elif f=="matches":

        print(
"• Few feature matches increased localization error."
)

    elif "depth" in f:

        print(
"• Scene depth variation influenced localization."
)

    elif "orb" in f:

        print(
"• Limited ORB keypoints reduced pose estimation accuracy."
)

print()

print("Recommendations")

print()

print("✓ Reduce UAV speed")

print("✓ Increase illumination")

print("✓ Enable Visual-Inertial Localization")

print("✓ Avoid texture-poor regions")