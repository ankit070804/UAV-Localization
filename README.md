# UAV-Localization

Explainable AI (XAI) system for predicting and explaining **visual localization error** in UAVs (drones) flying in GPS-denied environments, using RGB, semantic segmentation, and depth data from the **TartanAir** simulation dataset.

The system doesn't just predict *how much* localization error to expect — it explains *why*, using SHAP-based feature attribution translated into a verified, human-readable report by a local LLM (via Ollama).

---

## How it works

```
TartanAir dataset (RGB + Depth + Segmentation + Pose)
              │
              ▼
   ┌─────────────────────────┐
   │   Feature Extraction     │
   │  • Visual (ORB, blur,    │
   │    contrast, brightness) │
   │  • Semantic (class %)    │
   │  • Depth (mean, entropy) │
   │  • Motion (dx, dy, dz)   │
   └─────────────────────────┘
              │
              ▼
   ┌─────────────────────────┐
   │  Random Forest Regressor │
   │  predicts localization   │
   │  error (meters)          │
   └─────────────────────────┘
              │
              ▼
   ┌─────────────────────────┐
   │   SHAP Explainer          │
   │  → top contributing       │
   │    features                │
   └─────────────────────────┘
              │
              ▼
   ┌─────────────────────────┐
   │  Reasoning Engine          │
   │  (reasoning_engine.py)     │
   │  SHAP feature → verified   │
   │  domain fact + recommend.  │
   └─────────────────────────┘
              │
              ▼
   ┌─────────────────────────┐
   │  LLM Report Generator      │
   │  (llm_engine.py, Ollama /  │
   │   llama3.2)                │
   │  → Professional XAI report │
   └─────────────────────────┘
```

The key design principle: **the LLM never reasons on its own about causes**. It only rewrites facts that the `reasoning_engine.py` module has already verified from SHAP output, into plain, professional English. This prevents hallucinated explanations.

---

## Project structure

| File | Purpose |
|---|---|
| `dataset_loader.py` | Loads TartanAir sequences (RGB, depth, segmentation, pose) frame by frame |
| `feature_extractor.py` | Extracts visual features from RGB frames (ORB keypoints, blur, contrast, brightness, edge density) |
| `semantic_extractor.py` | Extracts per-class percentage coverage from segmentation masks |
| `semantic_labels.py` | Class ID → semantic label mapping |
| `depth_features.py` | Extracts depth-based features (mean depth, depth entropy, valid depth ratio) |
| `pose_analyser.py` / `pose_analysis.py` | Computes UAV motion features (dx, dy, dz) between consecutive poses |
| `generate_all_features.py` | Runs feature extraction across all sequences/frames |
| `generate_all_labels.py` | Generates localization error labels for all sequences |
| `merge_dataset.py` / `merge_all_dataset.py` | Merges per-sequence features and labels into a single training dataset |
| `train_model.py` | Trains the Random Forest regressor, saves the model + feature importance plots/metrics |
| `evaluation_model.py` | Evaluates the trained model on held-out data |
| `loso_evaluation.py` | Leave-One-Sequence-Out cross-validation |
| `xai.py` | SHAP explainability utilities |
| `reasoning_engine.py` | Converts SHAP output into verified, rule-based domain facts and recommendations (no LLM involved) |
| `llm_engine.py` | Builds the LLM prompt from verified facts and calls a local **Ollama** model (`llama3.2`) to generate the final report |
| `uav_xai_system.py` | End-to-end demo: loads a frame, predicts error, runs SHAP, prints top reasons + recommendations |
| `localization.py` | Core localization utilities |
| `test_system.py` | System/integration tests |
| `main.py` | Feature-extraction and dataset-inspection entry point (RGB/semantic/depth visualization) |

**Generated artifacts** (produced by the scripts above, not hand-written):
`features.csv`, `all_features.csv`, `localization_labels.csv`, `all_localization_labels.csv`, `training_dataset.csv`, `training_dataset_all.csv`, `feature_importance.csv`, `evaluation_feature_importance.csv`, `evaluation_results.csv`, `loso_results.csv`, `test_predictions.csv`, `rf_localization.pkl`, `feature_names.pkl`, and plots (`actual_vs_predicted.png`, `error_distribution.png`, `feature_importance.png`, `shap_bar.png`, `shap_summary.png`, `sample_error.png`).

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed locally, with the `llama3.2` model pulled:
  ```bash
  ollama pull llama3.2
  ```
- The [TartanAir](https://theairlab.org/tartanair-dataset/) dataset (RGB, depth, segmentation, and pose data) downloaded locally

### Python packages

```bash
pip install pandas numpy opencv-python scikit-learn shap joblib matplotlib ollama
```

> Adjust based on the actual imports in `dataset_loader.py` / `feature_extractor.py` / `depth_features.py` if you're using extra libraries (e.g. `Pillow`, `scipy`).

---

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/ankit070804/UAV-Localization.git
   cd UAV-Localization
   ```
2. Update the dataset path. Several scripts (`main.py`, `uav_xai_system.py`, `generate_all_features.py`, etc.) currently hardcode a local Windows path, e.g.:
   ```python
   DATASET_PATH = r"E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy\P000"
   ```
   Update this to point to your own TartanAir sequence folder before running anything.
3. Make sure Ollama is running in the background (`ollama serve`) before running `llm_engine.py` or any script that generates a final report.

---

## Usage

### 1. Explore a dataset sequence / generate features for one sequence
```bash
python main.py
```
Extracts visual, semantic, and depth features frame-by-frame for the configured sequence, saves `features.csv`, and (optionally) visualizes semantic/depth information.

### 2. Build the full training dataset (all sequences)
```bash
python generate_all_features.py
python generate_all_labels.py
python merge_all_dataset.py
```
Produces `training_dataset_all.csv`, the merged feature + label table used for training.

### 3. Train the model
```bash
python train_model.py
```
Trains a `RandomForestRegressor` on `training_dataset_all.csv` to predict `localization_error`, then saves:
- `rf_localization.pkl` — the trained model
- `feature_names.pkl` — the feature column order
- `feature_importance.csv` / `.png`, `actual_vs_predicted.png`, `error_distribution.png`
- Prints MAE, RMSE, and R² on a 20% held-out test split.

### 4. Evaluate the model
```bash
python evaluation_model.py
python loso_evaluation.py   # Leave-One-Sequence-Out cross-validation
```

### 5. Run the end-to-end explainability demo
```bash
python uav_xai_system.py
```
Loads a sample frame, predicts localization error, runs SHAP to find the top contributing features, and prints a rule-based explanation with recommendations (no LLM needed for this step).

### 6. Generate a natural-language XAI report (LLM)
```bash
python llm_engine.py
```
Runs a dummy example end-to-end: SHAP values → `reasoning_engine.explain()` → verified facts → `llm_engine.generate_report()` → a professional report via the local `llama3.2` model, structured as:
```
Localization Summary
Reasons for Localization Error
Recommendations
Risk Level (Low / Medium / High)
```

---

## Model details

- **Model:** `RandomForestRegressor` (300 trees, `scikit-learn`)
- **Target:** `localization_error` (meters)
- **Features used:** ORB keypoint count, blur, contrast, brightness, edge density, per-class semantic percentage coverage, mean depth, depth entropy, valid depth ratio, and frame-to-frame motion (`dx`, `dy`, `dz`)
- **Explainability:** SHAP `TreeExplainer` on the trained Random Forest
- **Risk bands** used in generated reports:
  - Low: error < 0.30 m
  - Medium: 0.30–0.80 m
  - High: > 0.80 m

---

## Notes

- The LLM (`llm_engine.py`) is explicitly instructed never to mention SHAP, Random Forest, machine learning, ORB-SLAM, SIFT, or SURF by name in the generated report — it only produces a clean, professional explanation for an end user based on pre-verified facts.
- Dataset paths are currently hardcoded to a local machine path in several scripts — update these before running on a new machine.
- `__pycache__/` is currently committed to the repo; consider adding a `.gitignore` to exclude it along with large generated artifacts (`.pkl`, `.csv`, `.png`) if you don't want them version-controlled.

---
