from dataset_loader import TartanAirDatasetLoader
from feature_extractor import FeatureExtractor
from semantic_extractor import SemanticExtractor
from depth_features import DepthFeatureExtractor

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================

GENERATE_FEATURES = True
INSPECT_SEMANTICS = True
INSPECT_DEPTH = True

# =====================================================
# LOAD DATASET
# =====================================================

dataset = TartanAirDatasetLoader(
    r"E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy\P000"
)

feature_extractor = FeatureExtractor()
semantic_extractor = SemanticExtractor()
depth_extractor = DepthFeatureExtractor()

# =====================================================
# PART 1 : FEATURE EXTRACTION
# =====================================================

if GENERATE_FEATURES:

    rows = []

    print("\nExtracting Features...\n")

    for i in range(len(dataset)):

        sample = dataset.get_sample(i)

        # ----------------------------------------
        # RGB Features
        # ----------------------------------------

        visual_features = feature_extractor.extract(sample["rgb"])

        # ----------------------------------------
        # Semantic Features
        # ----------------------------------------

        semantic_features = semantic_extractor.extract(
            sample["segmentation"]
        )

        # ----------------------------------------
        # Depth Features
        # ----------------------------------------

        depth_features = depth_extractor.extract(
            sample["depth"]
        )

        # ----------------------------------------
        # Merge All Features
        # ----------------------------------------

        features = {}

        features.update(visual_features)
        features.update(semantic_features)
        features.update(depth_features)

        features["frame"] = i

        rows.append(features)

    df = pd.DataFrame(rows)

    # Replace missing semantic values with zero
    df.fillna(0, inplace=True)

    print("\n========== FEATURE TABLE ==========\n")
    print(df)

    print("\n==============================================")
    print("TOTAL FEATURES :", len(df.columns))
    print("==============================================")

    print("\n========== COLUMN NAMES ==========\n")

    for col in df.columns:
        print(col)

    print("\n==============================================")

    df.to_csv("features.csv", index=False)

    print("\nfeatures.csv saved successfully!")

# =====================================================
# PART 2 : SEMANTIC INFORMATION
# =====================================================

if INSPECT_SEMANTICS:

    sample = dataset.get_sample(0)

    print("\n========== SEMANTIC INFORMATION ==========\n")

    semantic_extractor.inspect_classes(
        sample["segmentation"]
    )

# =====================================================
# PART 3 : DEPTH INFORMATION
# =====================================================

if INSPECT_DEPTH:

    sample = dataset.get_sample(0)

    depth = sample["depth"]

    print("\n========== DEPTH INFORMATION ==========\n")

    print("Shape       :", depth.shape)
    print("Datatype    :", depth.dtype)
    print("Minimum     :", np.min(depth))
    print("Maximum     :", np.max(depth))

    print("\n========== CHANNEL SHAPES ==========\n")

    print("Red   :", depth[:, :, 0].shape)
    print("Green :", depth[:, :, 1].shape)
    print("Blue  :", depth[:, :, 2].shape)
    print("Alpha :", depth[:, :, 3].shape)

    # ----------------------------------------
    # Original Depth Image
    # ----------------------------------------

    plt.figure(figsize=(8, 8))
    plt.imshow(depth)
    plt.title("Original Depth Image")
    plt.axis("off")
    plt.show()

    # ----------------------------------------
    # Individual Channels
    # ----------------------------------------

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    titles = [
        "Red Channel",
        "Green Channel",
        "Blue Channel",
        "Alpha Channel"
    ]

    for i in range(4):

        ax[i // 2][i % 2].imshow(
            depth[:, :, i],
            cmap="gray"
        )

        ax[i // 2][i % 2].set_title(titles[i])

        ax[i // 2][i % 2].axis("off")

    plt.tight_layout()
    plt.show()

print("\n==============================================")
print("Pipeline completed successfully!")
print("==============================================")