import cv2
import numpy as np
import pandas as pd

from dataset_loader import TartanAirDatasetLoader

# =====================================================
# DATASET
# =====================================================

dataset = TartanAirDatasetLoader(
    r"E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy\P000"
)

# =====================================================
# ORB Detector
# =====================================================

orb = cv2.ORB_create(5000)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Approximate camera intrinsic matrix
# (Good enough for prototype)

K = np.array([
    [320,   0, 320],
    [0,   320, 320],
    [0,     0,   1]
], dtype=np.float64)

rows = []

print("Running Localization...\n")

# =====================================================
# LOOP
# =====================================================

for i in range(len(dataset)-1):

    sample1 = dataset.get_sample(i)
    sample2 = dataset.get_sample(i+1)

    img1 = sample1["rgb"]
    img2 = sample2["rgb"]

    pose1 = sample1["pose"]
    pose2 = sample2["pose"]

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:

        continue

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 8:

        continue

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    )

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    )

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        cv2.RANSAC,
        0.999,
        1.0
    )

    if E is None:

        continue

    _, R, t, mask = cv2.recoverPose(
        E,
        pts1,
        pts2,
        K
    )

    # -------------------------
    # Ground Truth Translation
    # -------------------------

    gt_motion = pose2[:3] - pose1[:3]

    gt_distance = np.linalg.norm(gt_motion)

    est_distance = np.linalg.norm(t)

    localization_error = abs(
        gt_distance - est_distance
    )

    rows.append({

        "frame": i,

        "matches": len(matches),

        "ground_truth_distance": gt_distance,

        "estimated_distance": est_distance,

        "localization_error": localization_error

    })

    if i % 10 == 0:

        print(
            f"Frame {i:3d} | "
            f"GT={gt_distance:.3f} | "
            f"EST={est_distance:.3f} | "
            f"ERR={localization_error:.3f}"
        )

# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(rows)

df.to_csv(
    "localization_labels.csv",
    index=False
)

print("\n===================================")
print("Localization Completed")
print("===================================")

print(df.head())

print("\nSaved as localization_labels.csv")