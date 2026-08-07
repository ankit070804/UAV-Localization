from dataset_loader import TartanAirDatasetLoader
from pose_analyser import PoseAnalyzer

# ==========================================
# Load Dataset
# ==========================================

dataset = TartanAirDatasetLoader(
    r"E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy\P000"
)

pose_analyzer = PoseAnalyzer()

print("\n========================================")
print("POSE ANALYSIS")
print("========================================\n")

print(f"Total Frames : {len(dataset)}\n")

for i in range(min(10, len(dataset) - 1)):

    sample1 = dataset.get_sample(i)
    sample2 = dataset.get_sample(i + 1)

    pose1 = sample1["pose"]
    pose2 = sample2["pose"]

    motion = pose_analyzer.motion_features(pose1, pose2)

    print(f"Frame {i} -> {i+1}")
    print("-" * 35)
    print(f"dx                    : {motion['dx']:.6f}")
    print(f"dy                    : {motion['dy']:.6f}")
    print(f"dz                    : {motion['dz']:.6f}")
    print(f"Translation Distance  : {motion['translation_distance']:.6f}")
    print()