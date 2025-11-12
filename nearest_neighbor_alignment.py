from pointtorch import read
from scipy.spatial import cKDTree
import numpy as np
from pointtree.evaluation import semantic_segmentation_metrics
import json
from textwrap import indent

segmentation_tool = "point2tree"

# Load clouds
target_path = "./data/manual_shift_segmented.las"
target = read(target_path)

prediction_path = f"./data/old/{segmentation_tool}_segmented.las"
prediction = read(prediction_path)

# Convert to numpy arrays
pred_xyz = prediction[['x','y','z']].to_numpy()
gt_xyz   = target[['x','y','z']].to_numpy()

# Labels (optional, for evaluation)
pred_labels = prediction['classification'].to_numpy()
gt_labels   = target['classification'].to_numpy()

# Build KDTree on predicted points
tree = cKDTree(pred_xyz)

# Find nearest predicted point for each target point
dists, idxs = tree.query(gt_xyz, k=1)

# Check distance distribution
print("Distances: min {:.6f}, median {:.6f}, max {:.6f}".format(dists.min(), np.median(dists), dists.max()))

# Threshold to account for floating-point differences
threshold = dists.max() + 0.000001
mask_within = dists <= threshold

# Only keep predicted points that match target points
matched_indices = idxs[mask_within]

# Filter prediction DataFrame
prediction_aligned = prediction.iloc[matched_indices].reset_index(drop=True)

print("Prediction cloud aligned to target. Number of points:", len(prediction_aligned))

target.sort_values(by=['x','y','z'], ascending=True, inplace=True, ignore_index=True)
prediction_aligned.sort_values(by=['x','y','z'], ascending=True, inplace=True, ignore_index=True)

class_map = {
    "wood":         64,
    "leaf tree":    65,
    "human-made":   66,
    "ground":       67,
    "grasses":      68,
    "animal/human": 69,
    "uncertain":    70,
    "leaf shrub":   71
}

aggregate_classes = {
    "leaves":  [65, 68, 71],
    "other":   [64, 66, 67, 69, 70]
}

evaluation = semantic_segmentation_metrics(
    target['classification'],
    prediction_aligned['classification'],
    class_map,
    aggregate_classes=aggregate_classes
)

# Replace NaN with 0 to avoid JSON serialization issues
evaluation_clean = {k: float(np.nan_to_num(v)) for k, v in evaluation.items()}

output_path = f"./output/{segmentation_tool}_evaluation_results.txt"
with open(output_path, "w") as f:
    f.write(f"{segmentation_tool} Semantic Segmentation Evaluation Results\n")
    f.write("=" * 50 + "\n\n")

    # Print each metric with alignment
    for key, val in evaluation_clean.items():
        f.write(f"{key:<30}: {val:.4f}\n")

print(f"Results saved to {output_path}")