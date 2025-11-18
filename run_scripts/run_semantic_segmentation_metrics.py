""" 
pointtree.evaluation.semantic_segmentation_metrics(

    target: ndarray,
    prediction: ndarray,
    class_map: Dict[str, int],
    aggregate_classes: Dict[str, List[int]] | None = None,

) → Dict[str, float][source]

    Calculates semantic segmentation metrics.

    Parameters:

            target – Ground truth semantic class IDs for each point.

            prediction – Predicted semantic class IDs for each point.

            class_map – A dictionary mapping class names to numeric class IDs.

            aggregate_classes – A dictionary with which aggregations of classes can be defined. The keys are the names of the aggregated classes and the values are lists of the IDs of the classes to be aggregated.

    Returns:

        A dictionary containing the following keys for each semantic class: "<class_name>_iou", "<class_name>_precision", "<class_name>_recall". For each aggregated class, the keys "<class_name>_iou_aggregated", "<class_name>_precision_aggregated", "<class_name>_recall_aggregated" are provided.
"""
from pointtorch import read
from scipy.spatial import cKDTree
import numpy as np
from pointtree.evaluation import semantic_segmentation_metrics
from evaluation_helper import create_confusion_matrix
import json
from textwrap import indent

segmentation_tool = "fsct_shift"

# Load point cloud (supports .txt, .csv, .las, .laz, .ply)
target_path = "./data/manual_segmented.las"
target = read(target_path)
#print(target.columns)

prediction_path = f"./data/{segmentation_tool}_segmented.las"
prediction = read(prediction_path)
#print(prediction.columns)

#print(target.shape)
#print(prediction.shape)

target.sort_values(by=['x','y','z'], ascending=True, inplace=True, ignore_index=True)
prediction.sort_values(by=['x','y','z'], ascending=True, inplace=True, ignore_index=True)

class_map = {
    "crown":        65,
    "shrub":        71,
    "grasses":      68,
    "wood":         64,
    "ground":       67,
    "human-made":   66,
    "animal/human": 69,
    "uncertain":    70
}

aggregate_classes = {
    "leaves":  [65, 68, 71],
    "other":   [64, 66, 67, 69, 70]
}

#evaluation = semantic_segmentation_metrics(
#    target['classification'],
#    prediction['classification'],
#    class_map,
#    aggregate_classes=aggregate_classes
#)
#
## Replace NaN with 0 to avoid JSON serialization issues
#evaluation_clean = {k: float(np.nan_to_num(v)) for k, v in evaluation.items()}
#
#output_path = f"./output/{segmentation_tool}_evaluation_results.txt"
#with open(output_path, "w") as f:
#    f.write(f"{segmentation_tool} Semantic Segmentation Evaluation Results\n")
#    f.write("=" * 50 + "\n\n")
#
#    # Print each metric with alignment
#    for key, val in evaluation_clean.items():
#        f.write(f"{key:<30}: {val:.4f}\n")
#
#print(f"Results saved to {output_path}")

aggregate_classes = {
    "vegetation":  [65, 68, 71],
    "wood":   [64],
    "ground": [67],
    "human-made": [66],
    "other": [70, 69]
}

png_path = f"./output/{segmentation_tool}_confusion_matrix.png"
matrix, labels = create_confusion_matrix(
    target["classification"],
    prediction["classification"],
    class_map,
    None, #aggregate_classes,
    png_path
)

print(matrix)
print(labels)