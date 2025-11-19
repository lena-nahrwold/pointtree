"""Sort pointclouds based on nearest neighbors."""

__all__ = ["nearest_neighbor_sorting"]

from scipy.spatial import cKDTree
import numpy as np
from pointtorch import read

def nearest_neighbor_sorting(
    target: np.ndarray,
    prediction: np.ndarray
) -> np.ndarray:

    pred_xyz    = prediction[['x','y','z']].values
    gt_xyz      = target[['x','y','z']].values
    pred_labels = prediction['classification'].to_numpy()
    gt_labels   = target['classification'].to_numpy()

    # Build KDTree on predicted points
    tree = cKDTree(pred_xyz)

    # Query nearest neighbor for each target point
    dists, idxs = tree.query(gt_xyz, k=1)

    # Check distance distribution
    print("Distances: min {:.6f}, median {:.6f}, max {:.6f}".format(dists.min(), np.median(dists), dists.max()))

    # Use threshold slightly above maximum expected float differences
    threshold   = 0.01
    mask_within = dists <= threshold

    # Assign predicted labels to GT points
    aligned_pred_labels = np.full(len(gt_labels), -1, dtype=np.int16)
    aligned_pred_labels[mask_within] = pred_labels[idxs[mask_within]]

    # Keep only matched points
    prediction = aligned_pred_labels[mask_within]
    target     = gt_labels[mask_within]

    print("Matched points:", prediction.shape[0], "/", len(gt_labels))

    return prediction