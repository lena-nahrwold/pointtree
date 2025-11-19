"""Align pointclouds based on nearest neighbors."""

__all__ = ["nearest_neighbor_alignment"]

from pointtorch import read
from scipy.spatial import cKDTree
import numpy as np
import json
from textwrap import indent

def nearest_neighbor_alignment(
    target: np.ndarray,
    prediction: np.ndarray
) -> np.ndarray:

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

    return prediction_aligned

