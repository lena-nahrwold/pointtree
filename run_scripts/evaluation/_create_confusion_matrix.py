"""Confusion Matrix for Segmentation Method Output."""

__all__ = ["create_confusion_matrix"]

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(
    target: np.ndarray,
    prediction: np.ndarray,
    class_map: Dict[str, int],
    aggregate_classes: Optional[Dict[str, List[int]]] = None,
    png_path: str = "confusion_matrix.png",
):
    """
    Calculates confusion matrix and optionally aggregates classes.
    Also generates a heatmap PNG.

    Args:
        target: Ground truth semantic class IDs (N,).
        prediction: Predicted semantic class IDs (N,).
        class_map: Mapping from class names to numeric class IDs.
        aggregate_classes: Optional dictionary that maps new class names to 
                           lists of numeric class IDs to merge.
        png_path: Path where the confusion matrix heatmap PNG will be saved.
    """
    if len(target) != len(prediction):
        raise ValueError("Target and prediction must have the same shape.")
    
    def save_confusion_matrix(cm, png_path):
        # Normalize matrix (row-wise)
        cm_normalized = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
        matrix_to_plot = cm_normalized.values

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix_to_plot, interpolation="nearest")

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels)

        # Draw values with adaptive text color
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                value = matrix_to_plot[i, j]
                rgba = im.cmap(im.norm(value))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if lum > 0.5 else "white"

                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center",
                    color=text_color
                )

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()
    
    ids = list(class_map.values())
    cm = confusion_matrix(target, prediction, labels=ids)
    class_labels = list(class_map.keys())

    # Convert to DataFrame for easier plotting
    cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    save_confusion_matrix(cm, f"{png_path}.png")

    if aggregate_classes:
        #aggregated_class_ids = []
        #for class_ids in aggregate_classes.values():
        #    aggregated_class_ids.extend(class_ids)

        # Create a mapping id → aggregated_name or original_name
        reverse_map = {v: k for k, v in class_map.items()}

        # Build lookup table: id → aggregated ID index
        all_ids = sorted(class_map.values())
        id_to_new = {}

        # Assign aggregated classes
        for new_name, id_list in aggregate_classes.items():
            for cid in id_list:
                id_to_new[cid] = new_name

        # Assign remaining classes to themselves
        for cid in all_ids:
            if cid not in id_to_new:
                id_to_new[cid] = reverse_map[cid]

        # Create new labels
        target_labels = np.array([id_to_new[c] for c in target])
        pred_labels   = np.array([id_to_new[c] for c in prediction])

        # Unique class names for consistent matrix
        class_labels = list(aggregate_classes.keys())

        # Build confusion matrix
        cm = pd.crosstab(
            pd.Series(target_labels, name="Actual"),
            pd.Series(pred_labels, name="Predicted"),
            rownames=["Actual"], colnames=["Predicted"],
            dropna=False
        ).reindex(index=class_labels, columns=class_labels, fill_value=0)

        save_confusion_matrix(cm, f"{png_path}_aggregated.png")

    return 

