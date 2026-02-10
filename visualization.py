"""Visualization functions for classification results."""

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def generate_confusion_matrix(results: List[Dict], output_file: str, model_name: Optional[str] = None) -> str:
    """
    Generate and save a confusion matrix visualization (raw and normalized).
    
    Args:
        results: List of result dictionaries with 'ground_truth' and 'model_prediction' keys
        output_file: Path to save the confusion matrix image (PNG format)
        model_name: Optional model name to include in titles
    
    Returns:
        Path to the saved confusion matrix file
    """
    # Filter out results with null predictions
    valid_results = [r for r in results if r.get("model_prediction") is not None]
    
    if not valid_results:
        print("Warning: No valid predictions to generate confusion matrix")
        return ""
    
    # Extract ground truth and predictions
    ground_truth = [r.get("ground_truth") for r in valid_results]
    predictions = [r.get("model_prediction") for r in valid_results]
    
    # Define all possible classes and their labels
    # Order requested: Neither (00), Anomaly (10), Hazard (01), Both (11)
    class_codes = ["00", "10", "01", "11"]
    class_labels = {
        "00": "Neither",
        "01": "Hazard",
        "10": "Anomaly",
        "11": "Both"
    }
    
    # Create label lists for display
    display_labels = [class_labels[code] for code in class_codes]
    
    # Generate confusion matrix using sklearn
    cm = confusion_matrix(
        ground_truth,
        predictions,
        labels=class_codes
    )

    # Normalized confusion matrix (row-normalized so rows sum to 1)
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
        cm_norm = np.nan_to_num(cm_norm)  # replace any NaNs from zero rows

    # Create the plot with two subplots: raw counts and normalized
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Confusion Matrix", "Normalized Confusion Matrix"]
    if model_name:
        titles = [f"Confusion Matrix - {model_name}", f"Normalized Confusion Matrix - {model_name}"]

    for ax, matrix, title, fmt in zip(axes, [cm, cm_norm], titles, ['d', '.2f']):
        im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(matrix.shape[1]),
               yticks=np.arange(matrix.shape[0]),
               xticklabels=display_labels,
               yticklabels=display_labels,
               title=title,
               ylabel='Ground Truth',
               xlabel='Predicted')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = matrix.max() / 2. if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > thresh else "black")

    fig.tight_layout()

    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {output_file}")
    return output_file
