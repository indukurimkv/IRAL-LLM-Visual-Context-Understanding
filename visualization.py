"""Visualization functions for classification results."""

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def generate_confusion_matrix(results: List[Dict], output_file: str) -> str:
    """
    Generate and save a confusion matrix visualization.
    
    Args:
        results: List of result dictionaries with 'ground_truth' and 'model_prediction' keys
        output_file: Path to save the confusion matrix image (PNG format)
    
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
    class_codes = ["00", "01", "10", "11"]
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
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=display_labels,
           yticklabels=display_labels,
           title='Confusion Matrix',
           ylabel='Ground Truth',
           xlabel='Predicted')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_file}")
    return output_file
