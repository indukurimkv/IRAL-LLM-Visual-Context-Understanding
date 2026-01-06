"""Metrics calculation functions for classification results."""

from typing import Dict, List


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score using standard formulas.
    
    Uses macro-averaging: calculates precision/recall per class, then averages.
    Filters out results where model_prediction is None.
    
    Args:
        results: List of result dictionaries with 'ground_truth' and 'model_prediction' keys
    
    Returns:
        Dictionary with 'precision', 'recall', and 'f1_score' keys
    """
    # Filter out results with null predictions
    valid_results = [r for r in results if r.get("model_prediction") is not None]
    
    if not valid_results:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    # Get all unique classes
    all_classes = set()
    for r in valid_results:
        all_classes.add(r.get("ground_truth"))
        all_classes.add(r.get("model_prediction"))
    all_classes = sorted(list(all_classes))
    
    # Calculate precision and recall per class
    precisions = []
    recalls = []
    
    for cls in all_classes:
        # True Positives: predicted as cls and actually cls
        tp = sum(1 for r in valid_results 
                 if r.get("ground_truth") == cls and r.get("model_prediction") == cls)
        
        # False Positives: predicted as cls but actually not cls
        fp = sum(1 for r in valid_results 
                 if r.get("ground_truth") != cls and r.get("model_prediction") == cls)
        
        # False Negatives: actually cls but predicted as something else
        fn = sum(1 for r in valid_results 
                 if r.get("ground_truth") == cls and r.get("model_prediction") != cls)
        
        # Calculate precision for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)
        
        # Calculate recall for this class
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    
    # Macro-averaging: average across all classes
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    # Calculate F1 score from averaged precision and recall
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    
    return {
        "precision": round(avg_precision, 4),
        "recall": round(avg_recall, 4),
        "f1_score": round(f1_score, 4)
    }
