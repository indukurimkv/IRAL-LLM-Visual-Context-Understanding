#!/usr/bin/env python3
"""
Standalone script for analyzing existing classification results.

This script loads results from a JSON file and generates metrics and visualizations
without requiring API calls or re-running classification.
"""

import argparse
import json
import os
import sys

from metrics import calculate_metrics
from visualization import generate_confusion_matrix


def main():
    """Main function for analyzing results."""
    parser = argparse.ArgumentParser(
        description="Analyze existing classification results without API calls"
    )
    parser.add_argument(
        "--input",
        default="classification_results.json",
        help="Input JSON file path (default: classification_results.json)"
    )
    parser.add_argument(
        "--confusion-matrix-output",
        default=None,
        help="Output path for confusion matrix image (default: <input_file>_confusion_matrix.png)"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name to include in confusion matrix title"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Load results from JSON file
    print(f"Loading results from {args.input}...")
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.input}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {args.input}: {e}")
        sys.exit(1)
    
    # Extract results list
    results = data.get("results", [])
    if not results:
        print("Error: No results found in input file")
        sys.exit(1)
    
    print(f"Found {len(results)} results")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    # Print metrics
    print("\n" + "="*50)
    print("METRICS SUMMARY")
    print("="*50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("="*50)
    
    # Calculate accuracy
    matches = sum(1 for r in results if r.get("match", False))
    accuracy = matches / len(results) if results else 0.0
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy:.2%})")
    
    # Generate confusion matrix
    if args.confusion_matrix_output is None:
        base_name = os.path.splitext(args.input)[0]
        confusion_matrix_file = f"{base_name}_confusion_matrix.png"
    else:
        confusion_matrix_file = args.confusion_matrix_output
    
    print(f"\nGenerating confusion matrix...")
    confusion_matrix_path = generate_confusion_matrix(results, confusion_matrix_file, model_name=args.model_name)
    
    if confusion_matrix_path:
        print(f"\nAnalysis complete!")
        print(f"Confusion matrix saved to: {confusion_matrix_path}")
    else:
        print("\nWarning: Could not generate confusion matrix")


if __name__ == "__main__":
    main()
