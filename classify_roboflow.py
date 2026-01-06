#!/usr/bin/env python3
"""
Roboflow to OpenRouter Classification Script

This script fetches images and annotations from Roboflow via REST API,
queries OpenRouter's Claude 3.5 Sonnet Vision model for AH classification,
and stores results with ground truth annotations.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from openrouter_client import OpenRouterClient
from processor import ClassificationProcessor
from roboflow_client import RoboflowAPIClient

# Load environment variables from .env file
load_dotenv()


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Classify Roboflow images using OpenRouter API"
    )
    # Arguments for Roboflow project identification (can be from .env or CLI)
    parser.add_argument(
        "--workspace-id",
        default=os.getenv("ROBOFLOW_WORKSPACE_ID"),
        help="Roboflow workspace ID (or set ROBOFLOW_WORKSPACE_ID in .env file)"
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("ROBOFLOW_PROJECT_ID"),
        help="Roboflow project ID (or set ROBOFLOW_PROJECT_ID in .env file)"
    )
    # API keys can be provided via CLI, .env file, or environment variables
    parser.add_argument(
        "--roboflow-api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY in .env file)"
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY in .env file)"
    )
    # Optional arguments with defaults
    parser.add_argument(
        "--output",
        default="classification_results.json",
        help="Output JSON file path (default: classification_results.json)"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="OpenRouter model to use (default: anthropic/claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--confusion-matrix-output",
        default=None,
        help="Output path for confusion matrix image (default: <output_file>_confusion_matrix.png)"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate that API keys are provided (either via CLI, .env, or env vars)
    if not args.roboflow_api_key:
        print("Error: Roboflow API key is required. Set ROBOFLOW_API_KEY in .env file or use --roboflow-api-key")
        sys.exit(1)
    
    if not args.openrouter_api_key:
        print("Error: OpenRouter API key is required. Set OPENROUTER_API_KEY in .env file or use --openrouter-api-key")
        sys.exit(1)
    
    # Validate that workspace and project IDs are provided
    if not args.workspace_id:
        print("Error: Roboflow workspace ID is required. Set ROBOFLOW_WORKSPACE_ID in .env file or use --workspace-id")
        sys.exit(1)
    
    if not args.project_id:
        print("Error: Roboflow project ID is required. Set ROBOFLOW_PROJECT_ID in .env file or use --project-id")
        sys.exit(1)
    
    # Initialize API clients with credentials
    print("Initializing clients...")
    roboflow_client = RoboflowAPIClient(
        args.roboflow_api_key,
        args.workspace_id,
        args.project_id
    )
    
    openrouter_client = OpenRouterClient(args.openrouter_api_key, args.model)
    
    # Create main processor that orchestrates the workflow
    processor = ClassificationProcessor(roboflow_client, openrouter_client)
    
    # Process all images in the project
    try:
        processor.process_all_images()
    except KeyboardInterrupt:
        # Handle user interruption gracefully - save partial results
        print("\nInterrupted by user. Saving partial results...")
    except Exception as e:
        # Handle fatal errors
        print(f"Fatal error: {e}")
        sys.exit(1)
    
    # Save all results and statistics to JSON file
    processor.save_results(args.output, args.confusion_matrix_output)


if __name__ == "__main__":
    main()
