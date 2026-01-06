# Roboflow to OpenRouter Classification Script

This script fetches images and annotations from Roboflow via REST API, queries OpenRouter's Claude 3.5 Sonnet Vision model for AH classification (Anomaly/Hazard), and stores results with ground truth annotations.

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
```bash
export ROBOFLOW_API_KEY="your_roboflow_api_key"
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

You can get your Roboflow API key from: https://app.roboflow.com/settings/api
You can get your OpenRouter API key from: https://openrouter.ai/keys

## Usage

Make sure the virtual environment is activated:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then run the script:
```bash
python classify_roboflow.py \
    --workspace-id "your_workspace_id" \
    --project-id "your_project_id" \
    --output "classification_results.json"
```

### Command-line Arguments

- `--workspace-id` (required): Roboflow workspace ID
- `--project-id` (required): Roboflow project ID
- `--roboflow-api-key`: Roboflow API key (or set `ROBOFLOW_API_KEY` env var)
- `--openrouter-api-key`: OpenRouter API key (or set `OPENROUTER_API_KEY` env var)
- `--output`: Output JSON file path (default: `classification_results.json`)
- `--model`: OpenRouter model to use (default: `anthropic/claude-3.5-sonnet`)

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "results": [
    {
      "image_name": "29_cylinder.png",
      "image_id": "sUpv1FtqlhvWUNxL31ND",
      "ground_truth": "11",
      "model_prediction": "11",
      "model_response": "The image shows...",
      "timestamp": "2025-01-XX...",
      "match": true
    }
  ],
  "errors": [
    {
      "image_name": "failed_image.png",
      "error": "Error message",
      "timestamp": "2025-01-XX..."
    }
  ],
  "summary": {
    "total_images": 100,
    "successful": 98,
    "failed": 2,
    "accuracy": 0.85
  }
}
```

## How It Works

1. **Fetches images from Roboflow**: Uses REST API to get list of images and their metadata
2. **Extracts annotations**: Parses the `hazards-and-anomalies` annotation to get the ground truth 2-digit code (format: AH where A=anomaly, H=hazard)
3. **Downloads images**: Downloads each image from Roboflow
4. **Queries OpenRouter**: Sends image to Claude 3.5 Sonnet Vision model for classification
5. **Stores results**: Saves predictions alongside ground truth with accuracy metrics

## Error Handling

The script includes:
- Retry logic with exponential backoff for API calls
- Error logging for failed classifications
- Graceful handling of missing annotations
- Rate limiting between requests

## Notes

- The script processes images sequentially with a 0.5s delay between requests to avoid rate limits
- Images without annotations are skipped
- Partial results are saved if the script is interrupted (Ctrl+C)
