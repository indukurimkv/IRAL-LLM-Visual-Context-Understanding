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

3. Set up API keys and configuration:

   **Option 1: Using a .env file (Recommended)**
   
   Create a `.env` file in the project root with the following content:
   ```bash
   ROBOFLOW_API_KEY=your_roboflow_api_key
   ROBOFLOW_WORKSPACE_ID=your_workspace_id
   ROBOFLOW_PROJECT_ID=your_project_id
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```
   
   **Option 2: Using environment variables**
   ```bash
   export ROBOFLOW_API_KEY="your_roboflow_api_key"
   export ROBOFLOW_WORKSPACE_ID="your_workspace_id"
   export ROBOFLOW_PROJECT_ID="your_project_id"
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

If you've set up a `.env` file with all required values, you can simply run:
```bash
python classify_roboflow.py --output "classification_results.json"
```

Or specify values via command-line arguments (overrides .env file):
```bash
python classify_roboflow.py \
    --workspace-id "your_workspace_id" \
    --project-id "your_project_id" \
    --output "classification_results.json"
```

### Command-line Arguments

- `--workspace-id`: Roboflow workspace ID (or set `ROBOFLOW_WORKSPACE_ID` in .env file)
- `--project-id`: Roboflow project ID (or set `ROBOFLOW_PROJECT_ID` in .env file)
- `--roboflow-api-key`: Roboflow API key (or set `ROBOFLOW_API_KEY` in .env file)
- `--openrouter-api-key`: OpenRouter API key (or set `OPENROUTER_API_KEY` in .env file)
- `--output`: Output JSON file path (default: `classification_results.json`)
- `--model`: OpenRouter model to use (default: `anthropic/claude-3.5-sonnet`)

**Note**: All configuration values can be provided via `.env` file, environment variables, or command-line arguments. Command-line arguments take precedence over `.env` file values.

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
