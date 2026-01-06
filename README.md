# Roboflow to OpenRouter Classification Script

This script fetches images and annotations from Roboflow via REST API, queries OpenRouter's Claude 3.5 Sonnet Vision model for AH classification (Anomaly/Hazard), and stores results with ground truth annotations.

## Project Structure

The project is organized into modular components:

- **`classify_roboflow.py`** - Main entry point script
- **`roboflow_client.py`** - Roboflow API client
- **`openrouter_client.py`** - OpenRouter API client
- **`mock_openrouter_client.py`** - Mock client for testing without API calls
- **`annotation_parser.py`** - Annotation parsing utilities
- **`processor.py`** - Main classification processor
- **`metrics.py`** - Metrics calculation functions (precision, recall, F1)
- **`visualization.py`** - Confusion matrix visualization
- **`analyze_results.py`** - Standalone script for analyzing existing results

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

### Running Classification

Make sure the virtual environment is activated:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Basic usage** (with `.env` file):
```bash
python classify_roboflow.py --output "classification_results.json"
```

**With command-line arguments**:
```bash
python classify_roboflow.py \
    --workspace-id "your_workspace_id" \
    --project-id "your_project_id" \
    --roboflow-api-key "your_roboflow_api_key" \
    --openrouter-api-key "your_openrouter_api_key" \
    --output "classification_results.json" \
    --model "anthropic/claude-3.5-sonnet"
```

**Mock mode** (for testing without API costs):
```bash
# Random results
python classify_roboflow.py --mock-mode

# Reproducible results with seed
python classify_roboflow.py --mock-mode --mock-seed 42
```

### Analyzing Existing Results

You can analyze existing results files without re-running classification:

```bash
python analyze_results.py --input classification_results.json
```

With custom confusion matrix output:
```bash
python analyze_results.py \
    --input classification_results.json \
    --confusion-matrix-output custom_matrix.png
```

### Command-line Arguments

#### classify_roboflow.py

- `--workspace-id`: Roboflow workspace ID (or set `ROBOFLOW_WORKSPACE_ID` in .env file)
- `--project-id`: Roboflow project ID (or set `ROBOFLOW_PROJECT_ID` in .env file)
- `--roboflow-api-key`: Roboflow API key (or set `ROBOFLOW_API_KEY` in .env file)
- `--openrouter-api-key`: OpenRouter API key (or set `OPENROUTER_API_KEY` in .env file) - *Not required in mock mode*
- `--output`: Output JSON file path (default: `classification_results.json`)
- `--model`: OpenRouter model to use (default: `anthropic/claude-3.5-sonnet`)
- `--confusion-matrix-output`: Output path for confusion matrix image (default: `<output_file>_confusion_matrix.png`)
- `--mock-mode`: Enable mock mode - simulates OpenRouter API responses without making actual API calls (saves credits)
- `--mock-seed`: Random seed for mock mode (for reproducible results). Only used when `--mock-mode` is enabled

#### analyze_results.py

- `--input`: Input JSON file path (default: `classification_results.json`)
- `--confusion-matrix-output`: Output path for confusion matrix image (default: `<input_file>_confusion_matrix.png`)

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
      "model_response": "11: Anomalous Dangerous: Both anomaly and hazard detected",
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
    "accuracy": 0.85,
    "precision": 0.8234,
    "recall": 0.8100,
    "f1_score": 0.8166,
    "confusion_matrix_file": "classification_results_confusion_matrix.png"
  }
}
```

The script also generates a confusion matrix visualization saved as a PNG image file.

## Classification Codes

The classification uses a 2-digit code system:

- **00**: Safe - No anomalies or hazards detected
- **01**: Dangerous - Hazard detected but no anomalies
- **10**: Anomalous - Anomaly detected but no hazards
- **11**: Anomalous Dangerous - Both anomaly and hazard detected

## How It Works

1. **Fetches images from Roboflow**: Uses REST API to get list of images and their metadata
2. **Extracts annotations**: Parses annotations to get the ground truth 2-digit code (format: AH where A=anomaly, H=hazard)
3. **Downloads images**: Downloads each image from Roboflow
4. **Queries OpenRouter**: Sends image to Claude 3.5 Sonnet Vision model for classification (or uses mock client in mock mode)
5. **Calculates metrics**: Computes precision, recall, and F1 score using macro-averaging
6. **Generates visualization**: Creates confusion matrix showing ground truth vs predicted classifications
7. **Stores results**: Saves predictions alongside ground truth with comprehensive metrics

## Mock Mode

Mock mode allows you to test the full workflow without making actual API calls:

- **No API costs**: Test without spending credits
- **Fast testing**: No network latency
- **Reproducible**: Optional seed for consistent test results
- **Same interface**: Drop-in replacement, no code changes needed

Mock mode generates random classification codes (00, 01, 10, 11) with uniform distribution. The response format matches the real API, so all downstream processing works identically.

## Error Handling

The script includes:
- Retry logic with exponential backoff for API calls
- Error logging for failed classifications
- Graceful handling of missing annotations
- Rate limiting between requests
- Validation of API keys and configuration

## Notes

- The script processes images sequentially with a 0.5s delay between requests to avoid rate limits
- Images without annotations are skipped
- Partial results are saved if the script is interrupted (Ctrl+C)
- Mock mode still requires Roboflow API access (for fetching image lists and metadata)
- Metrics are calculated using macro-averaging across all classes
- Confusion matrix uses word labels (Neither, Hazard, Anomaly, Both) instead of codes for clarity

## Testing

You can test the full workflow using mock mode:

```bash
# Test with random results
python classify_roboflow.py --mock-mode

# Test with reproducible results
python classify_roboflow.py --mock-mode --mock-seed 42
```

Then analyze the results:
```bash
python analyze_results.py --input classification_results.json
```