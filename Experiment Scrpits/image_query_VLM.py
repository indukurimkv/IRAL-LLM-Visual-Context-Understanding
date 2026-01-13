import json
from pathlib import Path
from openai import OpenAI
from roboflow import Roboflow
from datetime import datetime

# --- Download dataset in openai format ---
rf = Roboflow(api_key="mHudEyXIfnCbIykoNIcm")
project = rf.workspace("vlm-in-context-anomaly-and-hazard-detection").project("vlm-in-context-anom-haz-annot-1p23m")
version = project.version(1)
dataset = version.download("openai")

# Locate the JSON file inside the exported folder
dataset_dir = Path(dataset.location)
json_files = list(dataset_dir.glob("*.json"))

if not json_files:
    raise FileNotFoundError(f"No JSON file found in {dataset_dir}")
json_path = json_files[0]  # usually dataset.json or data.json

print(f"Loaded dataset JSON: {json_path}")

with open(json_path, "r") as f:
    data = json.load(f)

# --- OpenAI setup ---
client = OpenAI(api_key="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")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = Path(f"roboflow_hazard_results_{timestamp}.txt")

images = data.get("images", [])
print(f"Found {len(images)} image entries.\n")

for item in images[:5]:  # test on first 5
    img_url = item.get("url")
    print(f"Analyzing remote image: {img_url}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are inspecting an environment for anomalies and hazards. "
                    "Use these OSHA definitions:\n"
                    "Hazard: Anything that could cause immediate harm to people or damage property.\n"
                    "Anomaly: Anything inconsistent with common-sense reasoning or contextual expectations.\n"
                    "Classify the image based on these definitions."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Report the hazard binary code of this image with a one-sentence explanation: Safe 00, Anomalous 10, Dangerous 01, or Anomalous Dangerous 11."
                    },
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            },
        ],
    )

    result = response.choices[0].message.content.strip()
    print("→", result)
    with open(output_file, "a") as f:
        f.write(f"{img_url}: {result}\n")

print(f"\n✅ Results saved to: {output_file}")
