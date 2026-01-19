#!/usr/bin/env python3
"""Quick test for pdf_report.generate_pdf_report."""

import io
import os
from PIL import Image
from pdf_report import generate_pdf_report


def make_image_bytes(color=(200, 220, 240), size=(640, 480)):
    img = Image.new('RGB', size, color=color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def main():
    items = [
        {
            "image_id": "img001",
            "image_name": "image1.png",
            "image_data": make_image_bytes(color=(240, 200, 200)),
            "ground_truth": "11",
            "model_predictions": {
                "anthropic/claude-3.5-sonnet": {"code": "11", "response": "11: Both anomaly and hazard detected"},
                "openai/gpt-4.1-mini": {"code": "10", "response": "10: Anomaly detected, no hazard"}
            }
        },
        {
            "image_id": "img002",
            "image_name": "image2.png",
            "image_data": make_image_bytes(color=(200, 240, 200)),
            "ground_truth": "00",
            "model_predictions": {
                "anthropic/claude-3.5-sonnet": {"code": "00", "response": "00: Safe"},
                "openai/gpt-4.1-mini": {"code": "01", "response": "01: Hazard detected"}
            }
        }
    ]

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "classification_report_TEST.pdf")
    res = generate_pdf_report(items, out_path)
    print(f"PDF generated: {res}")


if __name__ == "__main__":
    main()
