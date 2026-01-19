"""PDF report generation for classification results across models.

Generates a multi-page PDF where each page contains:
- A page number and image name at the top
- The image centered
- Ground truth annotation
- Predicted hazard codes (and optional responses) for each model
"""

import io
from typing import Dict, List, Optional
import textwrap

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _render_page(fig, image: Image.Image, title: str, annotation: str, model_entries: List[str]):
    """Render a single page with image and text sections.

    Uses a dedicated text axes with wrap enabled so long lines are
    automatically wrapped to the available width.
    """
    fig.suptitle(title, fontsize=14)

    # Image area
    ax_img = fig.add_axes([0.05, 0.20, 0.90, 0.65])  # left, bottom, width, height
    ax_img.imshow(image)
    ax_img.axis('off')

    # Text area (wrap within this axes)
    ax_text = fig.add_axes([0.05, 0.05, 0.90, 0.15])
    ax_text.axis('off')

    lines = [f"Annotation (ground truth): {annotation}"]
    if model_entries:
        lines.append("Model Predictions:")
        # Wrap each entry to a reasonable line width for readability
        # The actual wrap respects the axes width via wrap=True, but we
        # pre-wrap to avoid overly long single lines.
        for entry in model_entries:
            wrapped = textwrap.fill(entry, width=100)
            lines.append(wrapped)

    text_block = "\n".join(lines)
    ax_text.text(0.0, 0.95, text_block, transform=ax_text.transAxes,
                 va='top', ha='left', fontsize=10, wrap=True)


def generate_pdf_report(items: List[Dict], output_path: str):
    """Generate a PDF report summarizing all images and models.

    Each item in `items` should be a dict with keys:
        - image_id: str
        - image_name: str
        - image_data: bytes
        - ground_truth: str
        - model_predictions: Dict[str, Dict] where value has keys:
            - code: str (predicted code)
            - response: Optional[str] (full model response text)

    Args:
        items: list of aggregated image entries
        output_path: file path to write the PDF
    """
    if not items:
        return None

    with PdfPages(output_path) as pdf:
        for idx, it in enumerate(items, 1):
            # Load image from bytes
            try:
                img = Image.open(io.BytesIO(it["image_data"]))
            except Exception:
                # Fallback: create a placeholder image
                img = Image.new('RGB', (800, 600), color=(240, 240, 240))

            # Prepare text entries per model
            model_entries = []
            preds: Dict[str, Dict] = it.get("model_predictions", {})
            for model_name, pdata in preds.items():
                code = pdata.get("code", "N/A")
                response = pdata.get("response")
                if response:
                    # Truncate extremely long responses to keep layout tidy
                    short_resp = (response[:600] + "...") if len(response) > 600 else response
                    model_entries.append(f"- {model_name}: {code} | {short_resp}")
                else:
                    model_entries.append(f"- {model_name}: {code}")

            # Create figure page
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
            title = f"Image {idx}: {it.get('image_name', it.get('image_id', 'unknown'))}"
            annotation = it.get("ground_truth", "N/A")
            _render_page(fig, img, title, annotation, model_entries)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    return output_path
