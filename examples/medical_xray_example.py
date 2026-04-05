"""
Medical Image Analysis with RVLM — Chest X-ray Comparison.

Demonstrates RVLM's ability to iteratively analyze and compare medical images.

Usage:
    # Option 1: Use sample X-ray URLs (no download needed)
    python -m examples.medical_xray_example

    # Option 2: Use your own local images
    python -m examples.medical_xray_example --images path/to/scan1.jpg path/to/scan2.jpg

Recommended datasets (free):
    - COVID-19 Radiography: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
    - ChestX-ray14 (NIH): https://nihcc.app.box.com/v/ChestXray-NIHCC
    - RSNA Pneumonia: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from rvlm import RVLM
from rvlm.logger import RLMLogger

load_dotenv()

# Sample chest X-ray URLs from public medical repositories (CC-licensed)
# These are normal and pathological chest X-rays from OpenI (NIH)
SAMPLE_XRAYS = [
    # Normal chest X-ray
    "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
    # Chest X-ray with visible pathology
    "https://upload.wikimedia.org/wikipedia/commons/2/20/Pneumothorax_CXR.jpg",
]


def run_single_scan_analysis(rvlm: RVLM, image_path: str):
    """Analyze a single medical scan in detail."""
    print("\n" + "=" * 80)
    print("SINGLE SCAN ANALYSIS")
    print("=" * 80)

    result = rvlm.completion(
        prompt=(
            "You are a radiology assistant. Analyze this chest X-ray image systematically:\n"
            "1. Assess overall image quality and orientation\n"
            "2. Examine the lung fields (left and right) for any opacities, masses, or abnormalities\n"
            "3. Check the cardiac silhouette size and shape\n"
            "4. Evaluate the mediastinum and hilum\n"
            "5. Check the costophrenic angles\n"
            "6. Look at the bones and soft tissues\n"
            "7. Provide a structured summary of findings"
        ),
        images=[image_path],
        root_prompt="Systematic chest X-ray analysis",
    )

    print(f"\nResult: {result.response[:500]}...")
    print(f"Time: {result.execution_time:.2f}s")
    print(f"Usage: {result.usage_summary.to_dict()}")
    return result


def run_comparison_analysis(rvlm: RVLM, image_paths: list[str]):
    """Compare multiple scans — the core RVLM advantage."""
    print("\n" + "=" * 80)
    print(f"COMPARISON ANALYSIS — {len(image_paths)} images")
    print("=" * 80)

    result = rvlm.completion(
        prompt=(
            "You are a radiology assistant comparing chest X-rays. "
            "These images may be from the same or different patients.\n\n"
            "Perform a systematic comparison:\n"
            "1. First, examine each image individually — describe the key findings\n"
            "2. Compare lung fields across the images — note differences in opacity, "
            "volume, and any new or resolved findings\n"
            "3. Compare cardiac silhouettes — has the heart size changed?\n"
            "4. Check for any interval changes (new findings, resolved findings, "
            "progressing findings)\n"
            "5. Provide a structured comparison report with:\n"
            "   - Findings per image\n"
            "   - Key differences\n"
            "   - Clinical significance of changes"
        ),
        images=image_paths,
        root_prompt="Compare these chest X-rays and identify differences",
    )

    print(f"\nResult:\n{result.response}")
    print(f"\nTime: {result.execution_time:.2f}s")
    print(f"Usage: {result.usage_summary.to_dict()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Medical Image Analysis with RVLM")
    parser.add_argument(
        "--images",
        nargs="+",
        help="Paths or URLs to medical images. Defaults to sample chest X-rays.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "compare", "both"],
        default="compare",
        help="Analysis mode: single scan, comparison, or both.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Max REPL iterations (default: 8, increase for deeper analysis).",
    )
    args = parser.parse_args()

    images = args.images or SAMPLE_XRAYS

    # Validate local file paths exist
    for img in images:
        if not img.startswith(("http://", "https://", "data:")) and not os.path.exists(img):
            print(f"Error: Image file not found: {img}")
            sys.exit(1)

    logger = RLMLogger(log_dir="./logs/medical")

    rvlm = RVLM(
        backend="gemini",
        backend_kwargs={"model_name": "gemini-2.5-flash"},
        environment="local",
        max_depth=1,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=True,
    )

    if args.mode in ("single", "both"):
        run_single_scan_analysis(rvlm, images[0])

    if args.mode in ("compare", "both") and len(images) >= 2:
        run_comparison_analysis(rvlm, images[:2])
    elif args.mode == "compare" and len(images) < 2:
        print("Need at least 2 images for comparison. Running single analysis instead.")
        run_single_scan_analysis(rvlm, images[0])


if __name__ == "__main__":
    main()
