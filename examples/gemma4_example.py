"""
Gemma 4 Local Example — Run Google Gemma 4 locally via HuggingFace transformers.

Demonstrates using the RVLM framework with a local Gemma 4 vision-language model
for recursive image analysis without any API calls.

Prerequisites:
    Install PyTorch for your platform, then torchvision from the same distribution (matching
    CUDA/CPU). Then: pip install transformers accelerate Pillow
    (see https://pytorch.org/get-started/locally/ )

Usage:
    # Basic — uses local BraTS data by default (no network needed)
    python -m examples.gemma4_example --model google/gemma-4-E2B-it

    # Specify a BraTS patient
    python -m examples.gemma4_example --model google/gemma-4-E2B-it --patient BraTS-MEN-00004-000

    # Explicitly use BraTS mode
    python -m examples.gemma4_example --use-brats --patient BraTS-MEN-00010-000

    # Gemma 4 27B MoE (only ~4B active params, needs ~16 GB VRAM)
    python -m examples.gemma4_example

    # Smaller variants
    python -m examples.gemma4_example --model google/gemma-4-E2B-it   # 5B, ~10 GB VRAM
    python -m examples.gemma4_example --model google/gemma-4-E4B-it   # 8B, ~16 GB VRAM

    # Largest dense model (needs ~64 GB VRAM)
    python -m examples.gemma4_example --model google/gemma-4-31B-it

    # Custom image
    python -m examples.gemma4_example --image path/to/image.jpg

    # Text-only mode (disable vision)
    python -m examples.gemma4_example --no-vision --model google/gemma-4-26B-A4B-it

    # Adjust generation length
    python -m examples.gemma4_example --max-new-tokens 4096

    # With BraTS brain tumor data
    python -m examples.gemma4_example --image data/BraTS-MEN-Train/BraTS-MEN-00004-000/BraTS-MEN-00004-000-t1c.nii.gz

    # MIMIC chest X-ray
    python -m examples.gemma4_example --use-mimic --mimic-subject 10000032

    # If behind a corporate proxy / self-signed cert:
    python -m examples.gemma4_example --disable-ssl
    # Or:  HF_HUB_DISABLE_SSL=1 python -m examples.gemma4_example
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Prefer local data that doesn't require network access.
# Falls back to a Wikipedia URL only when no BraTS data is present.
DATA_DIR = Path(__file__).parent.parent / "data" / "BraTS-MEN-Train"
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/20/Pneumothorax_CXR.jpg"


# ------------------------------------------------------------------ #
#  BraTS NIfTI → PNG helpers (mirrors brats_example.py)               #
# ------------------------------------------------------------------ #


def _first_patient_dir() -> Path | None:
    """Return the first available BraTS patient directory, or None."""
    if not DATA_DIR.exists():
        return None
    dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    return dirs[0] if dirs else None


def _nifti_to_png(nifti_path: str, slice_idx: int, output_path: str | None = None) -> str:
    """Convert one axial slice of a NIfTI volume to a grayscale PNG."""
    import nibabel as nib
    import numpy as np
    from PIL import Image as PILImage

    data = np.asarray(nib.load(nifti_path).dataobj, dtype=np.float32)
    slice_idx = min(max(0, slice_idx), data.shape[2] - 1)
    sl = np.rot90(data[:, :, slice_idx])

    vmin, vmax = np.percentile(sl, (1, 99))
    sl = np.clip((sl - vmin) / (vmax - vmin) * 255, 0, 255) if vmax > vmin else np.zeros_like(sl)

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")
    PILImage.fromarray(sl.astype(np.uint8), mode="L").save(output_path)
    return output_path


def _get_peak_tumor_slice(seg_path: str) -> int:
    """Return the axial slice with the largest tumour cross-section."""
    import nibabel as nib
    import numpy as np

    mask = np.asarray(nib.load(seg_path).dataobj, dtype=np.uint8)
    per_slice = (mask > 0).sum(axis=(0, 1))
    return int(np.argmax(per_slice))


def prepare_brats_images(patient_id: str | None = None) -> tuple[list[str], dict[str, Any] | None]:
    """Convert BraTS NIfTI data to PNGs and return the list of image paths.

    Uses the same slice-selection logic as brats_example.py:
    peak tumour slice from the seg mask, or the middle slice as fallback.

    Returns:
        (png_paths, mask_stats)  — mask_stats is None when no seg mask exists.
    """
    patient_dir: Path | None
    if patient_id:
        patient_dir = DATA_DIR / patient_id
        if not patient_dir.exists():
            print(f"⚠  Patient '{patient_id}' not found in {DATA_DIR}")
            patient_dir = _first_patient_dir()
    else:
        patient_dir = _first_patient_dir()

    if patient_dir is None:
        return [], None

    pid = patient_dir.name
    print(f"📂 Using BraTS patient: {pid}")

    # Determine slice index
    seg_path = patient_dir / f"{pid}-seg.nii.gz"
    t1c_path = patient_dir / f"{pid}-t1c.nii.gz"
    mask_stats: dict[str, Any] | None = None

    if seg_path.exists():
        slice_idx = _get_peak_tumor_slice(str(seg_path))
        print(f"  Slice: {slice_idx} (peak tumour from seg mask)")
        # Compute mask stats for the router (same as brats_example.compute_mask_stats)
        try:
            from examples.brats_example import compute_mask_stats

            mask_stats = compute_mask_stats(str(seg_path))
            mask_stats["slice_idx"] = slice_idx
        except ImportError:
            pass
    elif t1c_path.exists():
        import nibabel as nib

        slice_idx = nib.load(str(t1c_path)).shape[2] // 2
        print(f"  Slice: {slice_idx} (middle slice fallback)")
    else:
        print("  ⚠  No NIfTI volumes found in patient directory")
        return [], None

    # Convert each available modality to PNG
    tmp_dir = tempfile.mkdtemp(prefix="gemma4_brats_")
    modalities = ["t1n", "t1c", "t2w", "t2f"]
    png_paths: list[str] = []

    for mod in modalities:
        nifti = patient_dir / f"{pid}-{mod}.nii.gz"
        if not nifti.exists():
            continue
        png = os.path.join(tmp_dir, f"{pid}-{mod}.png")
        _nifti_to_png(str(nifti), slice_idx, output_path=png)
        png_paths.append(png)
        print(f"  {mod} → {png}")

    return png_paths, mask_stats


def print_section(title: str, content: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(content)


def validate_image_sources(images: list[str]) -> None:
    for img in images:
        if img.startswith(("http://", "https://", "data:")):
            continue
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")


def build_backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.dtype,
        "device_map": args.device_map,
        "hf_token": args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        "vision": not args.no_vision,
        "enable_cache": not args.no_cache,
        "prefill_chunk_size": args.prefill_chunk_size,
    }


# ------------------------------------------------------------------ #
#  Single-pass (direct client call, no REPL)                          #
# ------------------------------------------------------------------ #


def run_single_pass(args: argparse.Namespace, images: list[str]) -> str:
    """Run a single-pass analysis using the raw HFLocalClient."""
    from rvlm.clients.hf_local import HFLocalClient

    print("\n⏳ Loading model locally (this may take a minute)...")
    client = HFLocalClient(**build_backend_kwargs(args))

    prompt_text = (
        "Analyze this image and provide:\n"
        "1) A detailed description of what you see.\n"
        "2) Any notable findings or abnormalities.\n"
        "3) Your overall assessment."
    )

    # Build an OpenAI-style multimodal prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for img_src in images:
        if img_src.startswith(("http://", "https://")):
            content.append({"type": "image_url", "image_url": {"url": img_src, "detail": "auto"}})
        else:
            import base64

            with open(img_src, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
                }
            )

    messages = [{"role": "user", "content": content}]

    print("⏳ Running single-pass inference...")
    start = time.perf_counter()
    response = client.completion(messages)
    elapsed = time.perf_counter() - start

    print_section("SINGLE-PASS OUTPUT", response)
    print(f"\n⏱  Time: {elapsed:.2f}s")
    usage = client.get_last_usage()
    print(f"📊 Tokens: {usage.total_input_tokens} in / {usage.total_output_tokens} out")
    cache_stats = client.get_cache_stats()
    if cache_stats:
        print(f"🗄  Cache: {cache_stats}")
    return response


# ------------------------------------------------------------------ #
#  RVLM recursive mode                                                #
# ------------------------------------------------------------------ #


def run_rvlm_recursive(
    args: argparse.Namespace,
    images: list[str],
    mask_stats: dict[str, Any] | None = None,
) -> str:
    """Run RVLM recursive analysis with the local Gemma 4 backend."""
    from rvlm import RVLM
    from rvlm.logger import RLMLogger
    from rvlm.router import RecursionRouter

    logger = RLMLogger(log_dir="./logs/gemma4")

    # Build a router.  When BraTS mask_stats are available (passed via param),
    # the router adapts iterations to tumour complexity; otherwise it uses a
    # neutral prior (complexity=0.5 → 5 iterations recommended).
    router = RecursionRouter.from_mask_stats(mask_stats, verbose=True)
    print(
        f"[Router] complexity={router.complexity_score:.2f}"
        f"  recommended_iters={router.recommended_max_iterations()}"
    )

    print("\n⏳ Initialising RVLM with local Gemma 4 (model loading may take a minute)...")
    rvlm = RVLM(
        backend="hf_local",
        backend_kwargs=build_backend_kwargs(args),
        environment="local",
        max_depth=1,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=True,
        router=router,
    )

    prompt = (
        "Analyze this image thoroughly. Describe what you see, identify any notable "
        "findings or abnormalities, and provide your overall assessment."
    )

    print("⏳ Running RVLM recursive analysis...")
    start = time.perf_counter()
    result = rvlm.completion(
        prompt=prompt,
        images=images,
        root_prompt="Gemma 4 local recursive image analysis",
    )
    elapsed = time.perf_counter() - start

    print_section("RVLM RECURSIVE OUTPUT", result.response)
    print(f"\n⏱  Total time: {elapsed:.2f}s")
    print(f"📊 Usage: {result.usage_summary.to_dict()}")

    return result


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 locally for image/text analysis via RVLM."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image source: URL, local path, or data URI.  "
        "If omitted, uses local BraTS data when available.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Multiple image sources (overrides --image).",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-26B-A4B-it",
        help="HuggingFace model ID or local path (default: google/gemma-4-26B-A4B-it). "
        "Options: google/gemma-4-E2B-it (5B), google/gemma-4-E4B-it (8B), "
        "google/gemma-4-26B-A4B-it (27B MoE), google/gemma-4-31B-it (33B).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map strategy (auto, cpu, cuda:0, etc.).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated models.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision (text-only mode).",
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "rvlm", "both"],
        help="Run mode: 'single' (direct), 'rvlm' (recursive), or 'both'.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="RVLM max REPL iterations (only for rvlm/both mode).",
    )
    parser.add_argument(
        "--use-mimic",
        action="store_true",
        help="Load images from local MIMIC-CXR data.",
    )
    parser.add_argument(
        "--mimic-subject",
        type=int,
        default=None,
        help="MIMIC subject_id to use with --use-mimic.",
    )
    parser.add_argument(
        "--use-brats",
        action="store_true",
        help="Load images from local BraTS brain tumour data (NIfTI → PNG).",
    )
    parser.add_argument(
        "--patient",
        default=None,
        help="BraTS patient ID to use with --use-brats (e.g. BraTS-MEN-00004-000). "
        "If omitted, uses the first available patient.",
    )
    parser.add_argument(
        "--disable-ssl",
        action="store_true",
        help="Disable SSL verification for HuggingFace Hub downloads "
        "(use when behind a corporate proxy or self-signed cert).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a formal LaTeX/PDF report after the RVLM analysis.",
    )
    parser.add_argument(
        "--report-dir",
        default="./reports",
        help="Directory in which to save the PDF report (default: ./reports).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the inference cache (vision feature cache, KV-prefix "
        "reuse, and chunked prefill).  Cache is ON by default.",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Max tokens per chunk during prefill (default: 512). "
        "Lower values reduce peak GPU memory; higher values are faster.",
    )
    args = parser.parse_args()

    # Apply SSL workaround early, before any HF Hub imports
    if args.disable_ssl:
        os.environ["HF_HUB_DISABLE_SSL"] = "1"

    # Resolve images — prefer local data sources (no network required)
    images: list[str]
    mask_stats: dict[str, Any] | None = None
    if args.use_mimic:
        try:
            from examples.mimic_example import get_patient_row, load_csv, select_images
        except ImportError as exc:
            print(f"Error: MIMIC mode requires mimic_example dependencies: {exc}")
            sys.exit(1)

        df = load_csv("validate")
        subject_id = args.mimic_subject or int(df.iloc[0]["subject_id"])
        row = get_patient_row(df, subject_id)
        images, view_types, _ = select_images(row, study_index=-1)
        print(f"Loaded MIMIC subject {subject_id}: {len(images)} image(s), views={view_types}")
    elif args.use_brats or (args.image is None and args.images is None):
        # Default: use local BraTS data (avoids network issues on HPC)
        images, mask_stats = prepare_brats_images(patient_id=args.patient)
        if not images:
            if args.use_brats:
                print(f"Error: No BraTS data found in {DATA_DIR}")
                sys.exit(1)
            # Ultimate fallback: remote sample image (may fail on restricted networks)
            print("⚠  No local BraTS data found — falling back to remote sample image")
            images = [SAMPLE_IMAGE_URL]
    elif args.images:
        images = args.images
    else:
        images = [args.image]

    try:
        validate_image_sources(images)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"🔧 Model: {args.model}")
    print(f"🖼  Images: {len(images)}")
    print(f"🎯 Mode: {args.mode}")
    print(f"📏 Max tokens: {args.max_new_tokens}")

    if args.mode in ("single", "both"):
        run_single_pass(args, images)

    if args.mode in ("rvlm", "both"):
        result = run_rvlm_recursive(args, images, mask_stats=mask_stats)

        # Generate a formal LaTeX/PDF report if requested
        if args.report and result is not None:
            from examples.latex_report import LatexReportGenerator

            patient_id = args.patient or "gemma4-analysis"
            gen = LatexReportGenerator(
                backend="hf_local",
                backend_kwargs=build_backend_kwargs(args),
            )
            pdf_path = gen.generate(
                result=result,
                patient_id=patient_id,
                mask_stats=mask_stats,
                output_dir=args.report_dir,
            )
            print(f"\n📄 Clinical report: {pdf_path}")


if __name__ == "__main__":
    main()
