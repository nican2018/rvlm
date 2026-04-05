"""
BraTS 2023 Meningioma Sub-region Characterization with RVLM.

Problem statement
-----------------
Given 4 MRI modalities and ground truth segmentation masks, can RVLM's
recursive reasoning produce a more accurate characterization of meningioma
sub-regions (NCR / ED / ET) than a single-pass VLM call — evaluated against
the mask?

Why BraTS-MEN is a segmentation dataset, not a detection dataset
----------------------------------------------------------------
Every patient in BraTS-MEN has a confirmed meningioma. Asking "is there a
tumour?" is trivially true for all cases. The dataset's value is the *where*
and *what*: the segmentation mask labels three distinct tumour sub-regions with
exact voxel-level ground truth:

    Label 1 → NCR  Necrotic Core        (dark on T1c, non-enhancing)
    Label 2 → ED   Peritumoral Edema    (bright on T2w / FLAIR)
    Label 3 → ET   Enhancing Tumour     (bright on T1c, NOT bright on T1n)

The real task is sub-region characterization and volume estimation — not
detection.

The RVLM advantage (the loop a single VLM call cannot do)
----------------------------------------------------------
    Iteration 1  REPL calls describe_image() on each of the 4 modalities
                 independently to identify signal anomalies per modality.

    Iteration 2  REPL calls llm_query_with_images() on the colour overlay to
                 interpret what each coloured region looks like on T1c.

    Iteration 3  REPL cross-references T1c + T2-FLAIR to confirm that the
                 enhancing region matches the ET overlay and the oedema zone
                 matches the ED overlay.

    Final        Structured report: sub-region descriptions + visual volume
                 estimates compared against the ground truth mask statistics.

Usage
-----
    # Analyse first patient (slice auto-selected from seg mask)
    python -m examples.brats_example

    # Specify a patient
    python -m examples.brats_example --patient BraTS-MEN-00004-000

    # Compare two patients (each at their own peak-tumour slice)
    python -m examples.brats_example --patient BraTS-MEN-00004-000 --patient2 BraTS-MEN-00008-000

    # Override the slice index
    python -m examples.brats_example --slice 80

Requires
--------
    pip install nibabel Pillow
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from rvlm import RVLM
from rvlm.logger import RLMLogger
from rvlm.router import RecursionRouter
from examples.latex_report import LatexReportGenerator

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "BraTS-MEN-Train"

# BraTS 2023 segmentation label definitions: label → (short name, full name, RGB colour)
SEG_LABELS: dict[int, tuple[str, str, tuple[int, int, int]]] = {
    1: ("NCR", "Necrotic Core",      (220,  50,  50)),   # red
    2: ("ED",  "Peritumoral Edema",  (255, 200,   0)),   # yellow
    3: ("ET",  "Enhancing Tumour",   (  0, 210, 100)),   # green
}
# Some BraTS releases encode ET as label 4; treat it identically to 3.
_ET_LABELS = {3, 4}

MRI_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
MRI_LABELS = {
    "t1n": "T1-weighted native",
    "t1c": "T1-weighted contrast-enhanced",
    "t2w": "T2-weighted",
    "t2f": "T2-FLAIR",
}


# ---------------------------------------------------------------------------
# Segmentation mask utilities
# ---------------------------------------------------------------------------

def get_signal_peak_slice(t1c_path: str, t1n_path: str) -> int:
    """Estimate the slice with the most tumour enhancement when no seg mask exists.

    Computes the per-slice mean of (normalised T1c − normalised T1n), clipped to
    positive values.  The slice with the highest mean difference is where contrast
    enhancement is most prominent — a physics-grounded proxy for tumour location
    that requires only the raw MRI volumes.
    """
    t1c = np.asarray(nib.load(t1c_path).dataobj, dtype=np.float32)
    t1n = np.asarray(nib.load(t1n_path).dataobj, dtype=np.float32)

    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-8)

    diff = (_norm(t1c) - _norm(t1n)).clip(0, None)   # enhancement = T1c brighter than T1n
    per_slice = diff.mean(axis=(0, 1))
    return int(np.argmax(per_slice))


def get_peak_tumor_slice(seg_path: str) -> int:
    """Return the axial slice with the largest tumour cross-section.

    Instead of the anatomical middle slice (which may not contain tumour),
    this finds the slice where the segmentation mask has the most labelled
    voxels — guaranteeing RVLM always sees the most informative view.
    """
    mask = np.asarray(nib.load(seg_path).dataobj, dtype=np.uint8)
    per_slice = (mask > 0).sum(axis=(0, 1))   # sum over X, Y → one count per Z
    return int(np.argmax(per_slice))


def compute_mask_stats(seg_path: str) -> dict:
    """Extract quantitative ground truth from the segmentation mask.

    Returns per-sub-region voxel counts and volumes in cc, whole-tumour
    centroid as a percentage of each image dimension, and the dominant
    hemisphere (left / right) and anterior-posterior position.
    """
    mask_img = nib.load(seg_path)
    mask = np.asarray(mask_img.dataobj, dtype=np.uint8)

    # Physical voxel volume in mm³ from the affine diagonal
    voxel_mm3 = float(np.prod(np.abs(np.diag(mask_img.affine[:3, :3]))))
    H, W, D = mask.shape

    stats: dict = {
        "total_slices": D,
        "voxel_mm3": round(voxel_mm3, 4),
    }

    total_voxels = 0
    for label, (short, _name, _color) in SEG_LABELS.items():
        region = (mask == label)
        if label == 3:                       # merge label 4 into ET
            region = region | (mask == 4)
        voxels = int(region.sum())
        total_voxels += voxels
        stats[f"{short}_voxels"] = voxels
        stats[f"{short}_volume_cc"] = round(voxels * voxel_mm3 / 1000, 2)

    stats["total_tumor_voxels"] = total_voxels
    stats["total_tumor_volume_cc"] = round(total_voxels * voxel_mm3 / 1000, 2)

    # Whole-tumour centroid → hemisphere and A/P labels
    tumor_mask = mask > 0
    if tumor_mask.any():
        centroid = np.argwhere(tumor_mask).astype(float).mean(axis=0)
        stats["centroid_x_pct"] = round(float(centroid[0]) / H * 100, 1)
        stats["centroid_y_pct"] = round(float(centroid[1]) / W * 100, 1)
        stats["centroid_z_pct"] = round(float(centroid[2]) / D * 100, 1)
        # BraTS LPS orientation: lower X index = left hemisphere
        stats["hemisphere"] = "left" if centroid[0] < H / 2 else "right"
        # Lower Y index = anterior
        stats["position_ap"] = "anterior" if centroid[1] < W / 2 else "posterior"

    return stats


def mask_to_overlay_png(
    mri_path: str,
    seg_path: str,
    slice_idx: int,
    output_path: str | None = None,
    alpha: float = 0.45,
) -> str:
    """Render an MRI slice with a colour-coded segmentation overlay.

    Colour coding:
        Red    = NCR  (Necrotic Core,     label 1)
        Yellow = ED   (Peritumoral Edema, label 2)
        Green  = ET   (Enhancing Tumour,  label 3 / 4)

    Args:
        mri_path:    MRI NIfTI to use as the base image (T1c recommended).
        seg_path:    Segmentation NIfTI containing BraTS labels.
        slice_idx:   Axial slice to render (use the peak-tumour slice).
        output_path: Destination PNG path.  Defaults to a temp file.
        alpha:       Overlay opacity  (0 = invisible, 1 = fully opaque).

    Returns:
        Path to the saved PNG.
    """
    # Load and normalise the MRI slice
    mri_data = np.asarray(nib.load(mri_path).dataobj, dtype=np.float32)
    slice_idx = min(max(0, slice_idx), mri_data.shape[2] - 1)
    mri_slice = np.rot90(mri_data[:, :, slice_idx])

    vmin, vmax = np.percentile(mri_slice, (1, 99))
    if vmax > vmin:
        mri_norm = np.clip((mri_slice - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    else:
        mri_norm = np.zeros_like(mri_slice, dtype=np.uint8)

    # Load the segmentation slice (same rotation)
    seg_data = np.asarray(nib.load(seg_path).dataobj, dtype=np.uint8)
    seg_slice = np.rot90(seg_data[:, :, slice_idx])

    # Start with a grayscale-RGB base
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1).astype(np.float32)

    # Blend each sub-region with its assigned colour
    for label, (_short, _name, color) in SEG_LABELS.items():
        region = (seg_slice == label)
        if label == 3:
            region = region | (seg_slice == 4)
        if not region.any():
            continue
        for c, v in enumerate(color):
            rgb[:, :, c] = np.where(
                region,
                rgb[:, :, c] * (1 - alpha) + v * alpha,
                rgb[:, :, c],
            )

    pil_img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), mode="RGB")

    if output_path is None:
        output_path = tempfile.mktemp(suffix="_overlay.png")

    pil_img.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# NIfTI → PNG (single modality)
# ---------------------------------------------------------------------------

def nifti_to_png(nifti_path: str, slice_idx: int, output_path: str | None = None) -> str:
    """Convert one axial slice of a NIfTI volume to a grayscale PNG."""
    data = np.asarray(nib.load(nifti_path).dataobj, dtype=np.float32)
    slice_idx = min(max(0, slice_idx), data.shape[2] - 1)
    sl = np.rot90(data[:, :, slice_idx])

    vmin, vmax = np.percentile(sl, (1, 99))
    sl = np.clip((sl - vmin) / (vmax - vmin) * 255, 0, 255) if vmax > vmin else np.zeros_like(sl)

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")

    Image.fromarray(sl.astype(np.uint8), mode="L").save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Patient directory helpers
# ---------------------------------------------------------------------------

def get_patient_dir(patient_id: str) -> Path:
    """Return the patient directory, falling back to the first available."""
    patient_dir = DATA_DIR / patient_id
    if patient_dir.exists():
        return patient_dir

    available = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    if not available:
        print(f"Error: No patient folders found in {DATA_DIR}")
        sys.exit(1)

    print(f"Patient '{patient_id}' not found. Available: {available[:5]}...")
    print(f"Using first patient: {available[0]}")
    return DATA_DIR / available[0]


def prepare_patient_images(
    patient_dir: Path,
    slice_idx: int | None = None,
    output_dir: str | None = None,
) -> tuple[dict[str, str], dict, int]:
    """Extract PNG images for all modalities plus the colour overlay.

    Slice selection priority
    ~~~~~~~~~~~~~~~~~~~~~~~~
    1. Explicit ``--slice`` argument (user override).
    2. Peak-tumour slice auto-detected from the segmentation mask.
    3. Middle slice fallback when no mask is present.

    Returns
    ~~~~~~~
    slices      Dict mapping key → PNG path.
                Keys: t1n, t1c, t2w, t2f, overlay (when mask exists).
    mask_stats  Ground truth statistics extracted from the seg mask
                (empty dict when mask is absent).
    slice_idx   The axial slice index that was used.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="brats_")

    patient_id = patient_dir.name
    seg_path = patient_dir / f"{patient_id}-seg.nii.gz"
    mask_stats: dict = {}

    # ---- Determine slice ----
    if slice_idx is not None:
        print(f"  Slice: {slice_idx} (user override)")
    elif seg_path.exists():
        slice_idx = get_peak_tumor_slice(str(seg_path))
        print(f"  Slice: {slice_idx} (peak tumour cross-section from seg mask)")
    else:
        # No seg mask — use signal-based proxy: max T1c-T1n enhancement difference
        t1c_file = patient_dir / f"{patient_id}-t1c.nii.gz"
        t1n_file = patient_dir / f"{patient_id}-t1n.nii.gz"
        if t1c_file.exists() and t1n_file.exists():
            slice_idx = get_signal_peak_slice(str(t1c_file), str(t1n_file))
            print(f"  Slice: {slice_idx} (peak T1c-T1n enhancement signal — no seg mask)")
        elif t1c_file.exists():
            probe = nib.load(str(t1c_file))
            slice_idx = probe.shape[2] // 2
            print(f"  Slice: {slice_idx} (middle slice fallback — T1n also missing)")
        else:
            slice_idx = 80
            print(f"  Slice: {slice_idx} (hard fallback — T1c not found)")

    # ---- Mask statistics ----
    if seg_path.exists():
        mask_stats = compute_mask_stats(str(seg_path))
        mask_stats["slice_idx"] = slice_idx
    else:
        print("  Warning: segmentation mask not found — ground truth unavailable")

    # ---- MRI modality PNGs ----
    slices: dict[str, str] = {}
    for suffix in MRI_MODALITIES:
        nifti_file = patient_dir / f"{patient_id}-{suffix}.nii.gz"
        if not nifti_file.exists():
            print(f"  Skipping {suffix}: file not found")
            continue
        png_path = os.path.join(output_dir, f"{patient_id}-{suffix}.png")
        nifti_to_png(str(nifti_file), slice_idx=slice_idx, output_path=png_path)
        slices[suffix] = png_path
        print(f"  {suffix} -> {png_path}")

    # ---- Colour overlay (T1c base + seg mask) ----
    if seg_path.exists() and "t1c" in slices:
        overlay_path = os.path.join(output_dir, f"{patient_id}-overlay.png")
        mask_to_overlay_png(
            mri_path=str(patient_dir / f"{patient_id}-t1c.nii.gz"),
            seg_path=str(seg_path),
            slice_idx=slice_idx,
            output_path=overlay_path,
        )
        slices["overlay"] = overlay_path
        print(f"  overlay -> {overlay_path}")

    return slices, mask_stats, slice_idx


# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------

def _format_mask_stats(stats: dict) -> str:
    """Format mask statistics into the ground truth block used in prompts."""
    if not stats:
        return "  (no segmentation mask available)"
    return (
        f"  Slice shown:          {stats.get('slice_idx', '?')} / "
        f"{stats.get('total_slices', '?')} axial slices "
        f"(peak tumour cross-section)\n"
        f"  Total tumour volume:  {stats.get('total_tumor_volume_cc', '?')} cc\n"
        f"    NCR (necrotic core):     {stats.get('NCR_volume_cc', 0):.2f} cc\n"
        f"    ED  (peritumoral oedema):{stats.get('ED_volume_cc', 0):.2f} cc\n"
        f"    ET  (enhancing tumour):  {stats.get('ET_volume_cc', 0):.2f} cc\n"
        f"  Centroid hemisphere: {stats.get('hemisphere', '?')}\n"
        f"  Centroid A/P:        {stats.get('position_ap', '?')}"
    )


# ---------------------------------------------------------------------------
# RVLM analysis functions
# ---------------------------------------------------------------------------

def run_segmentation_grounded_analysis(
    rvlm: RVLM,
    slices: dict[str, str],
    patient_id: str,
    mask_stats: dict,
):
    """Primary analysis: sub-region characterization grounded in the seg mask.

    This replaces the old "detect meningioma" framing.  The tumour is already
    confirmed present; the task is to characterize *what* the sub-regions look
    like across modalities and compare visual estimates against mask-derived
    ground truth — work that benefits from RVLM's iterative REPL loop.
    """
    # Image order: 4 modalities first, overlay last (most interpretable when seen after raw)
    image_order = [k for k in [*MRI_MODALITIES, "overlay"] if k in slices]
    image_paths = [slices[k] for k in image_order]

    image_index_labels = {
        "t1n":    "T1 native (no contrast)",
        "t1c":    "T1 contrast-enhanced",
        "t2w":    "T2-weighted",
        "t2f":    "T2-FLAIR",
        "overlay": "T1c + segmentation overlay  [Red=NCR  Yellow=ED  Green=ET]",
    }
    images_block = "\n".join(
        f"  Image {i}: {image_index_labels[k]}"
        for i, k in enumerate(image_order)
    )
    gt_block = _format_mask_stats(mask_stats)
    overlay_idx = image_order.index("overlay") if "overlay" in image_order else None
    has_gt = bool(mask_stats)

    # ---- Build prompt variables for GT vs. inference modes ----
    overlay_var = "overlay_q"   if has_gt else "enhancement_q"
    cross_var   = "cross_q"     if has_gt else "surround_q"

    if has_gt:
        ncr_cc = mask_stats.get('NCR_volume_cc', 0)
        ed_cc  = mask_stats.get('ED_volume_cc',  0)
        et_cc  = mask_stats.get('ET_volume_cc',  0)
        gt_header = (
            "GROUND TRUTH  (from segmentation mask — verify findings against this)\n"
            f"{gt_block}"
        )
        step2_a_code = (
            f"  {overlay_var} = llm_query_with_images("
            f"'GREEN (ET): LEFT or RIGHT side? How large? Any RED (NCR) or YELLOW (ED)?', [{overlay_idx}])"
        )
        step2_b_code = (
            f"  {cross_var} = llm_query_with_images("
            f"'T1c bright mass: LEFT or RIGHT side? FLAIR oedema ring? Midline shift?', [1, 3])"
        )
        gt_context_for_synthesis = (
            f"NCR={ncr_cc:.2f}cc  ED={ed_cc:.2f}cc  ET={et_cc:.2f}cc  "
            f"hemisphere={mask_stats.get('hemisphere', '?')}  {mask_stats.get('position_ap', '?')}"
        )
        subregions_for_synthesis = (
            f"NCR ({ncr_cc:.2f}cc GT), ED ({ed_cc:.2f}cc GT), ET ({et_cc:.2f}cc GT)"
        )
        overlay_label = "Colour overlay (GT seg)"
        cross_label   = "T1c + T2-FLAIR cross-check"
        final_section_label   = "AGREEMENT"
        final_section_content = "visual observations vs GT volumes — match or discrepancy"
    else:
        gt_header = (
            "MODE: Inference (no segmentation mask)\n"
            "  Slice selected using peak T1c-T1n enhancement signal."
        )
        step2_a_code = (
            f"  {overlay_var} = llm_query_with_images("
            f"'Brightest T1c region NOT bright on T1n: LEFT or RIGHT side? Size?', [1, 0])"
        )
        step2_b_code = (
            f"  {cross_var} = llm_query_with_images("
            f"'FLAIR signal around T1c mass? Dark centre on T1c? LEFT or RIGHT side?', [1, 3])"
        )
        gt_context_for_synthesis = "No ground truth. Visual inference only."
        subregions_for_synthesis = (
            "NCR (visual inference), ED (visual inference), ET (visual inference)"
        )
        overlay_label = "T1c vs T1n enhancement"
        cross_label   = "T1c + T2-FLAIR"
        final_section_label   = "CONFIDENCE"
        final_section_content = "high/medium/low per sub-region, strongest evidence modality"

    prompt = f"""\
You are a neuroradiology assistant characterizing a confirmed meningioma from BraTS 2023.

CRITICAL INSTRUCTIONS:
  1. Do NOT print(context) or probe the environment — start clinical analysis immediately.
  2. Store the final report string in a variable called `report`.
  3. End your REPL block with: FINAL_VAR("report")
  4. Do NOT write FINAL(report) — that captures the literal word "report", not its value.

Patient: {patient_id}

{gt_header}

Images ({len(image_paths)} axial slices, same brain level):
{images_block}

Write ONE REPL block with these exact steps:

  # STEP 1 — targeted per-modality descriptions (use these exact variable names)
  t1n_desc = describe_image(0, "Focal lesion: LEFT or RIGHT side of image? Dark centre (necrosis)?")
  t1c_desc = describe_image(1, "Enhancing mass: LEFT or RIGHT side? Size relative to brain? Borders?")
  t2w_desc = describe_image(2, "Hyperintense region: LEFT or RIGHT side? Extent, borders.")
  t2f_desc = describe_image(3, "FLAIR signal: oedema ring? Which SIDE? Extent beyond enhancing core?")

  # STEP 2 — overlay and cross-modal queries
{step2_a_code}
{step2_b_code}

  # STEP 3 — combine evidence and synthesise the report
  evidence = (
      f"T1 native: {{t1n_desc[:300]}}\\n"
      f"T1 contrast: {{t1c_desc[:300]}}\\n"
      f"T2 weighted: {{t2w_desc[:300]}}\\n"
      f"T2 FLAIR: {{t2f_desc[:300]}}\\n"
      f"{overlay_label}: {{{overlay_var}[:300]}}\\n"
      f"{cross_label}: {{{cross_var}[:300]}}"
  )
  report = llm_query(
      "RESPOND IN PLAIN TEXT ONLY. NO CODE BLOCKS. NO MARKDOWN FENCES.\\n"
      "Write a structured 5-section neuroradiology report.\\n"
      "Context: {gt_context_for_synthesis}\\n"
      "1. LOCATION: confirmed hemisphere (left/right from images), region, dural attachment\\n"
      "2. SUB-REGIONS: {subregions_for_synthesis}\\n"
      "3. MASS EFFECT: compression, midline shift, adjacent structures\\n"
      "4. MENINGIOMA FEATURES: dural tail, calcification, enhancement pattern, borders\\n"
      "5. {final_section_label}: {final_section_content}\\n\\n"
      f"EVIDENCE:\\n{{evidence}}"
  )

  # STEP 4 — return
  FINAL_VAR("report")\
"""

    print(f"\n{'='*80}")
    print(f"SEGMENTATION-GROUNDED ANALYSIS — {patient_id}")
    print(f"Ground truth:\n{gt_block}")
    print(f"{'='*80}\n")

    result = rvlm.completion(
        prompt=prompt,
        images=image_paths,
        root_prompt=(
            f"Characterize NCR/ED/ET sub-regions for {patient_id} "
            f"and verify against ground truth mask statistics."
        ),
    )

    print(f"\n{'='*80}")
    print(f"RESULT:\n{result.response}")
    print(f"\nTime: {result.execution_time:.2f}s")
    print(f"Usage: {result.usage_summary.to_dict()}")
    return result


def run_patient_comparison(
    rvlm: RVLM,
    slices1: dict[str, str],
    slices2: dict[str, str],
    stats1: dict,
    stats2: dict,
    pid1: str,
    pid2: str,
):
    """Compare meningioma characteristics across two patients.

    Each patient's images were extracted at their own peak-tumour slice, so
    the slice indices differ.  Ground truth stats for both are provided so
    RVLM can verify its comparative claims.
    """
    # Use T1c for comparison (most discriminative for tumour enhancement)
    modality = "t1c"
    if modality not in slices1 or modality not in slices2:
        fallbacks = ["t2f", "t2w", "t1n"]
        modality = next((k for k in fallbacks if k in slices1 and k in slices2), None)
    if modality is None:
        print("No common modality available for comparison.")
        return None

    image_paths = [slices1[modality], slices2[modality]]
    image_desc = [
        f"Image 0: {pid1} — {MRI_LABELS[modality]}  (slice {stats1.get('slice_idx', '?')})",
        f"Image 1: {pid2} — {MRI_LABELS[modality]}  (slice {stats2.get('slice_idx', '?')})",
    ]

    if "overlay" in slices1:
        image_paths.append(slices1["overlay"])
        image_desc.append(f"Image 2: {pid1} — overlay  [Red=NCR  Yellow=ED  Green=ET]")

    if "overlay" in slices2:
        image_paths.append(slices2["overlay"])
        image_desc.append(
            f"Image {len(image_paths) - 1}: {pid2} — overlay  [Red=NCR  Yellow=ED  Green=ET]"
        )

    images_block = "\n".join(f"  {d}" for d in image_desc)
    gt1 = _format_mask_stats(stats1)
    gt2 = _format_mask_stats(stats2)

    prompt = f"""\
You are comparing two confirmed meningioma cases from BraTS 2023.

PATIENT {pid1} — ground truth:
{gt1}

PATIENT {pid2} — ground truth:
{gt2}

Images:
{images_block}

Compare the two patients systematically:

Step 1  Describe each patient's T1c image independently (Images 0 and 1).
        Note enhancement pattern, tumour size impression, and location.

Step 2  Use the overlay images to compare sub-region distributions.
        Which patient has proportionally more oedema (ED) relative to tumour core?

Step 3  Quantitative comparison — verify against ground truth:
        • Which patient has the larger total tumour volume?  Does your visual
          impression match the GT numbers?
        • Which has more NCR (necrosis)?  More ET (enhancing tumour)?
        • Are both tumours in the same hemisphere?

Step 4  Radiological differences: enhancement pattern, mass effect, morphology,
        presence of dural tail, apparent grade characteristics.

Step 5  Comparative summary with clinical implications.

Call FINAL() with the comparative report.\
"""

    print(f"\n{'='*80}")
    print(f"PATIENT COMPARISON — {pid1}  vs  {pid2}")
    print(f"{'='*80}\n")

    result = rvlm.completion(
        prompt=prompt,
        images=image_paths,
        root_prompt=f"Compare meningioma sub-regions and burden: {pid1} vs {pid2}",
    )

    print(f"\n{'='*80}")
    print(f"RESULT:\n{result.response}")
    print(f"\nTime: {result.execution_time:.2f}s")
    print(f"Usage: {result.usage_summary.to_dict()}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BraTS 2023 Meningioma Sub-region Characterization with RVLM"
    )
    parser.add_argument(
        "--patient", default=None,
        help="Patient ID (e.g. BraTS-MEN-00004-000). Defaults to first in dataset.",
    )
    parser.add_argument(
        "--patient2", default=None,
        help="Second patient ID for cross-patient comparison.",
    )
    parser.add_argument(
        "--slice", type=int, default=None,
        help="Override axial slice index. Default: auto-detected from seg mask.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=12,
        help="Maximum REPL iterations for RVLM (default: 12).",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate a formal LaTeX/PDF report after the RVLM analysis.",
    )
    parser.add_argument(
        "--report-dir", default="./reports",
        help="Directory in which to save the PDF report (default: ./reports).",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Error: data directory not found: {DATA_DIR}")
        print("Place BraTS-MEN training data under data/BraTS-MEN-Train/")
        sys.exit(1)

    if args.patient is None:
        available = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
        if not available:
            print(f"Error: no patient folders found in {DATA_DIR}")
            sys.exit(1)
        args.patient = available[0]

    patient_dir = get_patient_dir(args.patient)
    print(f"\nPatient 1: {patient_dir.name}")
    print("Preparing images...")
    slices, mask_stats, _ = prepare_patient_images(patient_dir, slice_idx=args.slice)

    if not slices:
        print("Error: no modality files found.")
        sys.exit(1)

    logger = RLMLogger(log_dir="./logs/brats")

    # Build a RecursionRouter from the segmentation mask statistics.
    # This adaptively sets the iteration budget based on tumour complexity
    # (label entropy, volume, sub-region count, tiny-region flag).
    router = RecursionRouter.from_mask_stats(mask_stats, verbose=True)
    print(f"\n[Router] complexity={router.complexity_score:.2f}"
          f"  recommended_iters={router.recommended_max_iterations()}")

    rvlm = RVLM(
        backend="gemini",
        backend_kwargs={"model_name": "gemini-2.5-flash"},
        environment="local",
        max_depth=1,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=True,
        router=router,
    )

    result = run_segmentation_grounded_analysis(rvlm, slices, args.patient, mask_stats)

    if args.report and result is not None:
        gen = LatexReportGenerator(
            backend="gemini",
            backend_kwargs={"model_name": "gemini-2.5-flash"},
        )
        pdf_path = gen.generate(
            result=result,
            patient_id=args.patient,
            mask_stats=mask_stats,
            output_dir=args.report_dir,
        )
        print(f"\nClinical report: {pdf_path}")

    if args.patient2:
        patient_dir2 = get_patient_dir(args.patient2)
        print(f"\nPatient 2: {patient_dir2.name}")
        print("Preparing images...")
        slices2, mask_stats2, _ = prepare_patient_images(patient_dir2, slice_idx=args.slice)
        run_patient_comparison(
            rvlm, slices, slices2, mask_stats, mask_stats2, args.patient, args.patient2
        )


if __name__ == "__main__":
    main()
