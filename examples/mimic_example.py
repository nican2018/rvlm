"""
MIMIC-CXR Chest X-Ray Report Generation with RVLM.

Problem statement
-----------------
Given PA (posteroanterior) and Lateral chest X-ray images from a real clinical
visit, can RVLM's recursive REPL loop generate a structured radiology report
comparable to the ground truth written by a radiologist?

Dataset structure
-----------------
Each CSV row = one patient (subject_id) with images from multiple study visits:
    subject_id  -- integer patient identifier
    image       -- all JPG paths across all studies
    PA          -- PA-view image paths
    Lateral     -- lateral-view image paths
    AP          -- portable AP-view image paths
    view        -- list of view types present
    text        -- ground truth reports, one per study
    text_augment-- augmented / paraphrased GT reports

RVLM approach
-------------
Iteration 1  Context probe (by RVLM design — always happens).

Iteration 2  REPL block:
    Step 1  describe_image() per view: PA systematic review, Lateral if present.
    Step 2  llm_query_with_images() cross-view: compare cardiac silhouette,
            detect any consolidation, effusion, or pneumothorax.
    Step 3  llm_query() synthesis: generate formal Findings + Impression.
    Step 4  FINAL_VAR("report")

Usage
-----
    # Analyse first validation patient
    python -m examples.mimic_example

    # Specify a patient by subject_id
    python -m examples.mimic_example --subject 10000032

    # Use training set instead of validation
    python -m examples.mimic_example --split train --subject 10000032

    # Generate a PDF report
    python -m examples.mimic_example --subject 10000032 --report

    # Limit to most recent N studies (default: most recent 1)
    python -m examples.mimic_example --subject 10000032 --study-index 0

Requires
--------
    pip install pandas Pillow
"""

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

from rvlm import RVLM
from rvlm.logger import RLMLogger
from examples.latex_report import LatexReportGenerator, CXRLatexReportGenerator

load_dotenv = None
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_DIR   = Path(__file__).parent.parent / "data" / "MIMMIC"
IMAGES_DIR = DATA_DIR / "official_data_iccv_final"

VIEW_LABELS = {
    "PA":      "Posteroanterior (PA) chest X-ray",
    "Lateral": "Lateral chest X-ray",
    "AP":      "Anteroposterior (AP, portable) chest X-ray",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(split: str = "validate") -> pd.DataFrame:
    """Load the MIMIC-CXR CSV for the requested split."""
    fname = DATA_DIR / f"mimic_cxr_aug_{split}.csv"
    if not fname.exists():
        print(f"Error: CSV not found: {fname}")
        sys.exit(1)
    df = pd.read_csv(fname, encoding="utf-8")
    for col in ("image", "PA", "Lateral", "AP", "view", "text", "text_augment"):
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    return df


def get_patient_row(df: pd.DataFrame, subject_id: int) -> pd.Series:
    """Return the DataFrame row for a given subject_id."""
    matches = df[df["subject_id"] == subject_id]
    if matches.empty:
        print(f"Error: subject_id {subject_id} not found in this split.")
        sys.exit(1)
    return matches.iloc[0]


def select_images(row: pd.Series, study_index: int = -1) -> tuple[list[str], list[str], str]:
    """
    Select the chest X-ray images for one study visit from a patient row.

    Parameters
    ----------
    row : pd.Series
        A row from the MIMIC CSV (already parsed lists).
    study_index : int
        Which study to use. -1 = most recent (last in sorted order).
        0 = earliest, 1 = second, etc.

    Returns
    -------
    image_paths : list[str]
        Absolute paths to the selected images (PA first, then Lateral / AP).
    view_types  : list[str]
        Corresponding view labels ("PA", "Lateral", "AP").
    gt_report   : str
        Ground truth radiology report for the selected study.
    """
    # Group all images by study ID (extracted from path)
    all_paths: list[str] = row["image"]
    studies: dict[str, list[tuple[str, str]]] = {}   # study_id -> [(path, view_type)]

    for path in all_paths:
        # path e.g. "files/p10/p10000032/s50414267/02aa804e-....jpg"
        parts = Path(path).parts
        study_id = parts[3] if len(parts) >= 5 else "unknown"
        view = _detect_view(path, row)
        studies.setdefault(study_id, []).append((path, view))

    sorted_studies = sorted(studies.keys())   # alphabetical ≈ chronological (s-prefix)
    selected_study_id = sorted_studies[study_index]
    selected_imgs = studies[selected_study_id]

    # Order: PA first, then Lateral, then AP, then any other
    order = {"PA": 0, "Lateral": 1, "AP": 2}
    selected_imgs.sort(key=lambda x: order.get(x[1], 9))

    # Resolve absolute paths
    image_paths = [str(IMAGES_DIR / p) for p, _ in selected_imgs]
    view_types  = [v for _, v in selected_imgs]

    # Match the GT report: sorted_studies index -> text index
    gt_texts: list[str] = row["text"]
    gt_index = sorted_studies.index(selected_study_id)
    gt_report = gt_texts[gt_index] if gt_index < len(gt_texts) else "(no report)"

    return image_paths, view_types, gt_report


def _detect_view(img_path: str, row: pd.Series) -> str:
    """Return 'PA', 'Lateral', or 'AP' for an image path using the CSV lists."""
    rel = img_path  # paths in CSV are relative to IMAGES_DIR
    if img_path in (row.get("PA") or []):
        return "PA"
    if img_path in (row.get("Lateral") or []):
        return "Lateral"
    if img_path in (row.get("AP") or []):
        return "AP"
    return "Unknown"


# ---------------------------------------------------------------------------
# RVLM prompt builder
# ---------------------------------------------------------------------------

def _build_cxr_prompt(
    subject_id: int,
    image_paths: list[str],
    view_types:  list[str],
    gt_report:   str,
) -> str:
    """Build the RVLM user prompt for chest X-ray report generation."""

    # Image index list for the prompt
    images_block = "\n".join(
        f"  Image {i}: {VIEW_LABELS.get(v, v)}"
        for i, v in enumerate(view_types)
    )

    # Determine which image indices hold PA and Lateral views
    pa_idx  = next((i for i, v in enumerate(view_types) if v == "PA"),      None)
    lat_idx = next((i for i, v in enumerate(view_types) if v == "Lateral"), None)
    ap_idx  = next((i for i, v in enumerate(view_types) if v == "AP"),      None)

    # Step 1 — per-view descriptions
    step1_lines: list[str] = []
    if pa_idx is not None:
        step1_lines.append(
            f"  pa_desc = describe_image({pa_idx}, "
            f"\"PA chest X-ray systematic review: \"\n"
            f"    \"1) Lung fields — consolidation, opacity, hyperinflation, atelectasis, nodules\\n\"\n"
            f"    \"2) Cardiomediastinal — heart size (< half thorax?), contour, trachea, hila\\n\"\n"
            f"    \"3) Pleura — effusion (blunted costophrenic angles?), pneumothorax\\n\"\n"
            f"    \"4) Bones & soft tissue — rib fractures, calcifications, foreign bodies\\n\"\n"
            f"    \"5) Diaphragm — elevated? free subdiaphragmatic air?\")"
        )
    if lat_idx is not None:
        step1_lines.append(
            f"  lat_desc = describe_image({lat_idx}, "
            f"\"Lateral chest X-ray: retrosternal space (clear/filled?), \"\n"
            f"    \"retrocardiac opacity, posterior costophrenic angles, \"\n"
            f"    \"vertebral column density gradient (should get darker inferiorly).\")"
        )
    if ap_idx is not None and pa_idx is None:
        step1_lines.append(
            f"  pa_desc = describe_image({ap_idx}, "
            f"\"AP portable chest X-ray (note: AP view magnifies cardiac shadow): \"\n"
            f"    \"lung fields, cardiac size, pleural spaces, support devices.\")"
        )
    step1_code = "\n".join(step1_lines)

    # Step 2 — cross-view query
    cross_indices = [i for i, v in enumerate(view_types) if v in ("PA", "Lateral", "AP")]
    cross_indices_str = str(cross_indices)
    step2_code = (
        f"  cross_q = llm_query_with_images(\n"
        f"    \"Cross-view assessment: Is the cardiac silhouette enlarged? \"\n"
        f"    \"Is there consolidation (unilateral or bilateral)? \"\n"
        f"    \"Pleural effusion present? Pneumothorax? Pulmonary oedema?\",\n"
        f"    {cross_indices_str})"
    )

    # Build evidence vars used in synthesis
    desc_vars = []
    if pa_idx is not None or ap_idx is not None:
        desc_vars.append("pa_desc")
    if lat_idx is not None:
        desc_vars.append("lat_desc")

    if lat_idx is not None:
        evidence_code = (
            "  evidence = (\n"
            "      f\"PA view: {pa_desc[:400]}\\n\"\n"
            "      f\"Lateral view: {lat_desc[:400]}\\n\"\n"
            "      f\"Cross-view: {cross_q[:400]}\"\n"
            "  )"
        )
    elif ap_idx is not None and pa_idx is None:
        evidence_code = (
            "  evidence = (\n"
            "      f\"AP view: {pa_desc[:400]}\\n\"\n"
            "      f\"Cross-view: {cross_q[:400]}\"\n"
            "  )"
        )
    else:
        evidence_code = (
            "  evidence = (\n"
            "      f\"PA view: {pa_desc[:400]}\\n\"\n"
            "      f\"Cross-view: {cross_q[:400]}\"\n"
            "  )"
        )

    # GT for model reference (truncated)
    gt_block = gt_report[:600] if gt_report and gt_report != "(no report)" else "(not available)"

    prompt = f"""\
You are a board-certified radiologist generating a structured chest X-ray report.

CRITICAL INSTRUCTIONS:
  1. Do NOT print(context) or probe the environment — start analysis immediately.
  2. Store the final report string in a variable called `report`.
  3. End your REPL block with: FINAL_VAR("report")
  4. Do NOT write FINAL(report) — that captures the literal word "report".

Patient: MIMIC-CXR subject {subject_id}

Reference report (ground truth, for calibration only):
{gt_block}

Images ({len(image_paths)} chest X-ray view(s)):
{images_block}

Write ONE REPL block with these exact steps:

  # STEP 1 — systematic per-view image descriptions
{step1_code}

  # STEP 2 — cross-view assessment
{step2_code}

  # STEP 3 — synthesise formal radiology report
{evidence_code}
  report = llm_query(
      "RESPOND IN PLAIN TEXT ONLY. NO CODE BLOCKS. NO MARKDOWN.\\n"
      "Write a formal chest X-ray radiology report with two labelled sections:\\n"
      "FINDINGS:\\n"
      "  Systematically describe: lungs (parenchyma, vasculature), cardiac silhouette,\\n"
      "  mediastinum, pleural spaces, bones, and any support devices.\\n"
      "IMPRESSION:\\n"
      "  One to three concise clinical conclusions.\\n\\n"
      f"IMAGE EVIDENCE:\\n{{evidence}}"
  )

  # STEP 4 — return
  FINAL_VAR("report")\
"""
    return prompt


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_cxr_analysis(
    rvlm: RVLM,
    subject_id: int,
    image_paths: list[str],
    view_types:  list[str],
    gt_report:   str,
):
    """Run RVLM chest X-ray analysis for one patient study."""
    prompt = _build_cxr_prompt(subject_id, image_paths, view_types, gt_report)

    print(f"\n{'='*80}")
    print(f"CXR ANALYSIS — subject {subject_id}")
    print(f"Views: {view_types}")
    print(f"Ground truth (truncated):\n  {gt_report[:300].replace(chr(10), chr(10)+'  ')}")
    print(f"{'='*80}\n")

    result = rvlm.completion(
        prompt=prompt,
        images=image_paths,
        root_prompt=(
            f"Generate a structured chest X-ray radiology report for "
            f"MIMIC subject {subject_id}."
        ),
    )

    print(f"\n{'='*80}")
    print(f"GENERATED REPORT:\n{result.response}")
    print(f"\nTime: {result.execution_time:.2f}s")
    print(f"Usage: {result.usage_summary.to_dict()}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MIMIC-CXR Chest X-Ray Report Generation with RVLM"
    )
    parser.add_argument(
        "--subject", type=int, default=None,
        help="Patient subject_id. Defaults to first in the split.",
    )
    parser.add_argument(
        "--split", default="validate", choices=["train", "validate"],
        help="CSV split to use (default: validate).",
    )
    parser.add_argument(
        "--study-index", type=int, default=-1,
        help="Which study to use: -1=most recent (default), 0=earliest.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=4,
        help="Maximum REPL iterations for RVLM (default: 4).",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate a formal LaTeX/PDF report after the analysis.",
    )
    parser.add_argument(
        "--report-dir", default="./reports",
        help="Directory in which to save the PDF report (default: ./reports).",
    )
    args = parser.parse_args()

    print(f"Loading MIMIC-CXR {args.split} split...")
    df = load_csv(args.split)

    if args.subject is None:
        args.subject = int(df.iloc[0]["subject_id"])
        print(f"No subject specified — using first: {args.subject}")

    row = get_patient_row(df, args.subject)
    image_paths, view_types, gt_report = select_images(row, study_index=args.study_index)

    if not image_paths:
        print("Error: no images found for this patient/study.")
        sys.exit(1)

    print(f"Subject  : {args.subject}")
    print(f"Images   : {len(image_paths)} ({', '.join(view_types)})")
    for i, (p, v) in enumerate(zip(image_paths, view_types)):
        print(f"  Image {i}: [{v}] {p}")

    logger = RLMLogger(log_dir="./logs/mimic")

    rvlm = RVLM(
        backend="gemini",
        backend_kwargs={"model_name": "gemini-2.5-flash"},
        environment="local",
        max_depth=1,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=True,
    )

    result = run_cxr_analysis(rvlm, args.subject, image_paths, view_types, gt_report)

    if args.report and result is not None:
        cxr_info = {
            "subject_id": args.subject,
            "split": args.split,
            "views": view_types,
            "n_images": len(image_paths),
            "gt_report": gt_report,
        }
        gen = CXRLatexReportGenerator(
            backend="gemini",
            backend_kwargs={"model_name": "gemini-2.5-flash"},
        )
        pdf_path = gen.generate(
            result=result,
            patient_id=f"MIMIC-{args.subject}",
            cxr_info=cxr_info,
            output_dir=args.report_dir,
        )
        print(f"\nClinical report: {pdf_path}")


if __name__ == "__main__":
    main()
