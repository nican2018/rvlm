"""
LaTeX Report Generator for RVLM Meningioma Analysis.

Takes the free-text output of run_segmentation_grounded_analysis() and
produces a formal clinical PDF report suitable for physician review.

The generator makes one additional LLM call to extract the five clinical
sections from the RVLM output as clean plain text, then assembles a
LaTeX document using a predefined medical-report template and compiles
it to PDF with pdflatex.

Usage (standalone):
    from examples.latex_report import LatexReportGenerator
    gen = LatexReportGenerator(backend="gemini",
                                backend_kwargs={"model_name": "gemini-2.5-flash"})
    pdf_path = gen.generate(result, patient_id, mask_stats, output_dir=Path("."))
    print(f"Report saved to: {pdf_path}")
"""

import json
import re
import subprocess
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from rvlm.clients import get_client


# ---------------------------------------------------------------------------
# LaTeX character escaping
# ---------------------------------------------------------------------------

_LATEX_ESCAPE_MAP = str.maketrans({
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
})


def _esc(text: str) -> str:
    """Escape special LaTeX characters in a plain-text string."""
    return text.translate(_LATEX_ESCAPE_MAP)


def _strip_markdown(text: str) -> str:
    """Remove common markdown constructs (bold, italic, bullets, headers)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"__(.+?)__",     r"\1", text)
    text = re.sub(r"_(.+?)_",       r"\1", text)
    text = re.sub(r"^\s*[*\-]\s+",  "",    text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+",    "",    text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}",        "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section extraction via LLM
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a medical text extractor. The text below is an AI-generated neuroradiology \
analysis of a meningioma. Extract EXACTLY five sections from it and return them as \
a JSON object with these keys:

  "location"      - Tumour location: hemisphere, lobe / region, and dural attachment.
  "sub_regions"   - NCR, ED, and ET sub-region descriptions with volume data.
  "mass_effect"   - Mass effect, compression, midline shift, adjacent structures.
  "features"      - Meningioma features: enhancement, borders, dural tail, calcification.
  "agreement"     - Agreement or discrepancy between visual findings and ground truth.

STRICT RULES:
  - Output valid JSON only. No preamble, no explanation, no markdown code fences.
  - Each value must be a single plain-text paragraph (1-4 sentences). No bullet lists.
  - Do NOT use double-quotes inside the text (use single quotes if needed).
  - If a section cannot be determined from the text, write: "Not available."

SOURCE TEXT:
{report_text}
"""


def _extract_sections(client, report_text: str) -> dict[str, str]:
    """Call the LLM to extract the five clinical sections as plain JSON."""
    prompt = _EXTRACTION_PROMPT.format(report_text=report_text)
    messages = [{"role": "user", "content": prompt}]
    raw = client.completion(messages)

    # Strip any accidental markdown code fences the model might add
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```$",          "", raw.strip())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        fallback = _strip_markdown(report_text)[:400] + " [extraction failed]"
        data = {k: fallback for k in
                ("location", "sub_regions", "mass_effect", "features", "agreement")}

    for key in ("location", "sub_regions", "mass_effect", "features", "agreement"):
        data.setdefault(key, "Not available.")

    return data


# ---------------------------------------------------------------------------
# LaTeX template
# Note: placeholders use <<NAME>> so they never clash with LaTeX { } syntax.
# ---------------------------------------------------------------------------

_LATEX_TEMPLATE = r"""\documentclass[11pt,a4paper]{article}

%% --- packages ---
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[a4paper, top=2.2cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{geometry}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{parskip}
\usepackage{titlesec}

%% --- colours ---
\definecolor{headerblue}{RGB}{0,62,116}
\definecolor{ruleblue}{RGB}{0,112,192}
\definecolor{warnorange}{RGB}{180,90,0}

%% --- section style ---
\titleformat{\section}
  {\normalfont\large\bfseries\color{headerblue}}
  {}{0pt}{}[\vspace{-4pt}\textcolor{ruleblue}{\hrule height 0.8pt}\vspace{4pt}]
\titlespacing*{\section}{0pt}{14pt}{6pt}

%% --- header / footer ---
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[L]{\small\color{gray}\textit{AI-Assisted Neuroradiology Report --- CONFIDENTIAL}}
\fancyhead[R]{\small\color{gray}<<DATE>>}
\fancyfoot[C]{\small Page \thepage\ of \pageref{LastPage}}

\begin{document}

%% ===== TITLE BLOCK =====
\noindent
{\color{headerblue}\rule{\linewidth}{2pt}}\\[4pt]
{\LARGE\bfseries\color{headerblue} Neuroradiology Report}\\[2pt]
{\large AI-Assisted Meningioma Sub-Region Characterization}\\[2pt]
{\color{ruleblue}\rule{\linewidth}{1pt}}

\vspace{8pt}

%% ===== PATIENT / STUDY INFORMATION =====
\noindent
\begin{tabularx}{\linewidth}{>{}l<{}  X  >{}l<{}  X}
\textbf{Patient ID:}    & <<PATIENT_ID>> &
\textbf{Report Date:}   & <<DATE>> \\[3pt]
\textbf{Study:}         & BraTS 2023 Meningioma &
\textbf{Analysis System:} & RVLM (Recursive VLM) \\[3pt]
\textbf{Model:}         & <<MODEL_NAME>> &
\textbf{Exec.\ Time:}   & <<EXEC_TIME>>~s \\
\end{tabularx}

\vspace{10pt}

%% ===== QUANTITATIVE STATISTICS =====
\section{Quantitative Statistics (Segmentation Mask)}

<<STATS_BLOCK>>

%% ===== CLINICAL REPORT =====
\section{1.\enspace Tumour Location}

<<LOCATION>>

\section{2.\enspace Sub-Region Characterization}

<<SUB_REGIONS>>

\section{3.\enspace Mass Effect}

<<MASS_EFFECT>>

\section{4.\enspace Meningioma Features}

<<FEATURES>>

\section{5.\enspace Agreement: Visual Findings vs.\ Ground Truth}

<<AGREEMENT>>

%% ===== AI DISCLAIMER =====
\vfill
{\color{ruleblue}\rule{\linewidth}{0.6pt}}\\[3pt]
\noindent
{\small\color{warnorange}\textbf{AI-Generated Report --- For Research Use Only}}\\[2pt]
{\small
This report was produced automatically by the RVLM (Recursive Vision Language
Model) system and has \textbf{not} been reviewed or verified by a licensed
radiologist or clinician. It is intended solely for research and educational
purposes within the BraTS 2023 dataset context. \textbf{Do not use this report
for clinical decision-making.} All findings must be independently confirmed by
a qualified healthcare professional before any clinical action is taken.
}

\end{document}
"""

_STATS_WITH_GT = r"""\noindent
\begin{tabularx}{\linewidth}{l r r X}
\toprule
\textbf{Sub-Region} & \textbf{Volume (cc)} & \textbf{Share} & \textbf{Notes} \\
\midrule
NCR --- Necrotic Core      & <<NCR>>   & <<NCR_PCT>>\%  & Non-enhancing, dark on T1c \\
ED  --- Peritumoral Oedema & <<ED>>    & <<ED_PCT>>\%   & Bright on T2w / FLAIR \\
ET  --- Enhancing Tumour   & <<ET>>    & <<ET_PCT>>\%   & Bright on T1c, not on T1n \\
\midrule
\textbf{Total}             & \textbf{<<TOTAL>>} & \textbf{100\%} & \\
\bottomrule
\end{tabularx}

\smallskip
\noindent
\textbf{Peak slice:} <<SLICE_IDX>> of <<TOTAL_SLICES>> axial slices \quad
\textbf{Centroid:} <<HEMISPHERE>> hemisphere, <<POSITION_AP>>"""

_STATS_NO_GT = r"""\noindent
\textit{No segmentation mask was available for this case. The analysis slice was
selected using peak T1c--T1n contrast enhancement signal. Sub-region volumes
are estimated visually from image analysis only.}"""


def _build_stats_block(mask_stats: dict | None) -> str:
    if not mask_stats:
        return _STATS_NO_GT.strip()

    ncr   = mask_stats.get("NCR_volume_cc", 0.0)
    ed    = mask_stats.get("ED_volume_cc",  0.0)
    et    = mask_stats.get("ET_volume_cc",  0.0)
    total = ncr + ed + et or 1.0

    return (
        _STATS_WITH_GT
        .replace("<<NCR>>",        f"{ncr:.2f}")
        .replace("<<NCR_PCT>>",    f"{100 * ncr / total:.0f}")
        .replace("<<ED>>",         f"{ed:.2f}")
        .replace("<<ED_PCT>>",     f"{100 * ed / total:.0f}")
        .replace("<<ET>>",         f"{et:.2f}")
        .replace("<<ET_PCT>>",     f"{100 * et / total:.0f}")
        .replace("<<TOTAL>>",      f"{ncr + ed + et:.2f}")
        .replace("<<SLICE_IDX>>",  str(mask_stats.get("slice_idx",    "?")))
        .replace("<<TOTAL_SLICES>>", str(mask_stats.get("total_slices", "?")))
        .replace("<<HEMISPHERE>>", _esc(str(mask_stats.get("hemisphere",  "unknown"))))
        .replace("<<POSITION_AP>>", _esc(str(mask_stats.get("position_ap", "unknown"))))
    ).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LatexReportGenerator:
    """
    Converts an RVLM meningioma analysis result into a formal LaTeX / PDF report.

    Parameters
    ----------
    backend : str
        The LLM backend to use for section extraction (e.g. "gemini", "openai").
    backend_kwargs : dict | None
        Kwargs forwarded to get_client() (e.g. {"model_name": "gemini-2.5-flash"}).
    """

    def __init__(
        self,
        backend: str = "gemini",
        backend_kwargs: dict[str, Any] | None = None,
    ):
        self.client = get_client(backend, backend_kwargs)
        self.backend_kwargs = backend_kwargs or {}

    def generate(
        self,
        result: Any,
        patient_id: str,
        mask_stats: dict | None = None,
        output_dir: Path | str = ".",
    ) -> Path:
        """
        Generate a LaTeX PDF report from an RLMChatCompletion result.

        Parameters
        ----------
        result : RLMChatCompletion
            The object returned by rvlm.completion().
        patient_id : str
            Patient identifier (e.g. "BraTS-MEN-00008-000").
        mask_stats : dict | None
            Dictionary returned by compute_mask_stats(); None for inference mode.
        output_dir : Path | str
            Directory in which to write the .tex and .pdf files.

        Returns
        -------
        Path
            Absolute path to the generated PDF (or .tex if pdflatex failed).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n[LatexReport] Extracting clinical sections from RVLM output...")
        sections = _extract_sections(self.client, result.response)

        print("[LatexReport] Building LaTeX document...")
        latex_src = self._build_latex(result, patient_id, mask_stats, sections)

        tex_path = output_dir / f"{patient_id}_report.tex"
        tex_path.write_text(latex_src, encoding="utf-8")
        print(f"[LatexReport] LaTeX source saved: {tex_path}")

        pdf_path = self._compile(tex_path, output_dir)
        return pdf_path

    # ------------------------------------------------------------------
    def _build_latex(self, result, patient_id, mask_stats, sections):
        model_name = self.backend_kwargs.get("model_name", "unknown")

        return (
            _LATEX_TEMPLATE
            .replace("<<DATE>>",        date.today().isoformat())
            .replace("<<PATIENT_ID>>",  _esc(patient_id))
            .replace("<<MODEL_NAME>>",  _esc(model_name))
            .replace("<<EXEC_TIME>>",   f"{result.execution_time:.1f}")
            .replace("<<STATS_BLOCK>>", _build_stats_block(mask_stats))
            .replace("<<LOCATION>>",    _esc(_strip_markdown(sections["location"])))
            .replace("<<SUB_REGIONS>>", _esc(_strip_markdown(sections["sub_regions"])))
            .replace("<<MASS_EFFECT>>", _esc(_strip_markdown(sections["mass_effect"])))
            .replace("<<FEATURES>>",    _esc(_strip_markdown(sections["features"])))
            .replace("<<AGREEMENT>>",   _esc(_strip_markdown(sections["agreement"])))
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compile(tex_path: Path, output_dir: Path) -> Path:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            print("[LatexReport] pdflatex not found on PATH — PDF not compiled.")
            return tex_path

        # Run pdflatex from inside output_dir and pass only the filename.
        # This avoids Windows backslash-in-path being misread as LaTeX commands.
        for run in (1, 2):   # two passes for correct page references
            print(f"[LatexReport] pdflatex pass {run}/2...")
            proc = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", tex_path.name],
                cwd=str(output_dir),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 and run == 2:
                print("[LatexReport] Compilation errors (last 20 log lines):")
                for line in proc.stdout.splitlines()[-20:]:
                    print(f"  {line}")

        pdf_path = output_dir / tex_path.with_suffix(".pdf").name
        if pdf_path.exists():
            print(f"[LatexReport] PDF report ready: {pdf_path}")
            # Clean up LaTeX auxiliary files — keep only the PDF
            for suffix in (".tex", ".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk"):
                aux_file = output_dir / tex_path.with_suffix(suffix).name
                if aux_file.exists():
                    aux_file.unlink()
            print("[LatexReport] Cleaned up .tex / .aux / .log files")
        else:
            print(f"[LatexReport] PDF not found — check {tex_path.with_suffix('.log')}")
        return pdf_path


# ===========================================================================
# CXR (Chest X-Ray) variant
# ===========================================================================

_CXR_EXTRACTION_PROMPT = """\
You are a medical text extractor. The text below is an AI-generated chest X-ray
radiology report. Extract EXACTLY five sections and return them as a JSON object
with these keys:

  "lungs"      - Lung parenchyma findings: consolidation, opacities, atelectasis,
                 hyperinflation, nodules, vascular markings.
  "cardiac"    - Cardiac and mediastinal findings: heart size, contour, trachea,
                 hilar structures, mediastinal width.
  "pleura"     - Pleural findings: effusion (size/side), pneumothorax, thickening.
  "bones"      - Bones and soft tissue: rib fractures, calcifications, soft tissue
                 abnormalities, support devices (lines, tubes, pacemakers).
  "impression" - Impression / conclusion: 1-3 concise clinical statements.

STRICT RULES:
  - Output valid JSON only. No preamble, no explanation, no markdown code fences.
  - Each value must be 1-4 sentences of plain text. No bullet lists.
  - Do NOT use double-quotes inside the text (use single quotes if needed).
  - If a section cannot be determined, write: "No significant abnormality identified."

SOURCE TEXT:
{report_text}
"""

_CXR_LATEX_TEMPLATE = r"""\documentclass[11pt,a4paper]{article}

%% --- packages ---
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[a4paper, top=2.2cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{geometry}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{parskip}
\usepackage{titlesec}

%% --- colours ---
\definecolor{headerblue}{RGB}{0,62,116}
\definecolor{ruleblue}{RGB}{0,112,192}
\definecolor{warnorange}{RGB}{180,90,0}

%% --- section style ---
\titleformat{\section}
  {\normalfont\large\bfseries\color{headerblue}}
  {}{0pt}{}[\vspace{-4pt}\textcolor{ruleblue}{\hrule height 0.8pt}\vspace{4pt}]
\titlespacing*{\section}{0pt}{14pt}{6pt}

%% --- header / footer ---
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[L]{\small\color{gray}\textit{AI-Assisted Radiology Report --- CONFIDENTIAL}}
\fancyhead[R]{\small\color{gray}<<DATE>>}
\fancyfoot[C]{\small Page \thepage\ of \pageref{LastPage}}

\begin{document}

%% ===== TITLE BLOCK =====
\noindent
{\color{headerblue}\rule{\linewidth}{2pt}}\\[4pt]
{\LARGE\bfseries\color{headerblue} Chest X-Ray Radiology Report}\\[2pt]
{\large AI-Assisted Interpretation --- MIMIC-CXR}\\[2pt]
{\color{ruleblue}\rule{\linewidth}{1pt}}

\vspace{8pt}

%% ===== PATIENT / STUDY INFORMATION =====
\noindent
\begin{tabularx}{\linewidth}{>{}l<{}  X  >{}l<{}  X}
\textbf{Patient ID:}       & <<PATIENT_ID>> &
\textbf{Report Date:}      & <<DATE>> \\[3pt]
\textbf{Dataset:}          & MIMIC-CXR &
\textbf{Analysis System:}  & RVLM (Recursive VLM) \\[3pt]
\textbf{Model:}            & <<MODEL_NAME>> &
\textbf{Exec.\ Time:}      & <<EXEC_TIME>>~s \\[3pt]
\textbf{Views:}            & <<VIEWS>> &
\textbf{Images:}           & <<N_IMAGES>> \\
\end{tabularx}

\vspace{10pt}

%% ===== GROUND TRUTH REFERENCE =====
\section{Ground Truth Reference (Radiologist Report)}

{\small\itshape
<<GT_REPORT>>
}

%% ===== AI-GENERATED FINDINGS =====
\section{1.\enspace Lungs and Airways}

<<LUNGS>>

\section{2.\enspace Cardiac and Mediastinum}

<<CARDIAC>>

\section{3.\enspace Pleura and Diaphragm}

<<PLEURA>>

\section{4.\enspace Bones and Soft Tissue}

<<BONES>>

\section{5.\enspace Impression}

\textbf{<<IMPRESSION>>}

%% ===== AI DISCLAIMER =====
\vfill
{\color{ruleblue}\rule{\linewidth}{0.6pt}}\\[3pt]
\noindent
{\small\color{warnorange}\textbf{AI-Generated Report --- For Research Use Only}}\\[2pt]
{\small
This report was produced automatically by the RVLM (Recursive Vision Language
Model) system and has \textbf{not} been reviewed or verified by a licensed
radiologist or clinician. It is intended solely for research and educational
purposes within the MIMIC-CXR dataset context. \textbf{Do not use this report
for clinical decision-making.} All findings must be independently confirmed by
a qualified healthcare professional before any clinical action is taken.
}

\end{document}
"""


class CXRLatexReportGenerator(LatexReportGenerator):
    """
    LaTeX/PDF report generator for chest X-ray analysis (MIMIC-CXR).

    Produces a formal radiology-style PDF with five clinical sections:
    Lungs, Cardiac/Mediastinum, Pleura, Bones, and Impression.

    Parameters
    ----------
    backend : str
        LLM backend for section extraction (e.g. "gemini", "openai").
    backend_kwargs : dict | None
        Kwargs forwarded to get_client().
    """

    def generate(
        self,
        result: Any,
        patient_id: str,
        cxr_info: dict | None = None,
        output_dir: Path | str = ".",
    ) -> Path:
        """
        Generate a LaTeX PDF chest X-ray report.

        Parameters
        ----------
        result : RLMChatCompletion
            The object returned by rvlm.completion().
        patient_id : str
            Patient identifier string (e.g. "MIMIC-10000032").
        cxr_info : dict | None
            Optional dict with keys: views, n_images, gt_report, subject_id, split.
        output_dir : Path | str
            Directory in which to write the .tex and .pdf files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cxr_info = cxr_info or {}

        print("\n[CXRReport] Extracting clinical sections from RVLM output...")
        sections = self._extract_cxr_sections(result.response)

        print("[CXRReport] Building LaTeX document...")
        latex_src = self._build_cxr_latex(result, patient_id, cxr_info, sections)

        tex_path = output_dir / f"{patient_id}_cxr_report.tex"
        tex_path.write_text(latex_src, encoding="utf-8")
        print(f"[CXRReport] LaTeX source saved: {tex_path}")

        pdf_path = self._compile(tex_path, output_dir)
        return pdf_path

    def _extract_cxr_sections(self, report_text: str) -> dict[str, str]:
        prompt = _CXR_EXTRACTION_PROMPT.format(report_text=report_text)
        messages = [{"role": "user", "content": prompt}]
        raw = self.client.completion(messages)
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"```$", "", raw.strip())
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            fallback = _strip_markdown(report_text)[:400] + " [extraction failed]"
            data = {k: fallback for k in ("lungs", "cardiac", "pleura", "bones", "impression")}
        for key in ("lungs", "cardiac", "pleura", "bones", "impression"):
            data.setdefault(key, "No significant abnormality identified.")
        return data

    def _build_cxr_latex(self, result, patient_id, cxr_info, sections):
        model_name = self.backend_kwargs.get("model_name", "unknown")
        views_str  = ", ".join(cxr_info.get("views", ["unknown"]))
        n_images   = str(cxr_info.get("n_images", "?"))
        gt_raw     = cxr_info.get("gt_report", "Not available.")
        # Truncate GT to avoid overflowing page
        gt_display = _esc(_strip_markdown(gt_raw[:800]))

        return (
            _CXR_LATEX_TEMPLATE
            .replace("<<DATE>>",       date.today().isoformat())
            .replace("<<PATIENT_ID>>", _esc(patient_id))
            .replace("<<MODEL_NAME>>", _esc(model_name))
            .replace("<<EXEC_TIME>>",  f"{result.execution_time:.1f}")
            .replace("<<VIEWS>>",      _esc(views_str))
            .replace("<<N_IMAGES>>",   n_images)
            .replace("<<GT_REPORT>>",  gt_display)
            .replace("<<LUNGS>>",      _esc(_strip_markdown(sections["lungs"])))
            .replace("<<CARDIAC>>",    _esc(_strip_markdown(sections["cardiac"])))
            .replace("<<PLEURA>>",     _esc(_strip_markdown(sections["pleura"])))
            .replace("<<BONES>>",      _esc(_strip_markdown(sections["bones"])))
            .replace("<<IMPRESSION>>", _esc(_strip_markdown(sections["impression"])))
        )
