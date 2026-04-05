"""
Test MedGemma 4B multimodal with recursive refinement loops.

Usage examples:
    # vLLM/OpenAI-compatible endpoint
    python -m examples.medgemma_recursive_example --backend vllm --model medgemma-4b-it --base-url http://localhost:8000/v1

    # Gemini backend (if MedGemma is available via Google API in your account)
    python -m examples.medgemma_recursive_example --backend gemini --model medgemma-4b-it

    # Use a local image file
    python -m examples.medgemma_recursive_example --image path/to/scan.png

    # Use MIMIC-CXR sample from local dataset CSV/images
    python -m examples.medgemma_recursive_example --use-mimic --mimic-split validate --mimic-subject 10000032
"""

import argparse
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv

from rvlm import RVLM
from rvlm.logger import RLMLogger

load_dotenv()

SAMPLE_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/2/20/Pneumothorax_CXR.jpg"


def validate_image_source(image: str) -> None:
    if image.startswith(("http://", "https://", "data:")):
        return
    if not os.path.exists(image):
        raise FileNotFoundError(f"Image file not found: {image}")


def validate_image_sources(images: list[str]) -> None:
    for image in images:
        validate_image_source(image)


def build_backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model_name": args.model}
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.base_url:
        kwargs["base_url"] = args.base_url
    return kwargs


def print_section(title: str, content: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(content)


class HfLocalMedGemma:
    """Minimal local Hugging Face multimodal chat wrapper for MedGemma/Gemma-like VLMs."""

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 512,
        hf_token: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "hf_local backend requires transformers, torch, and Pillow to be installed."
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, token=hf_token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            token=hf_token,
        )
        # Avoid noisy generation warnings when model config carries a default max_length.
        self.model.generation_config.max_length = None
        self.max_new_tokens = max_new_tokens

    def completion(self, prompt: str, images: list[str]) -> str:
        import base64
        from io import BytesIO

        from PIL import Image

        pil_images: list[Image.Image] = []
        for source in images:
            if source.startswith(("http://", "https://")):
                import requests

                response = requests.get(source, timeout=30)
                response.raise_for_status()
                pil_images.append(Image.open(BytesIO(response.content)).convert("RGB"))
            elif source.startswith("data:"):
                _, b64_data = source.split(",", 1)
                pil_images.append(Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB"))
            else:
                pil_images.append(Image.open(source).convert("RGB"))

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for _ in pil_images:
            content.append({"type": "image"})
        messages = [{"role": "user", "content": content}]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=chat_text,
            images=pil_images,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()


def run_single_pass(rvlm: RVLM, images: list[str]) -> str:
    single_prompt = (
        "Analyze this medical image set and provide:\n"
        "1) key findings,\n"
        "2) likely abnormalities,\n"
        "3) initial diagnosis and differential diagnosis,\n"
        "4) one-line confidence estimate."
    )
    result = rvlm.completion(
        prompt=single_prompt,
        images=images,
        root_prompt="Single-pass baseline medical image analysis",
    )
    print_section("SINGLE-PASS OUTPUT", result.response)
    print(f"\nSingle-pass time: {result.execution_time:.2f}s")
    print(f"Single-pass usage: {result.usage_summary.to_dict()}")
    return result.response


def run_single_pass_hf(model: HfLocalMedGemma, images: list[str]) -> str:
    single_prompt = (
        "Analyze this medical image set and provide:\n"
        "1) key findings,\n"
        "2) likely abnormalities,\n"
        "3) initial diagnosis and differential diagnosis,\n"
        "4) one-line confidence estimate."
    )
    start = time.perf_counter()
    response = model.completion(single_prompt, images)
    elapsed = time.perf_counter() - start
    print_section("SINGLE-PASS OUTPUT", response)
    print(f"\nSingle-pass time: {elapsed:.2f}s")
    return response


def run_recursive_refinement(rvlm: RVLM, images: list[str], rounds: int) -> str:
    analysis = ""
    for stage in range(1, rounds + 1):
        if stage == 1:
            prompt = (
                "Analyze this medical image set/record and describe key findings, abnormalities, "
                "and your initial diagnosis."
            )
        elif stage == 2:
            prompt = (
                "Review your previous analysis and refine it.\n\n"
                f"Previous analysis:\n{analysis}\n\n"
                "Zoom in on suspicious regions (lesions, fractures, consolidations, lines/tubes). "
                "Correct possible errors, refine differential diagnosis, and provide confidence "
                "scores (0-100) for top differentials."
            )
        else:
            prompt = (
                "Based on all prior refinements, produce a final structured report.\n\n"
                f"Previous analysis:\n{analysis}\n\n"
                "Include:\n"
                "- final prioritized differential,\n"
                "- what additional tests/follow-up imaging are recommended,\n"
                "- what finding would change management most urgently."
            )

        result = rvlm.completion(
            prompt=prompt,
            images=images,
            root_prompt=f"Recursive medical refinement stage {stage}/{rounds}",
        )
        analysis = result.response
        print_section(f"RECURSIVE STAGE {stage}/{rounds}", analysis)
        print(f"\nStage {stage} time: {result.execution_time:.2f}s")
        print(f"Stage {stage} usage: {result.usage_summary.to_dict()}")

    return analysis


def run_recursive_refinement_hf(model: HfLocalMedGemma, images: list[str], rounds: int) -> str:
    analysis = ""
    for stage in range(1, rounds + 1):
        if stage == 1:
            prompt = (
                "Analyze this medical image set/record and describe key findings, abnormalities, "
                "and your initial diagnosis."
            )
        elif stage == 2:
            prompt = (
                "Review your previous analysis and refine it.\n\n"
                f"Previous analysis:\n{analysis}\n\n"
                "Zoom in on suspicious regions (lesions, fractures, consolidations, lines/tubes). "
                "Correct possible errors, refine differential diagnosis, and provide confidence "
                "scores (0-100) for top differentials."
            )
        else:
            prompt = (
                "Based on all prior refinements, produce a final structured report.\n\n"
                f"Previous analysis:\n{analysis}\n\n"
                "Include:\n"
                "- final prioritized differential,\n"
                "- what additional tests/follow-up imaging are recommended,\n"
                "- what finding would change management most urgently."
            )

        start = time.perf_counter()
        analysis = model.completion(prompt, images)
        elapsed = time.perf_counter() - start
        print_section(f"RECURSIVE STAGE {stage}/{rounds}", analysis)
        print(f"\nStage {stage} time: {elapsed:.2f}s")

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare single-pass vs recursive MedGemma-style analysis."
    )
    parser.add_argument(
        "--image",
        default=SAMPLE_IMAGE,
        help="Image source (URL, local path, or data URI).",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Optional list of image sources. Overrides --image.",
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "openai", "gemini", "litellm", "hf_local"],
        help="Backend to route requests through.",
    )
    parser.add_argument(
        "--model",
        default="medgemma-4b-it",
        help="Model name/ID accepted by the selected backend.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (commonly used for vLLM or gateways).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override (otherwise env vars are used).",
    )
    parser.add_argument(
        "--hf-model-path",
        default=None,
        help="Local model path for --backend hf_local. Defaults to --model.",
    )
    parser.add_argument(
        "--hf-max-new-tokens",
        type=int,
        default=512,
        help="Generation length for --backend hf_local.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token for gated/private repos. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of recursive refinement rounds.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="RVLM max REPL iterations per round.",
    )
    parser.add_argument(
        "--skip-single-pass",
        action="store_true",
        help="Skip baseline run and only execute recursive refinement.",
    )
    parser.add_argument(
        "--use-mimic",
        action="store_true",
        help="Load images from local MIMIC-CXR CSV/images via examples.mimic_example.",
    )
    parser.add_argument(
        "--mimic-split",
        default="validate",
        choices=["train", "validate"],
        help="MIMIC split to use with --use-mimic.",
    )
    parser.add_argument(
        "--mimic-subject",
        type=int,
        default=None,
        help="MIMIC subject_id. If omitted, uses first row in split.",
    )
    parser.add_argument(
        "--mimic-study-index",
        type=int,
        default=-1,
        help="MIMIC study index: -1=most recent, 0=earliest, etc.",
    )
    args = parser.parse_args()

    images: list[str]
    if args.use_mimic:
        try:
            from examples.mimic_example import get_patient_row, load_csv, select_images
        except ImportError as exc:
            print("Error: MIMIC mode requires dependencies used by examples.mimic_example.")
            print(f"Import failure: {exc}")
            print("Install requirements (e.g., pandas) and try again.")
            sys.exit(1)

        df = load_csv(args.mimic_split)
        subject_id = args.mimic_subject
        if subject_id is None:
            subject_id = int(df.iloc[0]["subject_id"])
            print(f"No --mimic-subject provided; using first subject: {subject_id}")
        row = get_patient_row(df, subject_id)
        images, view_types, _ = select_images(row, study_index=args.mimic_study_index)
        print(
            f"Loaded MIMIC subject {subject_id}, study {args.mimic_study_index}: "
            f"{len(images)} image(s), views={view_types}"
        )
    elif args.images:
        images = args.images
    else:
        images = [args.image]

    try:
        validate_image_sources(images)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if args.backend == "hf_local":
        model_path = args.hf_model_path or args.model
        hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        hf_model = HfLocalMedGemma(
            model_name_or_path=model_path,
            max_new_tokens=args.hf_max_new_tokens,
            hf_token=hf_token,
        )
        if not args.skip_single_pass:
            run_single_pass_hf(hf_model, images)
        final_recursive = run_recursive_refinement_hf(hf_model, images, args.rounds)
    else:
        logger = RLMLogger(log_dir="./logs/medgemma")
        rvlm = RVLM(
            backend=args.backend,
            backend_kwargs=build_backend_kwargs(args),
            environment="local",
            max_depth=1,
            max_iterations=args.max_iterations,
            logger=logger,
            verbose=True,
        )

        if not args.skip_single_pass:
            run_single_pass(rvlm, images)

        final_recursive = run_recursive_refinement(rvlm, images, args.rounds)
    print_section("FINAL RECURSIVE OUTPUT", final_recursive)


if __name__ == "__main__":
    main()
