"""
Evaluation metrics for RVLM outputs.

Provides numeric scoring of model responses against ground truth references.
All metrics return floats in [0, 1] where 1 is a perfect match.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics computed on a single prediction."""

    exact_match: float = 0.0
    f1: float = 0.0
    bleu: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    cosine_similarity: float = 0.0
    semantic_similarity: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        d = {
            "exact_match": self.exact_match,
            "f1": self.f1,
            "bleu": self.bleu,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "rouge_l": self.rouge_l,
            "cosine_similarity": self.cosine_similarity,
            "semantic_similarity": self.semantic_similarity,
        }
        d.update(self.extra)
        return d

    def __repr__(self) -> str:
        parts = [f"{k}={v:.4f}" for k, v in self.to_dict().items() if isinstance(v, float)]
        return f"EvaluationResult({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def compute_exact_match(prediction: str, reference: str) -> float:
    """Binary exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# BLEU (up to 4-gram, with brevity penalty)
# ---------------------------------------------------------------------------

def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Corpus-level BLEU score (single reference). Returns float in [0, 1]."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens:
        return 0.0

    # Clipped n-gram precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    # If any precision is 0, BLEU is 0
    if any(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)

    # Brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))

    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# ROUGE (1, 2, L)
# ---------------------------------------------------------------------------

def _rouge_n(prediction: str, reference: str, n: int) -> float:
    """ROUGE-N F1 score."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    pred_ngrams = _get_ngrams(pred_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)

    if not ref_ngrams or not pred_ngrams:
        return 0.0

    overlap = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
    precision = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0.0
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length via DP."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimized: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns:
        Dict with keys 'rouge_1', 'rouge_2', 'rouge_l'.
    """
    return {
        "rouge_1": _rouge_n(prediction, reference, 1),
        "rouge_2": _rouge_n(prediction, reference, 2),
        "rouge_l": _rouge_l(prediction, reference),
    }


# ---------------------------------------------------------------------------
# Cosine Similarity (bag-of-words TF)
# ---------------------------------------------------------------------------

def compute_cosine_similarity(prediction: str, reference: str) -> float:
    """Bag-of-words cosine similarity using term frequencies."""
    pred_tokens = Counter(_tokenize(prediction))
    ref_tokens = Counter(_tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    all_words = set(pred_tokens) | set(ref_tokens)
    dot = sum(pred_tokens.get(w, 0) * ref_tokens.get(w, 0) for w in all_words)
    mag_pred = math.sqrt(sum(v ** 2 for v in pred_tokens.values()))
    mag_ref = math.sqrt(sum(v ** 2 for v in ref_tokens.values()))

    if mag_pred == 0 or mag_ref == 0:
        return 0.0
    return dot / (mag_pred * mag_ref)


# ---------------------------------------------------------------------------
# Semantic Similarity (optional, requires sentence-transformers)
# ---------------------------------------------------------------------------

def compute_semantic_similarity(prediction: str, reference: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Cosine similarity of sentence embeddings.

    Requires `pip install sentence-transformers`. Returns 0.0 if not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return 0.0

    model = SentenceTransformer(model_name)
    embeddings = model.encode([prediction, reference], convert_to_numpy=True)

    dot = float(sum(a * b for a, b in zip(embeddings[0], embeddings[1])))
    mag_a = math.sqrt(float(sum(x ** 2 for x in embeddings[0])))
    mag_b = math.sqrt(float(sum(x ** 2 for x in embeddings[1])))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    # Clamp to [0, 1] since sentence similarity can be slightly negative
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


# ---------------------------------------------------------------------------
# Unified evaluate()
# ---------------------------------------------------------------------------

def evaluate(
    prediction: str,
    reference: str,
    metrics: list[str] | None = None,
) -> EvaluationResult:
    """Compute all (or selected) evaluation metrics.

    Args:
        prediction: Model output text.
        reference: Ground truth text.
        metrics: Optional list of metric names to compute. If None, computes all.
            Valid names: 'exact_match', 'f1', 'bleu', 'rouge', 'cosine', 'semantic'.

    Returns:
        EvaluationResult with numeric scores.
    """
    all_metrics = {"exact_match", "f1", "bleu", "rouge", "cosine", "semantic"}
    selected = set(metrics) if metrics else all_metrics

    result = EvaluationResult()

    if "exact_match" in selected:
        result.exact_match = compute_exact_match(prediction, reference)

    if "f1" in selected:
        result.f1 = compute_f1(prediction, reference)

    if "bleu" in selected:
        result.bleu = compute_bleu(prediction, reference)

    if "rouge" in selected:
        rouge = compute_rouge(prediction, reference)
        result.rouge_1 = rouge["rouge_1"]
        result.rouge_2 = rouge["rouge_2"]
        result.rouge_l = rouge["rouge_l"]

    if "cosine" in selected:
        result.cosine_similarity = compute_cosine_similarity(prediction, reference)

    if "semantic" in selected:
        result.semantic_similarity = compute_semantic_similarity(prediction, reference)

    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    predictions: list[str],
    references: list[str],
    metrics: list[str] | None = None,
) -> tuple[list[EvaluationResult], dict[str, float]]:
    """Evaluate a batch of predictions against references.

    Args:
        predictions: List of model outputs.
        references: List of ground truth texts (same length as predictions).
        metrics: Optional list of metric names to compute.

    Returns:
        Tuple of (per-sample results, averaged scores dict).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) must have the same length"
        )

    results = [evaluate(pred, ref, metrics) for pred, ref in zip(predictions, references)]

    # Average across all samples
    if not results:
        return results, {}

    keys = list(results[0].to_dict().keys())
    averages = {}
    for key in keys:
        values = [r.to_dict()[key] for r in results if isinstance(r.to_dict()[key], (int, float))]
        averages[f"avg_{key}"] = sum(values) / len(values) if values else 0.0

    return results, averages
