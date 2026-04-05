"""Smoke tests for RVLM package imports."""


def test_import_rlm():
    """RLM class can be imported."""
    from rvlm import RLM

    assert RLM is not None


def test_import_rvlm():
    """RVLM class can be imported."""
    from rvlm import RVLM

    assert RVLM is not None


def test_import_recursion_router():
    """RecursionRouter can be imported."""
    from rvlm import RecursionRouter

    assert RecursionRouter is not None


def test_import_inference_cache():
    """InferenceCache can be imported."""
    from rvlm import InferenceCache

    assert InferenceCache is not None


def test_import_logger():
    """RLMLogger can be imported."""
    from rvlm.logger import RLMLogger

    assert RLMLogger is not None


def test_import_evaluation():
    """Evaluation module can be imported."""
    from rvlm.evaluation import evaluate

    assert evaluate is not None
