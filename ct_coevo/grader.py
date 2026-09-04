"""
grader.py - Grading interface for CT-COEVO.

Loads metric.py from RecDevBench and grades submissions.
"""

import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def grade_submission(
    dataset_name: str,
    submission_path: str,
    data_dir: str,
) -> Tuple[Optional[float], str]:
    """
    Grade a submission using the dataset's metric.py.

    The returned score is on the normalized scale of the benchmark, where
    larger values are better. This is the convention used by the backward
    evolution phase, whose score deltas and tool credits assume higher scores
    indicate improvement. Datasets whose metric.py only exposes a raw grading
    function fall back to it, keeping the interface usable for any dataset.

    Args:
        dataset_name: Name of the dataset
        submission_path: Path to submission.csv
        data_dir: Path to data/public directory

    Returns:
        (score, status) tuple
    """
    sub_path = Path(submission_path)
    if not sub_path.exists():
        return None, "submission_missing"

    # Find metric.py: try multiple paths
    data_dir = Path(data_dir)
    metric_candidates = [
        data_dir / "utils" / "metric.py",
        data_dir / "data" / "utils" / "metric.py",
        data_dir.parent / "utils" / "metric.py",
        data_dir.parent.parent / "utils" / "metric.py",
    ]

    metric_path = None
    for cand in metric_candidates:
        if cand.exists():
            metric_path = cand
            break

    if metric_path is None:
        return None, "metric_py_not_found"

    # Find answers: try multiple paths
    answer_candidates = [
        data_dir / "data" / "private" / "answers.csv",
        data_dir / "private" / "answers.csv",
        data_dir.parent / "data" / "private" / "answers.csv",
        data_dir.parent / "private" / "answers.csv",
        data_dir / "answers.csv",
        data_dir.parent / "answers.csv",
    ]

    answer_path = None
    for cand in answer_candidates:
        if cand.exists():
            answer_path = cand
            break

    if answer_path is None:
        return None, "answers_not_found"

    try:
        # Load metric module
        spec = importlib.util.spec_from_file_location(f"metric_{dataset_name}", str(metric_path))
        if spec is None or spec.loader is None:
            return None, "metric_load_failed"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Load data
        submission = pd.read_csv(sub_path)
        answers = pd.read_csv(answer_path)

        # Prefer the normalized grading functions, whose score is larger-is-
        # better by construction; fall back to raw grading otherwise.
        for fn_name in ("grade_with_dacode", "grade_with_norm_score"):
            if hasattr(mod, fn_name):
                result = getattr(mod, fn_name)(submission, answers)
                if isinstance(result, dict):
                    for key in ("da_code", "norm_score", "normalized_score"):
                        if result.get(key) is not None:
                            return float(result[key]), "ok"
                    if result.get("raw_score") is not None:
                        return float(result["raw_score"]), "ok"
                return float(result), "ok"
        if hasattr(mod, "grade"):
            score = float(mod.grade(submission, answers))
            return score, "ok"
        elif hasattr(mod, "grade_raw_score"):
            score = float(mod.grade_raw_score(submission, answers))
            return score, "ok"
        else:
            return None, "no_grade_function"

    except Exception as e:
        return None, f"error:{str(e)}"
