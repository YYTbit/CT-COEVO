"""
grader.py — Grading interface for CT-COEVO.

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

    # Find metric.py: try data_dir/utils/metric.py, then parent/../utils/metric.py
    data_dir = Path(data_dir)
    metric_candidates = [
        data_dir / "utils" / "metric.py",
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

    # Find answers: try data/private/answers.csv
    answer_candidates = [
        data_dir / "private" / "answers.csv",
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

        # Grade
        if hasattr(mod, "grade"):
            score = float(mod.grade(submission, answers))
            return score, "ok"
        elif hasattr(mod, "grade_with_norm_score"):
            result = mod.grade_with_norm_score(submission, answers)
            return float(result.get("raw_score", 0)), "ok"
        else:
            return None, "no_grade_function"

    except Exception as e:
        return None, f"error:{str(e)}"
