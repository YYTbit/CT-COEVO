"""
CT-COEVO: An Autonomous Agent for Recommender System Design
via Context-Tool Co-Evolution

Usage:
    from ct_coevo import run_ct_coevo

    result = run_ct_coevo(
        dataset_name="acquire_valued_shoppers_challenge",
        data_dir="/path/to/data/public",
        api_key="sk-xxx",
        model="mimo-v2.5-pro",
        base_url="https://api.xxx.com/v1",
        timeout_sec=86400,
        evolve=True,  # True=evo, False=eval
    )
"""

from .memory import HierarchicalMemory, MemoryItem, MemoryLabel
from .toolkit import ScalableToolkit, ToolItem, ToolType
from .agent import CTCoEvoAgent, run_ct_coevo
from .grader import grade_submission

__all__ = [
    "CTCoEvoAgent",
    "run_ct_coevo",
    "grade_submission",
    "HierarchicalMemory",
    "MemoryItem",
    "MemoryLabel",
    "ScalableToolkit",
    "ToolItem",
    "ToolType",
]
