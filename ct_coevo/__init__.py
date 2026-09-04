"""
CT-COEVO: An Autonomous Agent for Recommender System Design
via Context-Tool Co-Evolution.

Usage:
    from ct_coevo import CTCoEvoAgent

    agent = CTCoEvoAgent(
        dataset_name="acquire_valued_shoppers_challenge",
        data_dir="/path/to/data/public",
        api_key="sk-xxx",
        model="deepseek-ai/DeepSeek-V3.2",
        base_url="https://api.xxx.com/v1",
        timeout_sec=86400,
        evolve=True,  # True=evo, False=eval
    )
    result = agent.run()
"""

from .memory import HierarchicalMemory, MemoryItem, MemoryLabel
from .toolkit import ScalableToolkit, ToolItem, ToolType
from .agent import CTCoEvoAgent
from .grader import grade_submission

__all__ = [
    "CTCoEvoAgent",
    "grade_submission",
    "HierarchicalMemory",
    "MemoryItem",
    "MemoryLabel",
    "ScalableToolkit",
    "ToolItem",
    "ToolType",
]
