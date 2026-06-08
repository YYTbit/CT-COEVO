"""
Evolution Loop
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '/data/yangyingtao02')

from ct_coevo.memory import HierarchicalMemory, MemoryItem, MemoryLabel
from ct_coevo.toolkit import ScalableToolkit, ToolItem, ToolType

# Shared state directories (persist across runs)
MEMORY_DIR = "/data/yangyingtao02/ct_coevo/state/global/memory"
TOOLKIT_DIR = "/data/yangyingtao02/ct_coevo/state/global/toolkit"

def write_evolved_memories():
    """Write evolved memories from debugging experience."""

    memory = HierarchicalMemory(MEMORY_DIR)

    # Check if memories already exist
    if memory.count() > 0:
        print(f"Memory already has {memory.count()} items, skipping seed")
        return

    print("Writing evolved memories...")

    # Memory 1: Data location
    memory.add(MemoryItem(
        title="Data files are in workspace root, not ./input/",
        summary="When generating code, data files (train.csv, test.csv, sample_submission.csv, etc.) are symlinked to the workspace root directory. Use pd.read_csv('file.csv') not pd.read_csv('./input/file.csv').",
        body="The workspace setup symlinks all public data files to the workspace root. The LLM-generated code often assumes ./input/ directory (from the paper's Listing 2 prompt) but the actual data is at ./ level. This causes FileNotFoundError. Fix: always read from current directory.",
        label=MemoryLabel.EXPERIENTIAL,
    ))

    # Memory 2: Submission format
    memory.add(MemoryItem(
        title="Submission must match sample_submission.csv exactly",
        summary="submission.csv must have the same columns and row count as sample_submission.csv. Read sample_submission.csv first to understand the required format.",
        body="Common errors: (1) Wrong column names, (2) Wrong row count, (3) Wrong data types. Always read sample_submission.csv at the start and ensure output matches exactly. Use pd.read_csv('sample_submission.csv') to check format.",
        label=MemoryLabel.EXPERIENTIAL,
    ))

    # Memory 3: Error handling
    memory.add(MemoryItem(
        title="Wrap code in try-except and print errors",
        summary="Generated code often fails silently. Wrap main logic in try-except and print full traceback for debugging.",
        body="The LLM-generated code may have import errors, column name mismatches, or data type issues. Always wrap in try-except Exception as e: print(traceback.format_exc()) to capture errors.",
        label=MemoryLabel.EXPERIENTIAL,
    ))

    # Memory 4: Simple baseline first
    memory.add(MemoryItem(
        title="Start with simple baseline, then iterate",
        summary="First step should be a simple baseline (e.g., popularity-based or mean prediction) to establish a valid submission. Then iterate with more complex models.",
        body="Complex models often fail on first try. A simple baseline (e.g., predict mean of target, or top-N popular items) ensures a valid submission.csv exists. Then improve incrementally.",
        label=MemoryLabel.EXPERIENTIAL,
    ))

    # Memory 5: Check data before coding
    memory.add(MemoryItem(
        title="Always inspect data structure first",
        summary="Before writing any code, check: (1) column names, (2) data types, (3) row counts, (4) sample_submission format.",
        body="Use pd.read_csv('file.csv').head() and .dtypes and .shape to understand data. Check sample_submission.csv for required output format. This prevents column name errors and format mismatches.",
        label=MemoryLabel.EXPERIENTIAL,
    ))

    # Memory 6: AUC needs both classes
    memory.add(MemoryItem(
        title="AUC metric requires predictions with variance",
        summary="If all predictions are the same value (e.g., all 0.5), AUC is undefined (0.5 or error). Ensure predictions have variance.",
        body="For binary classification tasks (CTR, repeat purchase), the model must produce varying predictions. A constant prediction (e.g., all 0.5) gives AUC=0.5 which is effectively random. Use a real model, not a constant.",
        label=MemoryLabel.EXPERIENTIAL,
    ))

    # Memory 7: Large data sampling
    memory.add(MemoryItem(
        title="Sample large datasets for speed",
        summary="transactions.csv can be very large (85M rows). Use nrows parameter or sample to avoid memory/time issues.",
        body="For initial exploration, use pd.read_csv('transactions.csv', nrows=1000000) to sample. For production, use chunked reading or parquet. The agent has limited time per step.",
        label=MemoryLabel.EXPERIENTIAL,
    ))

    # Memory 8: LightGBM for tabular
    memory.add(MemoryItem(
        title="LightGBM is reliable for tabular recommendation",
        summary="For CTR/binary classification tasks, LightGBM with proper feature engineering is a strong and reliable baseline.",
        body="LightGBM handles mixed feature types well, trains fast, and gives good results. Use it as a default baseline for tabular data. Key params: n_estimators=100, learning_rate=0.1, num_leaves=31.",
        label=MemoryLabel.EXPERIENTIAL,
    ))

    print(f"Written {memory.count()} evolved memories")


def write_evolved_tools():
    """Write evolved tools from debugging experience."""

    toolkit = ScalableToolkit(TOOLKIT_DIR)

    if toolkit.count() > 0:
        print(f"Toolkit already has {toolkit.count()} items, skipping seed")
        return

    print("Writing evolved tools...")

    # Tool 1: Simple baseline tool
    baseline_code = '''import pandas as pd
import numpy as np
import traceback

try:
    # Read data
    sample_sub = pd.read_csv("sample_submission.csv")
    print(f"Sample submission: {sample_sub.shape}, columns: {list(sample_sub.columns)}")

    # Check for train data
    import os
    train_files = [f for f in os.listdir(".") if "train" in f.lower() and f.endswith(".csv")]
    print(f"Train files: {train_files}")

    # Build submission matching sample format
    submission = sample_sub.copy()

    # Try to find target column and compute baseline
    for tf in train_files:
        train = pd.read_csv(tf)
        print(f"  {tf}: {train.shape}, columns: {list(train.columns)}")

        # Find numeric target columns
        numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col in submission.columns:
                mean_val = train[col].mean()
                submission[col] = mean_val
                print(f"  Baseline: {col} = {mean_val:.6f}")

    # Save
    submission.to_csv("submission.csv", index=False)
    print(f"Submission saved: {submission.shape}")
    print(f"First rows:\\n{submission.head()}")

except Exception as e:
    print(f"ERROR: {e}")
    print(traceback.format_exc())
'''

    toolkit.add(ToolItem(
        name="simple_baseline",
        tool_type=ToolType.GLOBAL,
        description="Simple baseline: reads sample_submission, fills with mean/mode predictions. Always produces valid submission.",
        source_code=baseline_code,
    ))

    print(f"Written {toolkit.count()} evolved tools")


def run_evolution_cycle():
    """Run one evolution cycle: agent run + memory update."""

    from ct_coevo.agent import CTCoEvoAgent

    datasets = [
        "acquire_valued_shoppers_challenge",
        "airbnb_recruiting_new_user_bookings",
        "avazu_ctr_prediction",
    ]

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Running: {dataset}")
        print(f"{'='*60}")

        data_dir = f"/data/yangyingtao02/new-rec-bench/novelrecbench/competitions/{dataset}/data/public"

        if not Path(data_dir).exists():
            print(f"  Data dir not found: {data_dir}")
            continue

        agent = CTCoEvoAgent(
            dataset_name=dataset,
            data_dir=data_dir,
            api_key=os.environ.get("CT_API_KEY", ""),
            model="mimo-v2.5-pro",
            base_url=os.environ.get("CT_API_URL", "https://api.siliconflow.cn/v1"),
            max_steps=5,
            timeout_sec=600,
            evolve=False,
            state_dir=f"/data/yangyingtao02/ct_coevo/state/global",
        )

        result = agent.run()
        print(f"  Best score: {result['best_score']}")
        print(f"  Memory: {result['memory_counts']}")


if __name__ == "__main__":
    # Initialize evolved memories and tools
    write_evolved_memories()
    write_evolved_tools()

    # Run evolution cycles
    for cycle in range(20):  # 20 cycles
        print(f"\n{'#'*60}")
        print(f"EVOLUTION CYCLE {cycle + 1}/20")
        print(f"{'#'*60}")

        run_evolution_cycle()

        print(f"\nCycle {cycle + 1} complete. Sleeping 60s...")
        time.sleep(60)
