TOOL_OUTPUT_MAX_CHARS = 500000  # Global: max chars for tool output (user specified 500000)

"""
agent.py — CT-COEVO Agent

Strictly follows the paper:
  - Uses all 11 prompt templates from prompts.py
  - Workspace isolation: copy public files to workspace
  - Multi-tool parallel call: agent can call multiple tools per step
  - Co-evolution loop: Eq. 1-4

Agent loop (Paper Eq. 1-4):
  E_t = Extract(q, o_{t-1}; M)
  (τ_t, θ_t) = π(q, E_t; K)
  o_t = Experiment(τ_t, θ_t; D)
  M ← M ∪ {Distill(q, τ_t, θ_t, o_t)}
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .memory import HierarchicalMemory, MemoryItem, MemoryLabel
from .toolkit import ScalableToolkit, ToolItem, ToolType
from .prompts import (
    build_context_descriptor_prompt,
    build_initial_solution_prompt,
    build_debugging_prompt,
    build_memory_guided_improvement_prompt,
    build_experimental_memory_synthesis_prompt,
    build_experiential_memory_distillation_prompt,
    build_memory_extraction_prompt,
    build_research_plan_prompt,
    build_tool_review_prompt,
    build_meta_tool_revision_prompt,
    build_post_task_audit_prompt,
)


class CTCoEvoAgent:
    """
    CT-COEVO Agent: Self-evolving recommendation agent with context-tool co-evolution.

    Paper Eq. 1-4:
      E_t = Extract(q, o_{t-1}; M)
      (τ_t, θ_t) = π(q, E_t; K)
      o_t = Experiment(τ_t, θ_t; D)
      M ← M ∪ {Distill(q, τ_t, θ_t, o_t)}
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        api_key: str = "",
        model: str = "deepseek-ai/DeepSeek-V3.2",
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout_sec: int = 86400,
        evolve: bool = True,
        state_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        workspace_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.timeout_sec = timeout_sec
        self.evolve = evolve
        self.phase = "evo" if evolve else "eval"

        # Paths
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_tag = f"{dataset_name}-{ts}"
        self.log_dir = Path(log_dir) if log_dir else Path(f"/data/yangyingtao02/ct_coevo/log/{run_tag}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Workspace isolation: copy public files to workspace
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path(f"/data/yangyingtao02/ct_coevo/workspace/{run_tag}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._setup_workspace()

        # Global state (read-only reference)
        self.global_state_dir = Path(state_dir) if state_dir else Path(f"/data/yangyingtao02/ct_coevo/state/global")
        global_memory_dir = self.global_state_dir / "memory"
        global_toolkit_dir = self.global_state_dir / "toolkit"

        # Local state (per-task working copy)
        state_base = Path(f"/data/yangyingtao02/ct_coevo/state/{run_tag}")
        memory_dir = state_base / "memory"
        toolkit_dir = state_base / "toolkit"

        # Copy global → local
        import shutil
        if global_memory_dir.exists():
            shutil.copytree(str(global_memory_dir), str(memory_dir), dirs_exist_ok=True)
        else:
            memory_dir.mkdir(parents=True, exist_ok=True)
        if global_toolkit_dir.exists():
            shutil.copytree(str(global_toolkit_dir), str(toolkit_dir), dirs_exist_ok=True)
        else:
            toolkit_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components from local copies
        self.memory = HierarchicalMemory(str(memory_dir))
        self.toolkit = ScalableToolkit(str(toolkit_dir))

        # Save paths for later consolidation
        self.local_memory_dir = memory_dir
        self.local_toolkit_dir = toolkit_dir
        self.global_memory_dir = global_memory_dir
        self.global_toolkit_dir = global_toolkit_dir

        # LLM client
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
        self.model = model

        # Checkpoint path
        self.checkpoint_path = self.workspace_dir / "checkpoint.json"

        # State (will be loaded from checkpoint if exists)
        self.current_step = 0
        self.best_score = 0.0
        self.start_time = time.time()
        self.tried_tools: List[Dict] = []
        self._elapsed_offset = 0.0  # accumulated time from previous runs

        # Try to resume from checkpoint
        self._load_checkpoint()

        # Message log for debugging
        self.message_log_path = self.log_dir / "message_log.jsonl"

    def _log_message(self, event_type: str, data: Dict[str, Any]):
        """Log a message to the message log file."""
        import json
        entry = {
            "timestamp": time.time(),
            "step": self.current_step,
            "event": event_type,
            "data": data,
        }
        try:
            with open(self.message_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _setup_workspace(self):
        """
        Workspace isolation: symlink public files to workspace.
        Paper: "start by copying the public folder to workspace"
        """
        for f in self.data_dir.iterdir():
            dst = self.workspace_dir / f.name
            if dst.exists():
                continue
            if f.is_file():
                os.symlink(str(f), str(dst))
            elif f.is_dir() and f.name not in ("__pycache__", ".git"):
                os.symlink(str(f), str(dst), target_is_directory=True)

    def _load_checkpoint(self):
        """Load checkpoint from previous run to resume."""
        if not self.checkpoint_path.exists():
            return
        try:
            with open(self.checkpoint_path, "r") as f:
                ckpt = json.load(f)
            self.current_step = ckpt.get("current_step", 0)
            self.best_score = ckpt.get("best_score", 0.0)
            self.tried_tools = ckpt.get("tried_tools", [])
            self._elapsed_offset = ckpt.get("elapsed", 0.0)
            print(f"[Checkpoint] Resumed from step {self.current_step}, "
                  f"best_score={self.best_score:.6f}, "
                  f"elapsed={self._elapsed_offset/60:.1f}min")
        except Exception as e:
            print(f"[Checkpoint] Failed to load: {e}")

    def _save_checkpoint(self):
        """Save checkpoint for resume."""
        elapsed = self._elapsed_offset + (time.time() - self.start_time)
        ckpt = {
            "current_step": self.current_step,
            "best_score": self.best_score,
            "tried_tools": self.tried_tools,
            "elapsed": elapsed,
            "timeout_sec": self.timeout_sec,
            "dataset": self.dataset_name,
            "phase": self.phase,
        }
        try:
            with open(self.checkpoint_path, "w") as f:
                json.dump(ckpt, f, indent=2)
        except Exception as e:
            print(f"[Checkpoint] Failed to save: {e}")

    def run(self) -> Dict[str, Any]:
        """
        Main agent loop. Implements Paper Eq. 1-4.
        """
        elapsed_so_far = self._elapsed_offset
        remaining = self.timeout_sec - elapsed_so_far

        print(f"\n{'='*60}")
        print(f"CT-COEVO Agent: {self.dataset_name}")
        print(f"Phase: {self.phase}")
        print(f"Memory: {self.memory.count()} | Toolkit: {self.toolkit.count()}")
        print(f"Workspace: {self.workspace_dir}")
        if self.current_step > 0:
            print(f"Resumed: step {self.current_step}, elapsed {elapsed_so_far/60:.1f}min, remaining {remaining/60:.1f}min")
        print(f"{'='*60}\n")

        # Read dataset description
        desc_path = self.workspace_dir / "description.md"
        task_description = ""
        if desc_path.exists():
            task_description = desc_path.read_text(encoding="utf-8")

        # Generate task descriptor (Listing 1)
        task_descriptor = self._generate_task_descriptor(task_description)

        # Change to workspace
        original_cwd = os.getcwd()
        os.chdir(str(self.workspace_dir))

        try:
            # Main loop - no step limit, only total time limit
            prev_output = ""
            while True:
                self.current_step += 1
                elapsed = self._elapsed_offset + (time.time() - self.start_time)

                if elapsed > self.timeout_sec:
                    print(f"[CT-COEVO] Total time limit reached after {elapsed/60:.1f}min")
                    break

                remaining = self.timeout_sec - elapsed
                print(f"\n--- Step {self.current_step} "
                      f"(elapsed: {elapsed/60:.1f}min, remaining: {remaining/60:.1f}min) ---")

                # Step 1: Extract context from memory (Listing 7, Paper Eq. 1)
                context_indices = self._extract_context(task_description)

                # Step 2: Select tools (Paper Eq. 2)
                # Multi-tool parallel call: agent can return multiple tool calls
                try:
                    tool_calls = self._select_tools(context_indices, task_description, previous_output=prev_output)
                except Exception as e:
                    print(f"  Tool selection failed: {e}")
                    print(f"  Skipping step {self.current_step}, will retry next iteration")
                    self.current_step -= 1  # Don't count this step
                    time.sleep(10)
                    continue

                if not tool_calls:
                    print("  No tools selected, creating initial solution...")
                    tool = self._create_initial_tool(task_description)
                    if tool:
                        tool_calls = [(tool, {})]

                # Step 3: Execute tools (Paper Eq. 3)
                # Multi-tool parallel call: execute each tool sequentially
                results = []
                for tool, config in tool_calls:
                    # For base tools, generate code using LLM
                    if tool.tool_type == ToolType.BASE and "code" not in config:
                        code = self._generate_code(tool.name, task_description)
                        config["code"] = code

                    print(f"  Executing tool: {tool.name} ({tool.tool_type.value})")
                    success, output, score = self._execute_tool(tool, config)
                    results.append((tool, config, success, output, score))

                    # Step 4: Distill into memory (Listing 5, Paper Eq. 4)
                    self._distill_memory(tool, config, task_description, output, success, score)

                    if success and score is not None and score > self.best_score:
                        self.best_score = score
                        print(f"  New best score: {score:.4f}")

                    self.tried_tools.append({
                        "tool_id": tool.tool_id,
                        "tool_name": tool.name,
                        "config": config,
                        "success": success,
                        "score": score,
                    })

                    # Update prev_output for next step's context
                    prev_output += f"\n[Step {self.current_step}] Tool: {tool.name}, Success: {success}, Score: {score}\n{output}\n"

                # Save checkpoint after each step
                self._save_checkpoint()

            # Post-task consolidation (Listing 11, Paper Section 4.2)
            if self.evolve:
                print(f"\n{'='*60}")
                print("Post-task consolidation...")
                self._consolidate(task_description)
                # Copy local → global
                self._merge_to_global()

        finally:
            os.chdir(original_cwd)

        elapsed = self._elapsed_offset + (time.time() - self.start_time)
        results_dict = {
            "dataset": self.dataset_name,
            "phase": self.phase,
            "steps_completed": self.current_step,
            "best_score": self.best_score,
            "elapsed_seconds": elapsed,
            "memory_counts": {
                "Exper.": self.memory.count(MemoryLabel.EXPERIENTIAL),
                "Expt.": self.memory.count(MemoryLabel.EXPERIMENTAL),
                "Exec.": self.memory.count(MemoryLabel.EXECUTION),
            },
            "toolkit_counts": {
                "global": self.toolkit.count(ToolType.GLOBAL),
                "temp": self.toolkit.count(ToolType.TEMPORARY),
                "base": self.toolkit.count(ToolType.BASE),
                "meta": self.toolkit.count(ToolType.META),
            },
            "tried_tools": self.tried_tools,
        }

        results_path = self.log_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {results_path}")
        print(f"Best score: {self.best_score:.4f}")
        return results_dict

    def _generate_task_descriptor(self, task_description: str) -> str:
        """Listing 1: Generate compact task descriptor for context retrieval."""
        prompt = build_context_descriptor_prompt(task_description)
        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] _generate_task_descriptor API empty, waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] _generate_task_descriptor API error: {e}, waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError("_generate_task_descriptor: API failed after 10 retries")

    def _generate_code(self, tool_name: str, task_description: str) -> str:
        """Generate code for base tools (python, bash) using LLM."""
        # Read sample submission for format
        sample_path = self.workspace_dir / "sample_submission.csv"
        sample_info = ""
        if sample_path.exists():
            import pandas as pd
            sample = pd.read_csv(sample_path)
            sample_info = f"Sample submission: columns={list(sample.columns)}, rows={len(sample)}"

        # List available files
        files = [f.name for f in self.workspace_dir.iterdir() if f.suffix == '.csv']
        files_info = f"Available CSV files: {files}"

        prompt = f"""You are a recommendation system expert. Write Python code to solve this task.

TASK: {task_description}

{files_info}

{sample_info}

Write a complete Python script that:
1. Reads the data files (sample large files with nrows=1000000)
2. Trains a recommendation model (LightGBM, XGBoost, or similar)
3. Generates submission.csv matching the sample format
4. Prints the evaluation metric

Output ONLY the Python code, no explanation:"""

        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=131072,
                )
                code = response.choices[0].message.content
                if not code or not code.strip():
                    wait = min(5 * (attempt + 1), 30)
                    print(f"  [Retry {attempt+1}/10] _generate_code API empty, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                # Extract code block if present
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                return code.strip()
            except Exception as e:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] _generate_code API error: {e}, waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError("_generate_code: API failed after 10 retries")

    def _generate_training_code(self) -> str:
        """Generate PyTorch training code using LLM when create_tool has no code."""
        # Read sample submission for format
        sample_path = self.workspace_dir / "sample_submission.csv"
        sample_info = ""
        if sample_path.exists():
            import pandas as pd
            sample = pd.read_csv(sample_path, nrows=3)
            sample_info = f"Sample submission columns: {list(sample.columns)}, first row: {sample.iloc[0].to_dict()}"

        # Get dataset info
        desc_path = self.workspace_dir / "description.md"
        task_desc = desc_path.read_text()[:2000] if desc_path.exists() else ""

        # List workspace files
        files = [f.name for f in self.workspace_dir.iterdir() if f.is_file() and not f.name.startswith('tool_')]

        prompt = f"""Write complete PyTorch training code for this recommendation task.

TASK: {task_desc[:1000]}

FILES IN WORKSPACE: {files}

{sample_info}

REQUIREMENTS:
1. Use PyTorch with CUDA (torch.cuda) for GPU training
2. Read data from current directory (train.json, test.json)
3. Train a recommendation model
4. Generate submission.csv in the correct format
5. Print evaluation metrics
6. Handle large datasets with sampling/chunking if needed

Write a COMPLETE, self-contained Python script. Output ONLY the code, no explanation."""

        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=16000,
                )
                code = response.choices[0].message.content or ""
                if not code.strip():
                    wait = min(5 * (attempt + 1), 30)
                    print(f"  [Retry {attempt+1}/10] _generate_training_code API empty, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                return code.strip()
            except Exception as e:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] _generate_training_code API error: {e}, waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError("_generate_training_code: API failed after 10 retries")

    def _extract_context(self, task_description: str) -> List[int]:
        """
        Listing 7: Extract relevant context from hierarchical memory.
        Paper Eq. 1: E_t = Extract(q, o_{t-1}; M)

        Returns indices of relevant memory items.
        Falls back to all Experiential items if LLM extraction fails.

        Architecture: separate stable/dynamic parts for cache hit rate.
        """
        if self.memory.count() == 0:
            return []

        # Get all items with titles and summaries
        previews = self.memory.get_titles_and_summaries()

        # Try LLM-based selection first
        preview_text = "\n".join(
            f"{i}. [{p['label']}] {p['title']}: {p['summary']}"
            for i, p in enumerate(previews)
        )

        # NOTE: mimo-v2.5 API returns empty when system + user message is too long
        # So we use a single user message, keeping it under 2000 chars
        preview_text = "\n".join(
            f"{i}. [{p['label']}] {p['title']}: {p['summary'][:100]}"
            for i, p in enumerate(previews)
        )

        prompt = f"""Select relevant memory items for next step.

MEMORY:
{preview_text[:1000]}

Step: {self.current_step}

Return JSON: {{"selected_indices": [1, 3, 5]}}"""

        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500,
                )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    wait = min(5 * (attempt + 1), 30)
                    print(f"  [Retry {attempt+1}/10] _extract_context API empty, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if "{" in content:
                    json_str = content[content.index("{"):content.rindex("}") + 1]
                    result = json.loads(json_str)
                    indices = result.get("selected_indices", [])
                    if indices:
                        print(f"  Context: {len(indices)} items selected via LLM")
                        return indices
                # Response parsed but no valid indices - retry
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] _extract_context no valid indices in response, waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] _extract_context API error: {e}, waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError("_extract_context: API failed after 10 retries")

    def _select_tools(
        self,
        context_indices: List[int],
        task_description: str,
        previous_output: str = "",
    ) -> List[Tuple[ToolItem, Dict[str, Any]]]:
        """
        Paper Eq. 2: (τ_t, θ_t) = π(q, E_t; K)
        Multi-tool parallel call: agent can return multiple tool calls.

        Architecture: separate stable/dynamic parts for cache hit rate.
        - System message (stable): task description, tool list, hardware info
        - User message (dynamic): step number, history, previous output, experience
        """
        available_tools = self.toolkit.list_tools()
        if not available_tools:
            return []

        # Build tool list
        tool_list = "\n".join(
            f"- {t.tool_id}: {t.name} ({t.tool_type.value}) - {t.description}"
            for t in available_tools
        )

        # Build context: show titles + summaries only (paper: select-then-expand)
        # Agent uses inspect_memory to read full body of selected items
        context_text = ""
        if context_indices:
            items = self.memory.items
            context_parts = []
            for i in context_indices:
                if i < len(items):
                    item = items[i]
                    context_parts.append(f"[{item.label.value}] {item.title}: {item.summary}")
            context_text = "\n".join(context_parts)

        # Find create_tool id
        create_tool_id = ""
        for t in available_tools:
            if t.name == "create_tool":
                create_tool_id = t.tool_id
                break

        # Build history summary from tried_tools (no official scores - agent can't see them)
        history_text = ""
        if hasattr(self, 'tried_tools') and self.tried_tools:
            history_lines = []
            for t in self.tried_tools:
                history_lines.append(f"- {t['tool_name']}: submission_valid={t['success']}")
            history_text = "\n".join(history_lines)

        # Build previous output section
        prev_output_section = ""
        if previous_output:
            prev_output_section = f"\n=== PREVIOUS STEP OUTPUT ===\n{previous_output}"

        # NOTE: mimo-v2.5 API returns empty when system message + user message > ~3000 chars
        # So we put everything in a single user message
        prompt = f"""You are a tool selection agent for recommendation systems. Output ONLY a JSON array.

HARDWARE: 4x NVIDIA RTX 3090 (24GB each)
TOOLS:
{tool_list}

TASK: {task_description[:300]}

STEP: {self.current_step}

EXPERIENCE:
{context_text[:500] if context_text else "None"}

HISTORY:
{history_text[:300] if history_text else "None"}
{prev_output_section[:300] if prev_output_section else ""}

Call create_tool (id={create_tool_id}) with name, code (PyTorch GPU training), description, review_time.
Or call bash to explore data.

review_time: seconds after which logs are returned to you. Tool keeps running in background.
-1 = wait forever until tool finishes. Recommended: 1800 (30min) for training, -1 for quick tasks.

Output ONLY JSON array: [{{"tool_id": "{create_tool_id}", "name": "model_v1", "code": "import torch...", "description": "desc", "review_time": 1800}}]"""

        try:
            # Log the prompt
            self._log_message("select_tools", {"prompt": prompt})

            # Retry until we get a valid response - no fallback for empty responses
            content = ""
            for attempt in range(10):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000,
                )
                content = response.choices[0].message.content or ""
                if content.strip():
                    break
                wait = min(5 * (attempt + 1), 30)  # 5, 10, 15, ..., 30
                print(f"  [Retry {attempt+1}/10] API returned empty, waiting {wait}s...")
                time.sleep(wait)

            # Log the response
            self._log_message("select_tools_response", {"content": content})

            if not content.strip():
                raise ValueError("API returned empty after 10 retries")

            # Robust JSON extraction: find first [ and last ]
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end + 1]
                # Fix newlines in code strings that break JSON parsing
                result = json.loads(json_str)
                tool_calls = []
                for item in result:
                    tool_id = item.get("tool_id")
                    tool = self.toolkit.get(tool_id)
                    if tool is None:
                        continue

                    # Extract review_time from tool call (-1 = wait forever)
                    review_time = item.get("review_time", -1)

                    # Handle create_tool: create a new algorithm tool
                    if tool.name == "create_tool":
                        new_name = item.get("name", f"tool_{int(time.time())}")
                        new_code = item.get("code", "")
                        new_desc = item.get("description", f"Algorithm tool for {self.dataset_name}")
                        if not new_code:
                            # Generate code using LLM
                            new_code = self._generate_training_code()
                        if new_code:
                            new_tool = self.toolkit.create_temporary(
                                name=new_name,
                                description=new_desc,
                                source_code=new_code,
                            )
                            print(f"  Created tool: {new_name} ({new_tool.tool_id})")
                            tool_calls.append((new_tool, {"review_time": review_time}))
                        continue

                    # Handle edit_tool: modify an existing tool and auto-execute
                    if tool.name == "edit_tool":
                        target_id = item.get("target_tool_id")
                        new_code = item.get("code")
                        if target_id and new_code:
                            target = self.toolkit.get(target_id)
                            if target:
                                target.source_code = new_code
                                print(f"  Edited tool: {target.name}")
                                # Auto-execute the edited tool
                                tool_calls.append((target, {"review_time": review_time}))
                        continue

                    # Normal tool call
                    config = item.get("config", {})
                    config["review_time"] = review_time
                    tool_calls.append((tool, config))

                if tool_calls:
                    return tool_calls
        except Exception as e:
            print(f"  Tool selection error: {e}")
            raise  # Propagate - let run() handle it

    def _create_initial_tool(self, task_description: str) -> Optional[ToolItem]:
        """Listing 2: Create initial solution when no tool exists."""
        # Read sample submission for format
        sample_path = self.workspace_dir / "sample_submission.csv"
        sample_info = ""
        if sample_path.exists():
            import pandas as pd
            sample = pd.read_csv(sample_path)
            sample_info = f"Sample submission columns: {list(sample.columns)}, rows: {len(sample)}"

        prompt = build_initial_solution_prompt(
            task_description=task_description,
            data_preview=sample_info,
            data_knowledge="Use pandas for data loading. Check columns and types.",
            model_knowledge="Start with a simple model like LightGBM or collaborative filtering.",
            installed_packages="pandas, numpy, scikit-learn, torch, lightgbm, xgboost",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=4000,
            )
            code = response.choices[0].message.content
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]

            tool = ToolItem(
                name="initial_solution",
                tool_type=ToolType.TEMPORARY,
                description=f"Initial solution for {self.dataset_name}",
                source_code=code.strip(),
            )
            self.toolkit.add(tool)
            print(f"  Created tool: {tool.tool_id}")
            return tool
        except Exception as e:
            print(f"  Tool creation error: {e}")
            return None

    def _execute_tool(self, tool: ToolItem, config: Dict[str, Any]) -> Tuple[bool, str, Optional[float]]:
        """
        Paper Eq. 3: o_t = Experiment(τ_t, θ_t; D)

        review_time: agent-defined time (seconds) after which logs are returned.
        - If tool finishes early → return immediately with full output
        - If tool runs longer → return partial output at review_time, tool keeps running in background
        """
        # For base tools (python, bash), use code from config
        if tool.tool_type == ToolType.BASE:
            code = config.get("code", tool.source_code)
        else:
            code = tool.source_code

        # Get review_time from config (agent-defined, default -1 = wait forever)
        review_time = config.get("review_time", -1)

        tool_path = self.workspace_dir / f"tool_{tool.tool_id}.py"
        tool_path.write_text(code, encoding="utf-8")

        # Use Popen to run tool in background (NOT subprocess.run which blocks)
        # Tool keeps running even after review_time
        proc = subprocess.Popen(
            [sys.executable, "-u", str(tool_path)],
            cwd=str(self.workspace_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for tool to finish or review_time
        if review_time == -1:
            # Wait forever until tool finishes
            stdout, _ = proc.communicate()
            output = stdout[:TOOL_OUTPUT_MAX_CHARS]
            success = proc.returncode == 0
        else:
            # Return partial logs at review_time, tool keeps running
            try:
                stdout, _ = proc.communicate(timeout=review_time)
                output = stdout[:TOOL_OUTPUT_MAX_CHARS]
                success = proc.returncode == 0
            except subprocess.TimeoutExpired:
                # Tool still running after review_time → return partial output, DON'T kill
                import select
                partial = ""
                try:
                    while select.select([proc.stdout], [], [], 0.1)[0]:
                        line = proc.stdout.readline()
                        if line:
                            partial += line
                        else:
                            break
                except Exception:
                    pass
                output = partial[:TOOL_OUTPUT_MAX_CHARS]
                output += f"\n[REVIEW] Tool still running after {review_time}s. Returning partial logs."
                success = True  # Not a failure, just partial

        # Check submission
        sub_path = self.workspace_dir / "submission.csv"
        has_sub = sub_path.exists()

        # Grade submission using metric.py (INTERNAL ONLY - agent does NOT see this)
        score = None
        if has_sub:
            from .grader import grade_submission
            score, grade_status = grade_submission(
                dataset_name=self.dataset_name,
                submission_path=str(sub_path),
                data_dir=str(self.data_dir),
            )
            if score is not None:
                print(f"  [Internal] score={score:.6f}")

        return success and has_sub, output, score

    def _distill_summary(self, tool_name: str, config: Dict, output: str, success: bool) -> str:
        """
        Use LLM to distill a compact, actionable summary from execution output.
        Paper: "the summary is a compact actionable takeaway"
        """
        # Truncate output for LLM
        output_snippet = output if output else "No output"
        status = "SUCCESS" if success else "FAILURE"

        prompt = f"""Summarize this tool execution in 1-2 sentences. Focus on WHAT was tried, WHAT happened, and the KEY lesson.

Tool: {tool_name}
Config: {json.dumps(config)}
Status: {status}
Output:
{output_snippet}

Write ONLY the summary, no preamble. Example formats:
- "Tried LightGBM on numeric features. Failed: KeyError on 'offer' column - column names don't match."
- "Trained DeepFM with 100 epochs. Success: submission.csv generated with 10000 rows."
"""

        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=200,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] distill_summary API empty, waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] distill_summary API error: {e}, waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError("distill_summary: API failed after 10 retries")

    def _distill_exec_trace(self, tool_name: str, config: Dict, output: str, success: bool) -> str:
        """
        Use LLM to generate a meaningful execution trace for algorithm tools.
        Paper: Exec. is per-tool performance trace.
        """
        output_snippet = output if output else "No output"
        status = "SUCCESS" if success else "FAILURE"

        prompt = f"""Write a concise execution trace for this algorithm tool run. Focus on what happened and key observations.

Tool: {tool_name}
Config: {json.dumps(config)}
Status: {status}
Output:
{output_snippet}

Write 1-2 sentences. Example:
- "Trained DeepFM 100 epochs on GPU. Loss converged to 0.45. Submission generated with 15000 rows."
- "Failed: CUDA out of memory with batch_size=4096. Need smaller batch or gradient accumulation."
"""

        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=200,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] distill_exec_trace API empty, waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] distill_exec_trace API error: {e}, waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError("distill_exec_trace: API failed after 10 retries")

    def _distill_memory(
        self,
        tool: ToolItem,
        config: Dict[str, Any],
        task_description: str,
        output: str,
        success: bool,
        score: Optional[float],
    ):
        """
        Listing 5: Synthesize Experimental memory.
        Paper Eq. 4: M ← M ∪ {Distill(q, τ_t, θ_t, o_t)}

        During execution, agent does NOT see official score.
        - Expt.: records what was tried and what happened (agent's observation)
        - Exec.: per-tool trace appended to tool record (agent's observation)
        Score is stored internally but NOT shown to agent.
        """
        status = "SUCCESS" if success else "FAILURE"

        # Use LLM to distill a compact summary from the output
        summary = self._distill_summary(tool.name, config, output, success)

        # Experimental memory: what the agent observed (no official score)
        expt_item = MemoryItem(
            title=f"{status}: {tool.name} on {self.dataset_name}",
            summary=summary,
            body=output,
            label=MemoryLabel.EXPERIMENTAL,
        )
        self.memory.add(expt_item)

        # Execution trace: ONLY for algorithm tools (global/temp), NOT base tools
        # Paper: Exec. is per-tool performance trace, written by LLM
        if tool.tool_type in (ToolType.GLOBAL, ToolType.TEMPORARY):
            exec_trace = self._distill_exec_trace(tool.name, config, output, success)
            tool.description = f"{tool.description}\n  Exec: {exec_trace}"
            if success:
                tool.success_count += 1
            else:
                tool.failure_count += 1
            print(f"  Tool {tool.name}: success={tool.success_count}, failure={tool.failure_count}")

        print(f"  Memory: Expt.={self.memory.count(MemoryLabel.EXPERIMENTAL)}")

    def _consolidate(self, task_description: str):
        """
        Post-task consolidation (Paper Section 4.2, Listing 11).

        Two consolidation axes:
        1. Memory: Expt. → distill into Exper. (transferable heuristics)
        2. Memory: Exec. → archive to corresponding tool's record
        3. Tool: effective temp tools → promote to global; underperforming → prune
        """
        print("  [Consolidation] Distilling Experimental → Experiential...")

        # 1. Distill Experimental (Expt.) into Experiential (Exper.)
        # Paper: all Expt. summaries → combined as body → LLM generates summary+title
        expt_items = [i for i in self.memory.items if i.label == MemoryLabel.EXPERIMENTAL]
        if expt_items:
            distilled = self.memory.distill_experiential(expt_items, llm_client=self.client)
            if distilled:
                self.memory.add(distilled)
                print(f"  [Consolidation] Distilled {len(expt_items)} Expt. → 1 Exper.: {distilled.title}")

        # 2. Exec. traces are already appended to tools in _distill_memory
        #    (Exec. count ≤ tool count, not memory items)

        # 3. Promote/prune temporary tools via LLM review
        temp_tools = self.toolkit.list_tools(ToolType.TEMPORARY)
        if temp_tools:
            history_text = "\n".join(
                f"- {t['tool_name']}: success={t['success']}, score={t['score']}"
                for t in self.tried_tools
            )
            prompt = build_post_task_audit_prompt(
                task_description=task_description,
                metric_name="score",
                final_score=self.best_score,
                best_score=self.best_score,
                tool_execution_history=history_text,
                experimental_entries="\n".join(f"- {i.title}" for i in expt_items[-10:]),
                temp_tools_list="\n".join(f"- {t.name}" for t in temp_tools),
            )
            content = ""
            for attempt in range(10):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=2000,
                    )
                    content = response.choices[0].message.content
                    if content and content.strip():
                        break
                    wait = min(5 * (attempt + 1), 30)
                    print(f"  [Retry {attempt+1}/10] _consolidate audit API empty, waiting {wait}s...")
                    time.sleep(wait)
                except Exception as e:
                    wait = min(5 * (attempt + 1), 30)
                    print(f"  [Retry {attempt+1}/10] _consolidate audit API error: {e}, waiting {wait}s...")
                    time.sleep(wait)
            if not content or not content.strip():
                raise RuntimeError("_consolidate: audit API failed after 10 retries")
            if "{" in content:
                json_str = content[content.index("{"):content.rindex("}") + 1]
                audit = json.loads(json_str)
                for tool_name, decision in audit.get("tool_decisions", {}).items():
                    action = decision.get("action", "keep")
                    if action == "promote":
                        for t in temp_tools:
                            if t.name == tool_name:
                                self.toolkit.promote_to_global(t.tool_id)
                                print(f"  [Consolidation] Promoted {tool_name} → global")
                    elif action == "prune":
                        for t in temp_tools:
                            if t.name == tool_name:
                                self.toolkit.prune(t.tool_id)
                                print(f"  [Consolidation] Pruned {tool_name}")

    def _merge_to_global(self):
        """
        Copy local memory/toolkit back to global (evo mode only).
        Paper: (M_t, K_t) -> (M_{t+1}, K_{t+1})
        """
        import shutil
        print("  [Merge] Copying local → global...")

        # Copy memory files (only new ones)
        for f in self.local_memory_dir.glob("*.md"):
            dst = self.global_memory_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
                print(f"  [Merge] New memory: {f.name}")

        # Copy toolkit metadata
        for f in self.local_toolkit_dir.glob("*.json"):
            dst = self.global_toolkit_dir / f.name
            shutil.copy2(str(f), str(dst))
        for f in self.local_toolkit_dir.glob("*.py"):
            dst = self.global_toolkit_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
                print(f"  [Merge] New tool: {f.name}")

        print(f"  [Merge] Global memory: {len(list(self.global_memory_dir.glob('*.md')))} items")
        print(f"  [Merge] Global toolkit: {len(list(self.global_toolkit_dir.glob('*.json')))} meta")


def run_ct_coevo(
    dataset_name: str,
    data_dir: str,
    api_key: str = "",
    model: str = "deepseek-ai/DeepSeek-V3.2",
    base_url: str = "https://api.siliconflow.cn/v1",
    timeout_sec: int = 86400,
    evolve: bool = True,
) -> Dict[str, Any]:
    """Convenience function to run CT-COEVO on any dataset."""
    agent = CTCoEvoAgent(
        dataset_name=dataset_name,
        data_dir=data_dir,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_sec=timeout_sec,
        evolve=evolve,
    )
    return agent.run()
