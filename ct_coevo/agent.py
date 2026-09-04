"""
agent.py - CT-COEVO agent

The agent runs a two-phase lifecycle:

  - Forward exploration: for each step the agent executes
        Memory Retrieval -> Tool Selection -> Experimentation ->
        Insight Distillation,
    producing a submission.csv for the environment grader. Only operational
    feedback (format checks, runtime errors) is visible during this phase;
    evaluation scores are never shown to the LLM.
  - Backward evolution: after task completion the objective
    scores are revealed and the full trajectory is distilled into reusable
    assets: trajectory analysis, memory evolution
    and tool evolution.

Similarity thresholds default to 0.8 for retrieval, tool matching, and
merging, with a 0.9 decay used when crediting tools; similarities come
from the Harrier-OSS (0.6B) embedding model (see embedding.py).
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
from .embedding import HarrierEmbedder
from .prompts import (
    build_retrieval_subset_prompt,
    build_step_plan_prompt,
    build_tool_candidate_config_prompt,
    build_create_tool_prompt,
    build_debug_fix_prompt,
    build_experimental_summary_prompt,
    build_tool_trace_prompt,
    build_step_insight_prompt,
    build_submission_critique_prompt,
    build_exec_evaluation_prompt,
    build_success_heuristic_prompt,
    build_neutral_operation_prompt,
    build_antipattern_prompt,
)

TOOL_OUTPUT_MAX_CHARS = 50000
MAX_TOKENS = 16384

# Tunable parameters
TAU1 = 0.8      # similarity threshold for Experiential title retrieval
TAU2 = 0.8      # similarity threshold for tool matching
TAU3 = 0.8      # similarity threshold for memory merging
GAMMA = 0.9     # decay factor when crediting tool calls
MAX_DEBUG_ROUNDS = 3  # cap on the error-driven fix rounds per tool run
RETRIEVAL_MAX_C = 20  # cap on summaries offered to the LLM (C candidates)
RETRIEVAL_MAX_K = 8   # cap on expanded bodies (top-K items)
MAX_TASK_DESC_CHARS = 8000  # task description length fed to the LLM


class CTCoEvoAgent:
    """
    CT-COEVO agent with context-tool co-evolution.

    Forward exploration executes
        Memory Retrieval -> Tool Selection -> Experimentation ->
        Insight Distillation
    and backward evolution turns execution traces and extracted context into
    reusable memory and toolkit assets.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        api_key: str = "",
        model: str = "deepseek-ai/DeepSeek-V3.2",
        base_url: str = "https://api.example.com/v1",
        timeout_sec: int = 86400,
        evolve: bool = True,
        state_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        harrier_model: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.timeout_sec = timeout_sec
        self.evolve = evolve
        self.phase = "evo" if evolve else "eval"

        # Paths
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_tag = f"{dataset_name}-{ts}"
        base_dir = Path.cwd()
        self.log_dir = Path(log_dir) if log_dir else base_dir / "ct_coevo" / "log" / run_tag
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Workspace isolation: symlink public files into the workspace
        self.workspace_dir = Path(workspace_dir) if workspace_dir else base_dir / "ct_coevo" / "workspace" / run_tag
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._setup_workspace()

        # Global state (read-only reference) and per-task working copies
        self.global_state_dir = Path(state_dir) if state_dir else base_dir / "ct_coevo" / "state" / "global"
        global_memory_dir = self.global_state_dir / "memory"
        global_toolkit_dir = self.global_state_dir / "toolkit"

        state_base = base_dir / "ct_coevo" / "state" / run_tag
        memory_dir = state_base / "memory"
        toolkit_dir = state_base / "toolkit"

        if global_memory_dir.exists():
            shutil.copytree(str(global_memory_dir), str(memory_dir), dirs_exist_ok=True)
        else:
            memory_dir.mkdir(parents=True, exist_ok=True)
        if global_toolkit_dir.exists():
            shutil.copytree(str(global_toolkit_dir), str(toolkit_dir), dirs_exist_ok=True)
        else:
            toolkit_dir.mkdir(parents=True, exist_ok=True)

        self.memory = HierarchicalMemory(str(memory_dir))
        self.toolkit = ScalableToolkit(str(toolkit_dir))

        # Bootstrap base tools if the toolkit is empty
        if self.toolkit.count() == 0:
            self._init_default_tools()

        self.local_memory_dir = memory_dir
        self.local_toolkit_dir = toolkit_dir
        self.global_memory_dir = global_memory_dir
        self.global_toolkit_dir = global_toolkit_dir

        # LLM client and semantic embedder (Harrier-OSS 0.6B)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
        self.model = model
        self.embedder = HarrierEmbedder(harrier_model)

        # Runtime state
        self.current_step = 0
        self.best_score = 0.0
        self.start_time = time.time()
        self._last_insight: str = ""          # insight carried over from the previous step
        self._event_pos = 0                   # global position of tool calls / submissions
        self._submission_events: List[Dict] = []  # {pos, score, step}
        self._tool_events: List[Dict] = []    # {pos, tool_id, name, success, error, timeout}

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------
    def _setup_workspace(self):
        """Symlink the public data files into the isolated workspace."""
        for f in self.data_dir.iterdir():
            dst = self.workspace_dir / f.name
            if dst.exists():
                continue
            if f.is_file():
                os.symlink(str(f), str(dst))
            elif f.is_dir() and f.name not in ("__pycache__", ".git"):
                os.symlink(str(f), str(dst), target_is_directory=True)

    def _init_default_tools(self):
        """Seed the base tools with a bash execution primitive."""
        bash_tool = ToolItem(
            name="bash",
            tool_type=ToolType.BASE,
            description="Execute shell commands in the workspace.",
            source_code="",
        )
        self.toolkit.add(bash_tool)

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------
    def _llm(self, prompt: str, temperature: float = 0.3,
             system: Optional[str] = None, max_tokens: int = MAX_TOKENS) -> str:
        """A single LLM completion with retries."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        for attempt in range(10):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] empty response, waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [Retry {attempt+1}/10] API error: {e}, waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError("API failed after 10 retries")

    def _extract_json_object(self, content: str) -> Optional[Dict]:
        """Extract a JSON object from an LLM response."""
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        json_str = content[start:end + 1]
        import re
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed = re.sub(r'(?<!\\)\n', '\\n', json_str)
            fixed = re.sub(r'(?<!\\)\t', '\\t', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None

    # ==================================================================
    # Forward Exploration
    # ==================================================================
    def run(self) -> Dict[str, Any]:
        """Run forward exploration, then backward evolution in evo mode."""
        print(f"\n{'='*60}")
        print(f"CT-COEVO Agent: {self.dataset_name}")
        print(f"Phase: {self.phase}")
        print(f"Memory: {self.memory.count()} | Toolkit: {self.toolkit.count()}")
        print(f"Workspace: {self.workspace_dir}")
        print(f"{'='*60}\n")

        # Task description q
        desc_path = self.data_dir / "data" / "public" / "description.md"
        task_description = ""
        if desc_path.exists():
            task_description = desc_path.read_text(encoding="utf-8")
        else:
            desc_path_alt = self.workspace_dir / "description.md"
            if desc_path_alt.exists():
                task_description = desc_path_alt.read_text(encoding="utf-8")
        q = task_description[:MAX_TASK_DESC_CHARS]

        original_cwd = os.getcwd()
        os.chdir(str(self.workspace_dir))

        try:
            while True:
                self.current_step += 1
                elapsed = time.time() - self.start_time
                if elapsed > self.timeout_sec:
                    print(f"[CT-COEVO] Total time limit reached after {elapsed/60:.1f}min")
                    break
                remaining = self.timeout_sec - elapsed
                print(f"\n--- Step {self.current_step} "
                      f"(elapsed: {elapsed/60:.1f}min, remaining: {remaining/60:.1f}min) ---")

                # ---- Memory Retrieval ----
                context_bodies = self._memory_retrieval(q)
                step_plan = self._step_planning(q, context_bodies)

                # ---- Tool Selection ----
                selected = self._tool_selection(q, step_plan)

                # ---- Experimentation ----
                if selected is None:
                    print("  [Step] no tool produced this step; retrying next step")
                    continue
                tool, config = selected
                outcome = self._run_tool_with_debug(tool, config, q, remaining)

                # ---- Insight Distillation ----
                results = [outcome] if outcome is not None else []
                insight = self._insight_distillation(results, q)
                self._last_insight = insight if insight else self._last_insight

                # Termination is handled by the time budget at the loop head
        finally:
            os.chdir(original_cwd)

        # ---- Backward Evolution (evolution mode only) ----
        if self.evolve:
            self._backward_evolution(q)
            self._merge_to_global()
        else:
            print("\n[Eval mode] Memory and toolkit are frozen; no evolution.")

        elapsed = time.time() - self.start_time
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
            "submission_events": self._submission_events,
        }
        results_path = self.log_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {results_path}")
        print(f"Best score: {self.best_score:.4f}")
        return results_dict

    # ------------------------------------------------------------------
    # 1. Memory Retrieval
    # ------------------------------------------------------------------
    def _memory_retrieval(self, q: str) -> List[str]:
        """Retrieve relevant Experiential context for the current task.

        The task description, extended with the previous-step insight when one
        exists, is embedded and matched against the titles of Experiential
        entries with the Harrier-OSS model. Entries above the retrieval
        threshold are returned, the LLM selects the relevant subset, and the
        selected full bodies are passed back for planning.
        """
        query = q
        if self._last_insight:
            query += f"\nPREVIOUS STEP INSIGHT:\n{self._last_insight}"

        try:
            retrieved = self.memory.retrieve_experiential(query, self.embedder, tau1=TAU1)
        except Exception as e:
            raise RuntimeError(
                "Memory retrieval needs the Harrier-OSS (0.6B) embedder. "
                "Install sentence-transformers and point --harrier-model at the "
                "model id or a local copy (or set HARRIER_MODEL_DIR).") from e

        if not retrieved:
            print("  [Retrieval] no Experiential entry above the threshold (0.8)")
            return []

        summaries = [i.summary for i in retrieved[:RETRIEVAL_MAX_C]]
        try:
            content = self._llm(build_retrieval_subset_prompt(summaries, self._last_insight),
                                temperature=0.2, max_tokens=500)
            parsed = self._extract_json_object(content)
            indices = parsed.get("selected_indices", []) if isinstance(parsed, dict) else []
            indices = [int(i) for i in indices if isinstance(i, (int, float)) or str(i).isdigit()]
        except Exception:
            indices = []
        if not indices:
            indices = list(range(min(RETRIEVAL_MAX_K, len(retrieved))))

        bodies = [retrieved[i].body for i in indices if 0 <= i < len(retrieved)][:RETRIEVAL_MAX_K]
        print(f"  [Retrieval] {len(retrieved)} matched above the threshold, expanded {len(bodies)} bodies")
        return bodies

    def _step_planning(self, q: str, context_bodies: List[str]) -> str:
        """Produce the step plan from the task, the previous insight, and
        the expanded memory bodies."""
        return self._llm(build_step_plan_prompt(q, self._last_insight, context_bodies),
                         temperature=0.3, max_tokens=800)

    # ------------------------------------------------------------------
    # 2. Tool Selection
    # ------------------------------------------------------------------
    def _tool_similarity_text(self, tool: ToolItem) -> str:
        """The text key of a tool is its description plus its execution log."""
        key = f"{tool.name}: {tool.description}"
        # Execution memory entries are titled "Exec: <tool_name>"
        for item in self.memory.items:
            if item.label == MemoryLabel.EXECUTION and item.title == f"Exec: {tool.name}":
                key += "\n" + item.body[-3000:]
                break
        return key

    def _matching_tools(self, q: str, step_plan: str) -> List[ToolItem]:
        """Return the algorithm tools (global + temporary) whose description,
        together with its recent execution log, scores above the matching
        threshold against the task and the current step plan."""
        candidates = self.toolkit.list_tools(ToolType.GLOBAL) + self.toolkit.list_tools(ToolType.TEMPORARY)
        if not candidates:
            return []
        query = (q + "\nSTEP PLAN:\n" + step_plan)[:4000]
        keys = [self._tool_similarity_text(t)[:2000] for t in candidates]
        try:
            sims = self.embedder.similarity_matrix([query], keys)[0]
        except Exception as e:
            raise RuntimeError(
                "Tool matching needs the Harrier-OSS (0.6B) embedder. "
                "Install sentence-transformers and point --harrier-model at the "
                "model id or a local copy (or set HARRIER_MODEL_DIR).") from e
        matched = [t for t, s in zip(candidates, sims) if float(s) > TAU2]
        print(f"  [Selection] {len(matched)}/{len(candidates)} tools above the matching threshold (0.8)")
        return matched

    def _tool_selection(self, q: str, step_plan: str) -> Optional[Tuple[ToolItem, Dict[str, Any]]]:
        """Select the tool for this step.

        If an algorithmic tool from the global or temporary toolkit matched,
        the LLM emits a configuration for it; otherwise a brand-new tool is
        created through the meta path and placed into the temporary toolkit.
        """
        k_cand = self._matching_tools(q, step_plan)

        if k_cand:
            base_tools = [f"- {t.tool_id}: {t.name} ({t.tool_type.value}) - {t.description}"
                          for t in self.toolkit.list_tools(ToolType.BASE)]
            candidates = [f"- {t.tool_id}: {t.name} ({t.tool_type.value}) - {t.description}"
                          for t in k_cand]
            prompt = build_tool_candidate_config_prompt(
                q, step_plan, candidates, base_tools)
            try:
                content = self._llm(prompt, temperature=0.3, max_tokens=2000)
                parsed = self._extract_json_object(content)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                tool = self.toolkit.get(str(parsed.get("tool_id", "")))
                if tool is not None:
                    config = dict(parsed.get("config") or {})
                    # base tools carry their own code inside the configuration
                    if tool.tool_type == ToolType.BASE and not config.get("code"):
                        print("  [Selection] base tool selected without code; skipping step")
                        return None
                    return tool, config
            print("  [Selection] no valid configuration produced; retrying next step")
            return None

        # Meta fallback: create a new algorithmic tool
        print("  [Selection] no matching tool; creating one via the meta path")
        tool = self._create_tool_via_meta(q, step_plan)
        if tool is None:
            return None
        return tool, {}

    def _create_tool_via_meta(self, q: str, step_plan: str) -> Optional[ToolItem]:
        """Let the LLM create a new tool (name, description, code) in the
        temporary toolkit."""
        try:
            content = self._llm(build_create_tool_prompt(q, step_plan),
                                temperature=0.5, max_tokens=MAX_TOKENS)
            parsed = self._extract_json_object(content)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            return None
        name = str(parsed.get("name") or f"tool_{int(time.time())}")[:80]
        code = str(parsed.get("code") or "")
        desc = str(parsed.get("description") or f"Algorithmic tool for {self.dataset_name}")[:500]
        code = self._fix_data_paths(code)
        if not code:
            return None
        tool = ToolItem(name=name, tool_type=ToolType.TEMPORARY, description=desc, source_code=code)
        self.toolkit.add(tool)
        print(f"  [Meta] created temporary tool {name} ({tool.tool_id})")
        return tool

    def _fix_data_paths(self, code: str) -> str:
        """Align LLM-generated code with the workspace data layout."""
        code = code.replace("./input/", "data/public/")
        code = code.replace("./data/", "data/")
        if "train.csv" in code and "data/public/train.csv" not in code:
            code = code.replace("train.csv", "data/public/train.csv")
        if "test.csv" in code and "data/public/test.csv" not in code:
            code = code.replace("test.csv", "data/public/test.csv")
        return code

    # ------------------------------------------------------------------
    # 3. Experimentation
    # ------------------------------------------------------------------
    def _run_tool_with_debug(
        self,
        tool: ToolItem,
        config: Dict[str, Any],
        q: str,
        remaining: float,
    ) -> Optional[Tuple[ToolItem, Dict, bool, str, Optional[float]]]:
        """Execute the tool; on failure the error log is handed back to the
        LLM, which keeps fixing the tool until it runs successfully.

        Each fix rewrites the tool's stored code in place. Only the final run
        is recorded as the tool's execution event, so a tool whose code
        eventually succeeds is audited as a clean successful call, and a tool
        that never succeeds is recorded once with its error flag set.
        """
        last_run = None
        for round_no in range(MAX_DEBUG_ROUNDS + 1):
            success, output, score, has_error, timed_out = self._execute_tool(tool, config, remaining)
            last_run = (success, output, score, has_error, timed_out)
            if success or timed_out or round_no == MAX_DEBUG_ROUNDS:
                break
            print(f"  [Debug] round {round_no + 1}: tool failed, asking LLM to fix")
            fixed = None
            try:
                content = self._llm(build_debug_fix_prompt(q, tool.name, tool.source_code, output),
                                    temperature=0.3, max_tokens=MAX_TOKENS)
                if "```python" in content:
                    content = content.split("```python")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                fixed = self._fix_data_paths(content.strip())
            except Exception as e:
                print(f"  [Debug] LLM fix failed: {e}")
            if not fixed:
                break
            tool.source_code = fixed
        success, output, score, has_error, timed_out = last_run
        self._record_tool_result(tool, success, has_error, timed_out, score)
        return tool, config, success, output, score

    def _execute_tool(
        self,
        tool: ToolItem,
        config: Dict[str, Any],
        remaining: float,
    ) -> Tuple[bool, str, Optional[float], bool, bool]:
        """Execute a tool in the workspace.

        The process runs to completion, or until the remaining task budget
        elapses (in which case the run is marked as timed out). Returns
        (success, output, internal_score, has_error, timed_out); the score is
        for bookkeeping only and is never shown to the agent.
        """
        if tool.tool_type == ToolType.BASE:
            code = config.get("code", tool.source_code)
        else:
            code = tool.source_code

        is_bash = tool.name == "bash"
        ext = ".sh" if is_bash else ".py"
        tool_path = self.workspace_dir / f"tool_{tool.tool_id}{ext}"
        tool_path.write_text(code, encoding="utf-8")

        cmd = ["bash", str(tool_path)] if is_bash else [sys.executable, "-u", str(tool_path)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.workspace_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        timed_out = False
        try:
            stdout, _ = proc.communicate(timeout=max(1.0, remaining))
            output = stdout[:TOOL_OUTPUT_MAX_CHARS]
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            try:
                stdout, _ = proc.communicate(timeout=1)
            except Exception:
                stdout = ""
            output = (stdout or "")[:TOOL_OUTPUT_MAX_CHARS]
            output += "\n[REVIEW] Tool exceeded the remaining budget and was terminated."
            timed_out = True

        has_error = (proc.returncode != 0 or (not output.strip())) and not timed_out
        sub_path = self.workspace_dir / "submission.csv"
        has_sub = sub_path.exists()

        # Internal grading; the result is used only for backward evolution.
        score = None
        if has_sub and not timed_out:
            from .grader import grade_submission
            try:
                score, grade_status = grade_submission(
                    dataset_name=self.dataset_name,
                    submission_path=str(sub_path),
                    data_dir=str(self.data_dir),
                )
                if grade_status != "ok":
                    score = None
            except Exception:
                score = None

        success = (not has_error) and has_sub and score is not None
        return success, output, score, has_error, timed_out

    # ------------------------------------------------------------------
    # 4. Insight Distillation
    # ------------------------------------------------------------------
    def _insight_distillation(
        self,
        results: List[Tuple[ToolItem, Dict, bool, str, Optional[float]]],
        q: str,
    ) -> str:
        """Summarize the step's Experimental record and tool traces into a
        step insight for the next iteration."""
        if not results:
            return ""
        step_output = "\n\n".join(
            f"Tool: {tool.name} ({tool.tool_type.value})\nSuccess: {success}\nOutput:\n{out[:5000]}"
            for tool, config, success, out, score in results
        )

        # summary of the Experimental record (body -> summary)
        try:
            summary = self._llm(build_experimental_summary_prompt(step_output),
                                temperature=0.2, max_tokens=600)
        except Exception:
            summary = step_output[:1500]

        expt_item = MemoryItem(
            title=f"Step {self.current_step}: {[t.name for t, _, _, _, _ in results]}",
            summary=summary,
            body=step_output[:50000],
            label=MemoryLabel.EXPERIMENTAL,
        )
        self.memory.add(expt_item)
        print(f"  Memory: Expt.={self.memory.count(MemoryLabel.EXPERIMENTAL)}")

        # per-tool analysis (log -> trace), appended to the tool Execution entry
        traces = []
        for tool, config, success, out, score in results:
            if tool.tool_type not in (ToolType.GLOBAL, ToolType.TEMPORARY):
                continue
            try:
                trace = self._llm(build_tool_trace_prompt(out),
                                  temperature=0.2, max_tokens=400)
            except Exception:
                trace = f"{tool.name}: {'success' if success else 'failure'} - {out[:400]}"
            traces.append(trace)
            self._append_exec_memory(tool, trace, success)

        # synthesize the step insight from the summary and the traces
        try:
            insight = self._llm(build_step_insight_prompt(summary, traces),
                                temperature=0.2, max_tokens=500)
        except Exception:
            insight = summary
        return insight

    def _append_exec_memory(self, tool: ToolItem, trace: str, success: bool):
        """Append the analytical trace to the tool's Execution item."""
        exec_title = f"Exec: {tool.name}"
        existing = None
        for item in self.memory.items:
            if item.label == MemoryLabel.EXECUTION and item.title == exec_title:
                existing = item
                break
        if existing:
            existing.body += f"\n\n--- Run {existing.body.count('--- Run') + 1} ---\n{trace}"
            existing.summary = existing.summary + " | " + trace if existing.summary else trace
            self.memory.update(existing)
        else:
            item = MemoryItem(title=exec_title, summary=trace, body=trace,
                              label=MemoryLabel.EXECUTION)
            self.memory.add(item)

        if success:
            tool.success_count += 1
        else:
            tool.failure_count += 1

    # ==================================================================
    # Backward Evolution
    # ==================================================================
    def _backward_evolution(self, q: str):
        """After task completion the objective scores are revealed
        (evolution mode only); runs trajectory analysis, memory evolution,
        and tool evolution."""
        print(f"\n{'='*60}")
        print("Backward evolution ...")
        self._record_submission_events()
        if not self._submission_events:
            print("  No graded submission available; skipping evolution.")
            return

        print("=== Score trajectory ===")
        for e in self._submission_events:
            print(f"  pos={e['pos']} score={e['score']:.6f} delta={e['delta']:+.6f}")

        # --- Trajectory Analysis: score deltas and step critiques ---
        critiques = self._trajectory_analysis(q)

        # --- Memory Evolution ---
        self._memory_evolution(q, critiques)

        # --- Tool Evolution ---
        self._tool_evolution()

    def _record_submission_events(self):
        """Turn the internally graded submissions into a chronological
        timeline with consecutive score deltas; the first submission is
        measured against an initial score of zero."""
        events = sorted([e for e in self._submission_events if e.get("score") is not None],
                        key=lambda e: e["pos"])
        prev = 0.0
        for e in events:
            e["delta"] = float(e["score"]) - prev
            prev = float(e["score"])
        self._submission_events = events

    def _trajectory_analysis(self, q: str) -> List[str]:
        """Ask the LLM to critique every submission from its producing steps
        and its score delta."""
        critiques: List[str] = []
        prev_pos = -1
        for e in self._submission_events:
            subtrajectory = self._tool_events_text(prev_pos, e["pos"])
            prev_pos = e["pos"]
            try:
                c = self._llm(build_submission_critique_prompt(
                    q, subtrajectory, e["delta"]),
                    temperature=0.2, max_tokens=600)
            except Exception:
                c = f"Submission at position {e['pos']} changed score by {e['delta']:+.4f}."
            critiques.append(c)
        return critiques

    def _tool_events_text(self, pos_from: int, pos_to: int) -> str:
        lines = []
        for ev in self._tool_events:
            if ev["pos"] > pos_from and ev["pos"] <= pos_to:
                lines.append(f"- [{ev['pos']}] {ev['name']}: "
                             f"success={ev['success']} error={ev['error']} timeout={ev['timeout']}")
        return "\n".join(lines) or "(no tool events in this window)"

    def _memory_evolution(self, q: str, critiques: List[str]):
        """
        Memory evolution:

          1. Each tool gets an evaluation summary synthesized from its
             execution history, the score delta, and the critique, and its
             Execution entry is refreshed with it.
          2. Experiential heuristics are distilled conditioned on the sign of
             the score delta (success / neutral / regression), then merged
             with highly similar entries.
        """
        # --- Execution-memory evaluation summaries ---
        for item in list(self.memory.items):
            if item.label != MemoryLabel.EXECUTION:
                continue
            tool_name = item.title[len("Exec: "):] if item.title.startswith("Exec: ") else item.title
            idx = self._submission_index_for_tool(tool_name)
            if idx is None:
                continue
            ev = self._submission_events[idx]
            try:
                summary = self._llm(build_exec_evaluation_prompt(
                    item.body, ev["delta"], critiques[idx] if idx < len(critiques) else ""),
                    temperature=0.2, max_tokens=400)
                if summary:
                    item.summary = summary
                    self.memory.update(item)
            except Exception:
                pass

        # --- Experiential distillation (three-way by delta sign) ---
        expt_items = self.memory.items_of(MemoryLabel.EXPERIMENTAL)
        prev_step = 0
        for i, e in enumerate(self._submission_events):
            delta = e["delta"]
            evidence = self._evidence_for_submission(
                prev_step, e.get("step", prev_step + 1), expt_items,
                critiques[i] if i < len(critiques) else "")
            prev_step = e.get("step", prev_step + 1)
            if not evidence.strip():
                continue
            if delta > 0:
                prompt = build_success_heuristic_prompt(evidence, delta)      # p_succ
            elif delta == 0:
                prompt = build_neutral_operation_prompt(evidence)             # p_neut
            else:
                prompt = build_antipattern_prompt(evidence, delta)            # p_anti
            try:
                body = self._llm(prompt, temperature=0.2, max_tokens=800)
            except Exception:
                body = ""
            if not body:
                continue
            # merge with similar entries (absorbing duplicates) before storing
            try:
                item = self.memory.merge_experiential(body, self.embedder, tau3=TAU3,
                                                      llm_client=self.client, model=self.model)
                print(f"  [Evol] Exper. += {item.title}")
            except Exception as exc:
                print(f"  [Evol] Experiential merge failed: {exc}")

    def _submission_index_for_tool(self, tool_name: str) -> Optional[int]:
        """Index of the submission that follows the tool's last recorded call."""
        call_positions = [ev["pos"] for ev in self._tool_events if ev["name"] == tool_name]
        if not call_positions:
            return None
        last_call = max(call_positions)
        for i, e in enumerate(self._submission_events):
            if e["pos"] > last_call:
                return i
        return len(self._submission_events) - 1

    def _evidence_for_submission(self, step_from: int, step_to: int,
                                 expt_items: List[MemoryItem], critique: str) -> str:
        """Experimental summaries recorded by the steps that produced the
        submission (steps in (step_from, step_to]), plus its critique."""
        parts = [f"CRITIQUE: {critique}"] if critique else []
        for item in expt_items:
            step = self._step_of(item)
            if step is not None and step_from < step <= step_to:
                parts.append(f"- {item.title}: {item.summary}")
        return "\n".join(parts)[:8000]

    def _step_of(self, item: MemoryItem) -> Optional[int]:
        """Parse the step number from an Experimental record title
        (titles have the form 'Step N: <tool names>')."""
        title = item.title or ""
        if title.startswith("Step "):
            rest = title[5:]
            num = rest.split(":")[0].strip()
            if num.isdigit():
                return int(num)
        return None

    # ------------------------------------------------------------------
    # Tool Evolution
    # ------------------------------------------------------------------
    def _tool_evolution(self):
        """Score every tool call and update the toolkit.

        Each successful call is credited with the decayed score deltas of all
        later submissions (decay 0.9 per position). Averaging the credit over
        a tool's successful calls gives its task score; subtracting the mean
        over all tools gives the relative score. A tool is retained iff its
        relative score is positive and none of its calls errored or timed out:
        qualifying temporary tools are promoted to the global toolkit, failing
        global tools are pruned, and the temporary toolkit is cleared
        afterwards.
        """
        submissions = self._submission_events
        if not submissions:
            return

        def contribution(pos: int) -> float:
            total = 0.0
            for e in submissions:
                L = e["pos"]
                if L > pos:
                    total += (GAMMA ** (L - pos)) * e["delta"]
            return total

        # per-tool statistics over successful calls
        scores: Dict[str, List[float]] = {}
        for ev in self._tool_events:
            if not ev["success"]:
                continue
            scores.setdefault(ev["name"], []).append(contribution(ev["pos"]))

        if not scores:
            print("  [ToolEvol] no successful calls to score; skipping")
            return

        means = [sum(v) / len(v) for v in scores.values()]
        global_mean = sum(means) / len(means)

        algorithm_tools = (self.toolkit.list_tools(ToolType.GLOBAL)
                           + self.toolkit.list_tools(ToolType.TEMPORARY))
        promote_ids, prune_ids = [], []
        for tool in algorithm_tools:
            vals = scores.get(tool.name, [])
            if not vals:
                # A tool without a successful call in this task has no audit
                # score (s_task is undefined), so it is neither credited nor
                # pruned here.
                print(f"  [ToolEvol] {tool.name}: no successful call in this task; skipped")
                continue
            s_task = sum(vals) / len(vals)
            a_task = s_task - global_mean
            tool_events = [ev for ev in self._tool_events if ev["name"] == tool.name]
            no_error = all(not ev["error"] for ev in tool_events)
            no_timeout = all(not ev["timeout"] for ev in tool_events)
            retain = a_task > 0 and no_error and no_timeout
            print(f"  [ToolEvol] {tool.name} ({tool.tool_type.value}): "
                  f"s_task={s_task:+.6f} a_task={a_task:+.6f} retain={retain}")
            if retain and tool.tool_type == ToolType.TEMPORARY:
                promote_ids.append(tool.tool_id)
            elif not retain and tool.tool_type == ToolType.GLOBAL:
                prune_ids.append(tool.tool_id)

        # apply the pruning set, then promote the survivors
        for tid in prune_ids:
            t = self.toolkit.get(tid)
            if t:
                self.toolkit.prune(tid)
                print(f"  [ToolEvol] pruned {t.name}")
        for tid in promote_ids:
            t = self.toolkit.get(tid)
            if t:
                self.toolkit.promote_to_global(tid)
                print(f"  [ToolEvol] promoted {t.name} -> global")

        # the temporary toolkit is cleared after each task
        cleared = [t.name for t in self.toolkit.list_tools(ToolType.TEMPORARY)]
        self.toolkit.clear_temporary()
        if cleared:
            print(f"  [ToolEvol] cleared temporary tools: {cleared}")

    def _merge_to_global(self):
        """Copy local memory/toolkit back to the global state (evo mode only)."""
        print("  [Merge] Copying local -> global...")
        self.global_memory_dir.mkdir(parents=True, exist_ok=True)
        self.global_toolkit_dir.mkdir(parents=True, exist_ok=True)
        for f in self.local_memory_dir.glob("*.md"):
            dst = self.global_memory_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
                print(f"  [Merge] New memory: {f.name}")
        for f in self.local_toolkit_dir.glob("*.json"):
            shutil.copy2(str(f), str(self.global_toolkit_dir / f.name))
        print(f"  [Merge] Global memory: {len(list(self.global_memory_dir.glob('*.md')))} items")
        print(f"  [Merge] Global toolkit: {len(list(self.global_toolkit_dir.glob('*.json')))} meta")

    # ------------------------------------------------------------------
    # Event bookkeeping
    # ------------------------------------------------------------------
    def _record_tool_result(self, tool: ToolItem, success: bool, has_error: bool,
                            timed_out: bool, score: Optional[float]):
        """Record a finished tool call with its position and flags."""
        self._event_pos += 1
        pos = self._event_pos
        self._tool_events.append({
            "pos": pos, "tool_id": tool.tool_id, "name": tool.name,
            "success": success, "error": has_error, "timeout": timed_out,
        })
        if success and score is not None and score > self.best_score:
            self.best_score = score
        if success and score is not None:
            self._submission_events.append({"pos": pos, "score": score, "step": self.current_step})
