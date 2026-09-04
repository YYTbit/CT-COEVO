"""
prompts.py - Prompt templates for CT-COEVO.

One builder per agent action, covering forward exploration (memory
retrieval, tool selection, debugging, insight distillation) and backward
evolution (trajectory analysis, memory evolution):

    Memory Retrieval      -> build_retrieval_subset_prompt / build_step_plan_prompt
    Tool Selection        -> build_tool_candidate_config_prompt / build_create_tool_prompt
    Debugging             -> build_debug_fix_prompt
    Insight Distillation  -> build_experimental_summary_prompt /
                             build_tool_trace_prompt / build_step_insight_prompt
    Trajectory Analysis   -> build_submission_critique_prompt
    Memory Evolution      -> build_exec_evaluation_prompt,
                             one distillation prompt per score-change direction
                             (improvement / neutral / regression),
                             build_memory_merge_prompt, build_title_summary_prompt

A few builders request JSON output that the agent parses programmatically;
the rest use plain text or markdown.
"""

from typing import List

# ----------------------------------------------------------------------
# Memory Retrieval
# ----------------------------------------------------------------------
def build_retrieval_subset_prompt(
    retrieved_summaries: List[str],
    previous_insight: str,
) -> str:
    """Pick the relevant subset from the retrieved summaries; their full
    bodies are then expanded by the agent for planning.
    """
    previews = "\n".join(
        f"[{i}] {s.strip()}" for i, s in enumerate(retrieved_summaries) if s.strip()
    )
    insight = previous_insight.strip() if previous_insight else "(none)"
    return f"""You are an expert recommendation system agent. Relevant historical
heuristics (summary level) were retrieved for the current task, together with
the insight distilled at the previous step.

PREVIOUS STEP INSIGHT:
{insight}

RETRIEVED SUMMARIES:
{previews}

Select the indices of the summaries that are worth expanding into full
bodies for planning the next step. Return JSON only:
{{"selected_indices": [0, 2, ...], "reasoning": "..."}}"""

def build_step_plan_prompt(
    task_description: str,
    previous_insight: str,
    context_bodies: List[str],
) -> str:
    """Plan the current step from the task, previous insight, and
    the expanded memory bodies."""
    bodies = "\n\n".join(
        f"### Memory {i}\n{b.strip()[:1200]}" for i, b in enumerate(context_bodies) if b.strip()
    ) or "(no matching memory bodies)"
    insight = previous_insight.strip() if previous_insight else "(none)"
    return f"""You are an expert recommendation system agent planning the next
experiment for the task below.

TASK:
{task_description}

PREVIOUS STEP INSIGHT:
{insight}

RETRIEVED MEMORY BODIES:
{bodies}

Produce a concise step plan: which modeling direction to try next, which
recommendation-specific aspect it targets (e.g., negative sampling, loss
alignment with the ranking metric, temporal split, feature interaction),
and what success would look like. Output ONLY the plan as plain text, no
headings."""

# ----------------------------------------------------------------------
# Tool Selection
# ----------------------------------------------------------------------
def build_tool_candidate_config_prompt(
    task_description: str,
    step_plan: str,
    candidate_tools: List[str],
    available_base: List[str],
) -> str:
    """Emit the configuration for one matched tool.

    Only algorithm tools whose similarity with the current task and step
    plan exceeded the matching threshold, plus the base tools, are offered;
    when nothing matched, the caller routes to build_create_tool_prompt
    instead.
    """
    tools_text = "\n".join(candidate_tools) if candidate_tools else "(none)"
    base_text = "\n".join(available_base) if available_base else "(none)"
    return f"""You are a tool selection agent for recommendation systems.
Choose ONE tool and provide its configuration for this step. Output ONLY a
JSON object.

TASK:
{task_description}

STEP PLAN:
{step_plan}

MATCHED ALGORITHM TOOLS:
{tools_text}

BASE TOOLS:
{base_text}

JSON format:
{{"tool_id": "TOOL_ID", "config": {{}}}}
- The tool executes to completion in an isolated workspace.
- If you select a base tool, put the exact command/code to run into
  config.code.
- Do not invent tool ids; only use the ids listed above.
- Prefer a matched algorithmic tool over exploration with base tools."""

def build_create_tool_prompt(
    task_description: str,
    step_plan: str,
) -> str:
    """Create a brand-new algorithmic tool when no existing tool matches.

    The LLM creates a new name, description, and executable code; the new
    asset is placed into the temporary tool list.
    """
    return f"""You are a recommendation system expert. No existing tool matches
the current context, so you must create a new algorithmic tool from scratch
via the meta mechanism. Output ONLY a JSON object.

TASK:
{task_description}

STEP PLAN:
{step_plan}

JSON format:
{{"name": "tool_name", "description": "what the tool does",
  "code": "complete, self-contained Python training script that reads the
           data (paths under data/public/), trains a model, writes
           submission.csv, and prints its own validation score"}}"""

def build_debug_fix_prompt(
    task_description: str,
    tool_name: str,
    tool_code: str,
    error_output: str,
) -> str:
    """Fix a tool after an execution error."""
    return f"""You are a recommendation system expert. The execution of tool
'{tool_name}' failed. Revise ONLY the code so that it runs successfully.
Keep the modeling approach unless it is the cause of the failure.

TASK:
{task_description}

CURRENT CODE:
{tool_code}

EXECUTION OUTPUT (error):
{error_output[-6000:]}

Output ONLY the revised Python code in a markdown code block, no explanation."""

def build_experimental_summary_prompt(experimental_body: str) -> str:
    """Summarize the step's Experimental record, covering what was tried,
    what happened, and the key observation."""
    return f"""Summarize the following experimental record in 2-3 sentences:
what was tried, what happened, and the key observation.

EXPERIMENTAL RECORD:
{experimental_body[:8000]}

Output ONLY the summary, no preamble."""

def build_tool_trace_prompt(tool_output: str) -> str:
    """Analyze one tool's execution log into a short analytical trace."""
    return f"""Analyze this tool execution log and write a concise analytical
trace (1-3 sentences): what the tool did, whether it succeeded, and any
diagnostic insight for future runs.

EXECUTION LOG:
{tool_output[:6000]}

Output ONLY the trace, no preamble."""

def build_step_insight_prompt(experimental_summary: str, tool_traces: List[str]) -> str:
    """Combine the experimental summary and the tool traces into a
    step insight that will guide the next iteration."""
    traces = "\n".join(f"- {t}" for t in tool_traces if t) or "(no tool traces)"
    return f"""Synthesize the step-level insight that will guide the next step.
Combine the experimental summary with the per-tool analytical traces.

EXPERIMENTAL SUMMARY:
{experimental_summary}

TOOL TRACES:
{traces}

Output ONLY a compact insight (2-4 sentences), no preamble."""

# ----------------------------------------------------------------------
# Backward Evolution: Trajectory Analysis
# ----------------------------------------------------------------------
def build_submission_critique_prompt(
    task_description: str,
    subtrajectory_text: str,
    delta: float,
) -> str:
    """Step-level critique of one submission from its producing steps and
    its score delta."""
    return f"""You are auditing one submission of a recommendation task.
Produce a step-level critique: which actions contributed to the score change
and which introduced errors.

TASK:
{task_description}

EXECUTION SUB-TRAJECTORY (steps that produced this submission):
{subtrajectory_text[:8000]}

SCORE DELTA:
{delta:+.6f}

Output ONLY the critique as plain text (3-6 sentences), no preamble."""

# ----------------------------------------------------------------------
# Backward Evolution: Memory Evolution
# ----------------------------------------------------------------------
def build_exec_evaluation_prompt(
    exec_history_body: str,
    delta: float,
    critique: str,
) -> str:
    """Tool evaluation summary from its execution history, the score delta,
    and the step critique."""
    return f"""Evaluate one algorithmic tool using its execution history, the
score delta of the submission it contributed to, and the step critique.
Produce a compact evaluation summary that will guide future tool evolution
(retention, pruning, or targeted modification).

EXECUTION HISTORY:
{exec_history_body[:8000]}

SCORE DELTA:
{delta:+.6f}

STEP CRITIQUE:
{critique}

Output ONLY the evaluation summary (2-4 sentences), no preamble."""

def build_success_heuristic_prompt(evidence: str, delta: float) -> str:
    """Distill the strategy behind a score improvement into a heuristic body."""
    return f"""The score improved (delta = {delta:+.6f}). Distill the strategy
that caused the improvement into a transferable Experiential heuristic body.

EVIDENCE (step insight + experimental summary):
{evidence[:8000]}

Write the heuristic body: (1) the pattern that worked, (2) the data
conditions under which it applies, (3) the recommended action, (4) known
exceptions. Output ONLY the body, no preamble."""

def build_neutral_operation_prompt(evidence: str) -> str:
    """Record a neutral operation so future runs avoid redundant trials."""
    return f"""The score was unchanged (delta = 0). Distill what was tried and
why it produced no gain, so that future runs avoid redundant trials.

EVIDENCE (step insight + experimental summary):
{evidence[:8000]}

Write the heuristic body: what was tried, the likely reason for the neutral
outcome, and when this operation can be skipped. Output ONLY the body, no
preamble."""

def build_antipattern_prompt(evidence: str, delta: float) -> str:
    """Encode an anti-pattern and its fix after a score regression."""
    return f"""The score dropped (delta = {delta:+.6f}). Distill the
anti-pattern that caused the regression and the fix that should be applied
next time.

EVIDENCE (step insight + experimental summary):
{evidence[:8000]}

Write the heuristic body: (1) the anti-pattern, (2) the conditions under
which it occurs, (3) the corrective action. Output ONLY the body, no
preamble."""

def build_memory_merge_prompt(body_i: str, body_new: str) -> str:
    """Merge two overlapping Experiential bodies into one."""
    return f"""Two Experiential heuristic bodies describe overlapping insights.
Integrate them into a single unified body without losing information:
keep the conditions, recommended actions, and exceptions of both, and merge
redundant wording.

EXISTING BODY:
{body_i}

NEW BODY:
{body_new}

Output ONLY the unified body, no preamble."""

def build_title_summary_prompt(body_syn: str) -> str:
    """Generate a title and a summary from a consolidated heuristic body."""
    return f"""Given a consolidated Experiential heuristic body, generate:
1. Title: one-line identifier (max 15 words) used for similarity retrieval.
2. Summary: the transferable insight (2-3 sentences) used for secondary
   filtering.

BODY:
{body_syn}

Output ONLY:
TITLE: <title>
SUMMARY: <summary>"""
