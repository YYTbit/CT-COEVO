"""
prompts.py — Prompt templates for CT-COEVO.

Paper-aligned (Listings 1-11) + minimal meta tool guidance.
"""

from typing import Optional, List, Dict, Any


def build_context_descriptor_prompt(task_description: str) -> str:
    """Listing 1: Context Prefetching."""
    return f"""You are an expert recommendation system researcher. Your job is to
summarize a recommendation competition task description into a single
dense paragraph optimized for embedding-based retrieval.

Your summary MUST:
- Focus ONLY on essential technical details present in the description.
- Be dense, precise, and semantic-rich.
- Stay strictly below 250 tokens.
- Contain all relevant elements: task type (rating prediction, CTR,
ranking, sequential, etc.), input format, output format, evaluation
metric (and how it is computed), dataset structure and key fields,
submission format, and any important constraints or rules.

TASK DESCRIPTION:
{task_description}

STRICT FORMAT REQUIREMENTS:
- Output MUST be a single paragraph.
- No bullet points, no numbering, no markdown.
- No headings, no blank lines, no lists.
- No backticks or code blocks.
- No special characters other than standard punctuation.
- Do NOT introduce information not explicitly present in the description.
- Do NOT explain ML concepts or provide suggestions.

Now output ONLY the summarized single paragraph, with no explanations:"""


def build_kickoff_message() -> str:
    """First-step kickoff message with bold emphasis."""
    return (
        "**START NOW. Your FIRST action must be create_tool to train a model.**\n\n"
        "**Call create_tool with:**\n"
        "- name: 'deepfm_v1'\n"
        "- code: complete PyTorch GPU training code\n"
        "- description: 'DeepFM for recommendation'\n"
        "- review_time: 3600 (1 hour for training)\n\n"
        "**DO NOT waste steps on exploration. Train models immediately.**"
    )


def build_initial_solution_prompt(
    task_description: str,
    data_preview: str,
    data_knowledge: str,
    model_knowledge: str,
    installed_packages: str,
) -> str:
    """Listing 2: Initial Solution Generation."""
    return f"""You are a recommendation system expert attending a competition. You need
to design an excellent and creative solution and implement it in Python.

# Task description
{task_description}

# Instructions

## Response format
Your response should be a brief outline of your proposed solution in
natural language (3-5 sentences), followed by a single markdown code
block which implements this solution and prints the evaluation metric.
No additional headings or text.

## Solution sketch guideline
- The solution sketch should be 3-5 sentences.
- Propose a modeling approach appropriate for this recommendation task.
- Consider: negative sampling strategy, loss function alignment with
the ranking metric, temporal split if timestamps are available.
- Do NOT suggest exploratory data analysis (EDA).
- The data is already prepared and available in the './input' directory.

## Implementation guideline
- The code MUST print the evaluation metric computed on a hold-out
validation set. Without this metric, the solution cannot be evaluated.
- SAVE PREDICTIONS in a 'submission.csv' file in the './submission/'
directory as described in the task description.
- The code should be a single-file Python program, self-contained.
- No parts of the code should be skipped.
- All input data is stored in "./input" directory.
- Use "./working" directory for temporary files.

## Recommendation-Specific Guidelines
- For implicit feedback tasks, prefer pairwise losses (e.g., BPR) over
pointwise surrogates when the metric is ranking-oriented.
- Use temporal splits when timestamps are available; avoid random splits
that may introduce leakage.
- For ID-heavy sparse features, ensure proper embedding initialization.
- If using negative sampling, ensure the sampling strategy matches the
evaluation protocol.

## Installed Packages
Your solution can use: pandas, numpy, scipy, scikit-learn, torch,
lightgbm, xgboost, implicit, surprise, recbole. Feel free to use any
other installed packages.

## Tool Creation
**create_tool(name, code, description, review_time)**: Creates algorithm tool, auto-executes, returns logs after review_time or when done.
**edit_tool(tool_id, code, review_time)**: Updates existing tool, auto-executes, returns logs after review_time or when done.
review_time: seconds to wait before reviewing (e.g., 3600). If tool finishes early, logs return immediately. If tool runs longer, logs return at review_time.

**CRITICAL: Each step should be a SUBSTANTIAL training run (hours, not minutes). Train multiple models in parallel. Do NOT waste steps on exploration.**

# Data preview
{data_preview}

# External knowledge
Here is information about data loading, preprocessing, model selection,
and training from recommendation experts.

## Data loading and preprocessing
{data_knowledge}

## Model selection and model training
{model_knowledge}"""


def build_debugging_prompt(
    task_description: str,
    buggy_code: str,
    terminal_output: str,
) -> str:
    """Listing 3: Debugging Failed Pipeline."""
    return f"""You are a recommendation system expert. Your previous solution had a bug
and/or did not produce a valid submission.csv, or the generated
submission.csv was in an incorrect format.
Based on the information below, revise it to fix the issue.

# Task description
{task_description}

# Instructions

## Response format
Your response should be a brief outline of the fix (3-5 sentences),
followed by a single markdown code block implementing the fix.

## Bugfix guideline
- Describe in natural language (3-5 sentences) how to fix the issue.
- Keep the core recommendation method the same; do not change the
modeling approach, only fix the code.
- If the code failed due to missing library, avoid using that library.
Do NOT attempt to install packages with pip or conda.
- All packages are already installed.

## Implementation guideline
- The code MUST print the evaluation metric on a hold-out validation set.
- SAVE PREDICTIONS in './submission/submission.csv'.
- Single-file, self-contained Python program.
- No skipped parts, no early termination.

# Data preview
{task_description}

# Previous (buggy) implementation
{buggy_code}

# Execution output
{terminal_output}"""


def build_memory_guided_improvement_prompt(
    task_description: str,
    data_preview: str,
    previous_memory_solution: str,
    improve_idea: str,
) -> str:
    """Listing 4: Memory-Guided Improvement."""
    return f"""You are a recommendation system expert. You are provided with previous
memory including previously developed solutions and a creative idea
derived from your accumulated experience. Implement this idea on top of
the previously developed solution.

# Task description
{task_description}

# Instructions

## Response format
Your response should be a brief outline (3-5 sentences), followed by a
single markdown code block implementing the improvement.

## Solution improvement guideline
- Describe how you improved the previous solution based on the creative
idea (3-5 sentences).
- Do NOT perform EDA.
- All packages are installed. Do NOT install anything with pip or conda.

## Implementation guideline
- The code MUST print the evaluation metric on a hold-out validation set.
- SAVE PREDICTIONS in './submission/submission.csv'.
- Single-file, self-contained Python program.
- No skipped parts, no early termination.

# Data preview
{data_preview}

# Previous memory and solution
{previous_memory_solution}

# Creative idea
This idea is derived from accumulated Experiential memory and may
improve performance.

Creative idea:
{improve_idea}"""


def build_experimental_memory_synthesis_prompt(
    task_description: str,
    tool_name: str,
    tool_config: str,
    execution_output: str,
    metric_name: str,
    metric_value: str,
    submission_valid: str,
) -> str:
    """Listing 5: Experimental Memory Synthesis."""
    return f"""You are an expert recommendation system researcher performing post-hoc
analysis of an experiment. Synthesize the following execution output
into a structured memory entry.

# Task description
{task_description}

# Tool executed
Tool name: {tool_name}
Tool configuration: {tool_config}

# Execution output
{execution_output}

# Evaluation result
Metric: {metric_name}
Score: {metric_value}
Submission valid: {submission_valid}

# Your Task
Generate a structured memory entry in the following JSON format:
{{
  "title": "One-line identifier (max 15 words)",
  "summary": "Compact description of what happened (2-3 sentences)",
  "body": "Full detail including: (1) what was tried, (2) what happened, (3) root cause if failure, (4) suggested next step",
  "type": "Expt.",
  "data_symptoms": "Key characteristics of the dataset observed",
  "decision": "The key design choice made",
  "failure_mode": "Description of failure if applicable, else null",
  "fix": "Repair strategy if applicable, else null"
}}

Requirements:
- Be specific about WHAT was tried and WHY it succeeded or failed.
- Identify the causal chain: data conditions -> design choice -> outcome.
- If the execution failed, classify the failure type (environment error,
syntax error, algorithm error, etc.).
- Keep the body under 500 tokens.

Output ONLY the JSON, no additional text:"""


def build_experiential_memory_distillation_prompt(
    experimental_entries: str,
    task_contexts: str,
) -> str:
    """Listing 6: Experiential Memory Distillation."""
    return f"""You are an expert recommendation system researcher. Multiple experiments
across different tasks have revealed recurring patterns. Your job is to
distill these into a transferable heuristic.

# Experimental Entries
{experimental_entries}

# Task Contexts
{task_contexts}

# Your Task
Analyze the experimental entries and identify recurring patterns that
generalize across tasks. Generate an Experiential memory entry:

{{
  "title": "One-line heuristic identifier (max 15 words)",
  "summary": "The transferable insight (2-3 sentences)",
  "body": "Full detail including: (1) the pattern observed, (2) under what data conditions it holds, (3) the recommended action, (4) known exceptions or limitations",
  "type": "Exper.",
  "applicability_conditions": "When does this heuristic apply?",
  "recommended_action": "What should the agent do?",
  "confidence": "high/medium/low based on evidence strength"
}}

Requirements:
- The heuristic MUST be transferable across different datasets.
- Be specific about the CONDITIONS under which the heuristic applies.
- Distinguish correlation from causation.
- If evidence is weak, state confidence as "low".

Output ONLY the JSON, no additional text:"""


def build_memory_extraction_prompt(
    task_description: str,
    current_step: int,
    max_steps: int,
    best_score: float,
    time_elapsed: str,
    recent_execution_output: str,
    memory_previews: str,
    K: int = 5,
) -> str:
    """Listing 7: Memory Extraction for Planning."""
    return f"""You are an expert recommendation system agent planning your next step.

# Current Task
{task_description}

# Current Progress
Step: {current_step}/{max_steps}
Best score so far: {best_score}
Time elapsed: {time_elapsed}

# Recent Execution
{recent_execution_output}

# Memory Previews
Below are previews of available memory items. Each preview contains
the title, summary, and type (Exper. = Experiential, Expt. = Experimental,
Exec. = Execution).

{memory_previews}

# Your Task
Select the TOP-{K} most relevant memory items for planning your next
step. Consider:
1. Experiential memories for transferable heuristics
2. Experimental memories to avoid repeating failed approaches
3. Execution memories for tool-specific performance insights

Return your selection as a JSON array of indices:
{{
  "selected_indices": [1, 3, 5, ...],
  "reasoning": "Brief explanation of why these memories are relevant"
}}

Output ONLY the JSON:"""


def build_research_plan_prompt(
    task_description: str,
    best_code: str,
    memory: str,
) -> str:
    """Listing 8: Research Plan Generation."""
    return f"""You are a recommendation system Grandmaster. Based on the task
description, current best code, and accumulated memory, identify
improvement directions.

# Task description
{task_description}

# Current Best Code
{best_code}

# Memory (accumulated experience and past attempts)
{memory}

# Your Task
Identify at least 3 major directions where the solution can be improved.
For each direction, propose practical and specific suggestions.

REQUIREMENTS:
- Do NOT suggest ensembling methods.
- Do NOT suggest cross-validation with k>5.
- Suggestions must be specific and unambiguous.
- Avoid using "e.g." and "or" -- commit to specific choices.
- Consider recommendation-specific aspects: negative sampling, loss
function alignment with ranking metrics, temporal splits, embedding
strategies for sparse IDs.

Your response MUST follow this JSON format:
{{
  "major_direction_1": {{
    "1": "Detailed specific suggestion 1",
    "2": "Detailed specific suggestion 2"
  }},
  "major_direction_2": {{
    "1": "Detailed specific suggestion 1",
    "2": "Detailed specific suggestion 2"
  }},
  "major_direction_3": {{
    "1": "Detailed specific suggestion 1",
    "2": "Detailed specific suggestion 2"
  }}
}}

Output ONLY the JSON:"""


def build_tool_review_prompt(
    tool_name: str,
    source_tool: str,
    modification_description: str,
    task_list: str,
    score_list: str,
    baseline_scores: str,
    improvement_summary: str,
    execution_logs: str,
) -> str:
    """Listing 9: Tool Review and Promotion."""
    return f"""You are a senior recommendation system engineer reviewing a code change.
Decide whether to promote this tool variant to the global toolkit.

# Tool Under Review
Tool name: {tool_name}
Source tool: {source_tool}
Modification applied: {modification_description}

# Execution Evidence
Tasks evaluated: {task_list}
Scores achieved: {score_list}
Baseline scores (source tool): {baseline_scores}
Score improvement: {improvement_summary}

# Execution Logs
{execution_logs}

# Your Decision
Evaluate whether this tool variant should be promoted to the global
toolkit. Consider:
1. Did it improve scores across multiple tasks?
2. Is the improvement consistent or due to variance?
3. Are there any failure modes or edge cases?
4. Is the code quality acceptable?

Return your decision as JSON:
{{
  "decision": "promote" or "reject",
  "confidence": "high/medium/low",
  "reasoning": "Detailed explanation of your decision",
  "suggested_modifications": "Any changes needed before promotion, or null if promoting as-is"
}}

Output ONLY the JSON:"""


def build_meta_tool_revision_prompt(
    tool_name: str,
    tool_code: str,
    experiential_evidence: str,
    execution_evidence: str,
    failure_summary: str,
) -> str:
    """Listing 10: Meta-Tool Revision."""
    return f"""You are a recommendation system engineering lead diagnosing a systematic
tool deficiency. Propose a targeted modification to fix the issue.

# Tool Under Revision
Tool name: {tool_name}
Current implementation: {tool_code}

# Diagnostic Evidence
Experiential memory entries indicating deficiency:
{experiential_evidence}

Execution memory showing failure patterns:
{execution_evidence}

# Failure Analysis
Common failure modes observed:
{failure_summary}

# Your Task
Propose a SPECIFIC, TARGETED modification to fix the identified
deficiency. The modification should:
1. Address the root cause, not symptoms
2. Preserve validated components that are working
3. Be minimal -- change only what is necessary
4. Be compatible with the existing tool interface

Return the modification as JSON:
{{
  "modification_type": "replace"/"insert"/"delete",
  "target_component": "Which part of the code to modify",
  "original_code": "The code being replaced (if applicable)",
  "new_code": "The replacement code",
  "reasoning": "Why this modification addresses the deficiency",
  "risk_assessment": "Potential risks of this change"
}}

Output ONLY the JSON:"""


def build_post_task_audit_prompt(
    task_description: str,
    metric_name: str,
    final_score: float,
    best_score: float,
    tool_execution_history: str,
    experimental_entries: str,
    temp_tools_list: str,
) -> str:
    """Listing 11: Post-Task Audit."""
    return f"""You are a recommendation system engineering lead performing a post-task
audit. Review all tool executions and consolidate learnings.

# Completed Task
{task_description}

# Final Result
Metric: {metric_name}
Final score: {final_score}
Best score achieved: {best_score}

# Tool Execution History
{tool_execution_history}

# Experimental Memory Entries
{experimental_entries}

# Temporary Tools Created
{temp_tools_list}

# Your Task
Perform a comprehensive audit:

1. TOOL AUDIT: For each temporary tool, decide whether to:
   - PROMOTE to global toolkit (if consistently effective)
   - PRUNE (if ineffective or superseded)
   - KEEP as-is (if promising but needs more evidence)

2. MEMORY CONSOLIDATION: Identify recurring Experimental patterns that
   should be distilled into Experiential heuristics.

3. LESSONS LEARNED: Extract key insights for future tasks.

Return your audit as JSON:
{{
  "tool_decisions": {{
    "tool_name_1": {{
      "action": "promote"/"prune"/"keep",
      "reasoning": "..."
    }},
    ...
  }},
  "new_experiential_entries": [
    {{
      "title": "...",
      "summary": "...",
      "body": "..."
    }},
    ...
  ],
  "lessons_learned": [
    "Lesson 1",
    "Lesson 2",
    ...
  ]
}}

Output ONLY the JSON:"""


