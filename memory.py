"""
Hierarchical Contextual Memory (M)

Each item μ_i ∈ M is a 4-tuple:
  μ_i = <ttl_i, sum_i, bod_i, ℓ_i>

Storage format: each memory item is a separate file
  - Filename: {label}_{sanitized_title}.md
  - Content: summary + body in markdown format

Three semantic types:
  - Experiential (Exper.): Globally accumulated design heuristics
  - Experimental (Expt.): Per-task trial records
  - Execution (Exec.): Per-tool performance traces
"""

import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryLabel(str, Enum):
    EXPERIENTIAL = "Exper."
    EXPERIMENTAL = "Expt."
    EXECUTION = "Exec."


class MemoryItem:
    """A single memory entry: (title, summary, body, label)."""

    def __init__(
        self,
        title: str,
        summary: str,
        body: str,
        label: MemoryLabel,
        item_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ):
        self.title = title
        self.summary = summary
        self.body = body
        self.label = label
        self.item_id = item_id or f"mem_{int(time.time()*1000)}"
        self.created_at = created_at or time.time()

    def to_filename(self) -> str:
        """Generate filename from label and title."""
        # Sanitize title for filename
        safe_title = re.sub(r'[^\w\s-]', '', self.title)
        safe_title = re.sub(r'[\s]+', '_', safe_title)
        safe_title = safe_title  # Limit length
        label_prefix = self.label.value.replace('.', '')
        return f"{label_prefix}_{safe_title}.md"

    def to_markdown(self) -> str:
        """Convert to markdown document format."""
        return f"""# {self.title}

**Type**: {self.label.value}

## Summary

{self.summary}

## Body

{self.body}
"""

    @classmethod
    def from_markdown(cls, filepath: Path) -> "MemoryItem":
        """Load a memory item from a markdown file."""
        content = filepath.read_text(encoding="utf-8")

        # Parse title (first # heading)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filepath.stem

        # Parse type
        type_match = re.search(r'\*\*Type\*\*:\s*(.+)$', content, re.MULTILINE)
        label_str = type_match.group(1).strip() if type_match else "Exper."
        try:
            label = MemoryLabel(label_str)
        except ValueError:
            label = MemoryLabel.EXPERIENTIAL

        # Parse summary
        summary_match = re.search(r'## Summary\s*\n\n(.+?)(?=\n\n##|\Z)', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        # Parse body
        body_match = re.search(r'## Body\s*\n\n(.+?)$', content, re.DOTALL)
        body = body_match.group(1).strip() if body_match else ""

        return cls(
            title=title,
            summary=summary,
            body=body,
            label=label,
            item_id=filepath.stem,
            created_at=filepath.stat().st_mtime,
        )


class HierarchicalMemory:
    """
    Hierarchical Contextual Memory bank M.

    Paper Section 4.1.1:
    - Storage: title-summary-body (coarse-to-fine)
    - Each item stored as a separate markdown file
    - Filename: {label}_{title}.md
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.items: List[MemoryItem] = []
        self._load()

    def _load(self):
        """Load memory items from individual markdown files."""
        self.items = []
        for filepath in self.storage_dir.glob("*.md"):
            try:
                item = MemoryItem.from_markdown(filepath)
                self.items.append(item)
            except Exception:
                continue

    def _save_item(self, item: MemoryItem):
        """Save a single memory item as a markdown file."""
        filepath = self.storage_dir / item.to_filename()
        filepath.write_text(item.to_markdown(), encoding="utf-8")

    def add(self, item: MemoryItem):
        """Add a memory item and persist as a file."""
        self.items.append(item)
        self._save_item(item)

    def count(self, label: Optional[MemoryLabel] = None) -> int:
        """Count items, optionally filtered by label."""
        if label is None:
            return len(self.items)
        return sum(1 for item in self.items if item.label == label)

    def extract(
        self,
        query: str,
        recent_output: str = "",
        top_k: int = 5,
    ) -> List[MemoryItem]:
        """
        Extract relevant memory items (hierarchical retrieval).

        Paper: "the LLM first scans titles to triage C candidates,
        then reads their summaries to select the top-K items,
        and finally expands only those K bodies for reasoning."
        """
        if not self.items:
            return []

        # Simple keyword-based retrieval
        scored = []
        query_lower = query.lower()
        for item in self.items:
            score = 0
            # Title matching
            if any(word in item.title.lower() for word in query_lower.split()):
                score += 3
            # Summary matching
            if any(word in item.summary.lower() for word in query_lower.split()):
                score += 2
            # Body matching
            if any(word in item.body.lower() for word in query_lower.split()):
                score += 1
            # Recency bonus
            score += 0.001 * item.created_at
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]

    def get_titles_and_summaries(self) -> List[Dict[str, str]]:
        """Get all titles and summaries for display to the agent."""
        return [
            {"title": item.title, "summary": item.summary, "label": item.label.value}
            for item in self.items
        ]

    def distill_experiential(self, items: List[MemoryItem], llm_client=None) -> Optional[MemoryItem]:
        """
        Distill Experimental summaries into Experiential memory.

        Paper flow:
        1. Collect all Expt. summaries
        2. Combine summaries as body of new Exper.
        3. LLM generates summary + title from that body
        """
        if not items:
            return None

        # Step 1: Collect all Expt. summaries as the body
        combined_body = "\n\n".join(
            f"[{item.title}]\n{item.summary}"
            for item in items
        )

        # Step 2: Use LLM to generate title + summary from body
        if llm_client:
            try:
                prompt = f"""You are distilling experimental records into a transferable heuristic.

EXPERIMENTAL RECORDS (summaries):
{combined_body}

Generate a transferable Experiential heuristic:
1. Title: one-line identifier (max 15 words)
2. Summary: the transferable insight (2-3 sentences)

Output ONLY:
TITLE: <title>
SUMMARY: <summary>"""

                response = llm_client.chat.completions.create(
                    model="mimo-v2.5-pro",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                )
                content = response.choices[0].message.content

                # Parse title and summary
                title = "Distilled heuristic"
                summary = content
                if "TITLE:" in content:
                    parts = content.split("TITLE:")[1].split("SUMMARY:")
                    title = parts[0].strip()
                    if len(parts) > 1:
                        summary = parts[1].strip()

                return MemoryItem(
                    title=title,
                    summary=summary,
                    body=combined_body[:500000],
                    label=MemoryLabel.EXPERIENTIAL,
                )
            except Exception:
                pass

        # Fallback: use first item's title
        return MemoryItem(
            title=f"Distilled from {len(items)} experiments",
            summary=items[0].summary if items else "",
            body=combined_body[:500000],
            label=MemoryLabel.EXPERIENTIAL,
        )


def load_seed_hierarchical(memory: HierarchicalMemory, seed_path: str) -> int:
    """Load seed memory from a markdown file."""
    # This is a placeholder - seed memories should be written through evolution
    return 0
