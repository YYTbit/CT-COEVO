"""
memory.py - Hierarchical Contextual Memory.

The memory bank is organized along two dimensions:
storage depth (title-summary-body, for coarse-to-fine retrieval) and semantic
type. Each item is stored as a (title, summary, body, label) tuple, and the
three semantic types play distinct roles:

    - Experiential: globally accumulated, transferable heuristics
    - Experimental: per-task trial records (executed actions and
      outcomes, raw material for generalization)
    - Execution:    per-tool performance traces and failure logs
      that feed the tool-evolution process

Storage format: one markdown file per item ({label}_{title}.md) with
title / summary / body sections.
"""

import re
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class MemoryLabel(str, Enum):
    EXPERIENTIAL = "Exper."
    EXPERIMENTAL = "Expt."
    EXECUTION = "Exec."


class MemoryItem:
    """A single memory entry storing (title, summary, body, label)."""

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
        """File name derived from the label and the sanitized title."""
        safe_title = re.sub(r'[^\w\s-]', '', self.title)
        safe_title = re.sub(r'[\s]+', '_', safe_title)
        label_prefix = self.label.value.replace('.', '')
        return f"{label_prefix}_{safe_title}.md"

    def to_markdown(self) -> str:
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

        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filepath.stem

        type_match = re.search(r'\*\*Type\*\*:\s*(.+)$', content, re.MULTILINE)
        label_str = type_match.group(1).strip() if type_match else "Exper."
        try:
            label = MemoryLabel(label_str)
        except ValueError:
            label = MemoryLabel.EXPERIENTIAL

        summary_match = re.search(r'## Summary\s*\n\n(.+?)(?=\n\n##|\Z)', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

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

    Items are stored one per markdown file and are organized along the
    title -> summary -> body depth axis and the semantic-type axis
    (Exper. / Expt. / Exec.).
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.items: List[MemoryItem] = []
        self._load()

    def _load(self):
        self.items = []
        for filepath in self.storage_dir.glob("*.md"):
            try:
                item = MemoryItem.from_markdown(filepath)
                self.items.append(item)
            except Exception:
                continue

    def _save_item(self, item: MemoryItem):
        filepath = self.storage_dir / item.to_filename()
        filepath.write_text(item.to_markdown(), encoding="utf-8")

    def add(self, item: MemoryItem):
        """Add an item to M and persist it."""
        self.items.append(item)
        self._save_item(item)

    def update(self, item: MemoryItem):
        """Persist an updated item in place (title/summary/body refresh)."""
        self._save_item(item)

    def remove_item(self, item: MemoryItem):
        """Remove an item from M and delete its file.

        Used when an assimilated entry is merged into a consolidated
        Experiential body.
        """
        self.items = [i for i in self.items if i is not item and i.item_id != item.item_id]
        try:
            (self.storage_dir / item.to_filename()).unlink(missing_ok=True)
        except Exception:
            pass

    def count(self, label: Optional[MemoryLabel] = None) -> int:
        if label is None:
            return len(self.items)
        return sum(1 for item in self.items if item.label == label)

    def items_of(self, label: Optional[MemoryLabel] = None) -> List[MemoryItem]:
        if label is None:
            return list(self.items)
        return [item for item in self.items if item.label == label]

    # ------------------------------------------------------------------
    # Memory Retrieval
    # ------------------------------------------------------------------
    def retrieve_experiential(
        self,
        query: str,
        embedder,
        tau1: float = 0.8,
    ) -> List[MemoryItem]:
        """Retrieve Experiential entries whose title matches the query.

        The joint query (task description + previous-step insight) is compared
        against the titles of Experiential entries with cosine similarity
        computed by the Harrier-OSS (0.6B) model; entries above the threshold
        tau1 = 0.8 are returned. The caller subsequently lets the LLM select a
        relevant subset from the returned summaries and expand the
        corresponding full bodies.
        """
        items = self.items_of(MemoryLabel.EXPERIENTIAL)
        if not items:
            return []
        sims = embedder.similarity_matrix([query], [item.title for item in items])[0]
        return [item for item, s in zip(items, sims) if float(s) > tau1]

    # ------------------------------------------------------------------
    # Memory Evolution
    # ------------------------------------------------------------------
    def merge_experiential(
        self,
        new_body: str,
        embedder,
        tau3: float = 0.8,
        llm_client=None,
        model: str = "deepseek-ai/DeepSeek-V3.2",
    ) -> MemoryItem:
        """Consolidate a distilled heuristic body into the Experiential bank.

        If an existing Experiential body is highly similar (cosine similarity
        above tau3 = 0.8) to the newly distilled heuristic body, the LLM merges
        the two into one and the assimilated entry is removed; otherwise the
        new body is kept as is. The LLM then produces the title and summary,
        and the fully formed Experiential entry is appended.
        """
        existing = self.items_of(MemoryLabel.EXPERIENTIAL)

        # --- similarity check against existing Experiential bodies ---
        matched: List[MemoryItem] = []
        if existing and new_body.strip():
            sims = embedder.similarity_matrix([new_body], [item.body for item in existing])[0]
            order = sorted(range(len(existing)), key=lambda j: float(sims[j]), reverse=True)
            for j in order:
                if float(sims[j]) > tau3:
                    matched.append(existing[j])
        if len(matched) > 1:
            # keep merging only the most similar entry; others are left intact
            matched = matched[:1]

        merged_body = new_body
        if matched:
            merged_body = self._llm_merge(matched[0].body, new_body, llm_client, model)

        title, summary = self._llm_title_summary(merged_body, llm_client, model)

        # --- remove the assimilated entry, append the new one ---
        for item in matched:
            self.remove_item(item)

        item = MemoryItem(
            title=title,
            summary=summary,
            body=merged_body,
            label=MemoryLabel.EXPERIENTIAL,
        )
        self.add(item)
        return item

    def _llm_merge(self, body_i: str, body_new: str, llm_client, model: str) -> str:
        """LLM integration of overlapping insights into a unified body."""
        if llm_client is None:
            return body_new if body_new.strip() else body_i
        from .prompts import build_memory_merge_prompt

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": build_memory_merge_prompt(body_i, body_new)}
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            content = (response.choices[0].message.content or "").strip()
            return content if content else (body_new if body_new.strip() else body_i)
        except Exception:
            return body_new if body_new.strip() else body_i

    def _llm_title_summary(self, body_syn: str, llm_client, model: str) -> Tuple[str, str]:
        """Generate (title, summary) from a consolidated body."""
        if llm_client is None:
            lines = [ln.strip() for ln in body_syn.splitlines() if ln.strip()]
            title = lines[0][:120] if lines else "Distilled heuristic"
            summary = body_syn.strip().split("\n")[0][:500] if body_syn.strip() else ""
            return title, summary
        from .prompts import build_title_summary_prompt

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": build_title_summary_prompt(body_syn)}
                ],
                temperature=0.2,
                max_tokens=300,
            )
            content = (response.choices[0].message.content or "").strip()
            title, summary = "Distilled heuristic", content
            if "TITLE:" in content:
                parts = content.split("TITLE:")[1].split("SUMMARY:")
                title = parts[0].strip()[:200] or title
                if len(parts) > 1:
                    summary = parts[1].strip()
            return title, summary
        except Exception:
            return "Distilled heuristic", body_syn.strip()[:500]
