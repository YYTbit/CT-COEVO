"""
Scalable Algorithmic Toolkit.

Four tool roles partition the toolkit:
  - Base tools: immutable execution primitives
  - Meta tools: operations that modify the toolkit itself
  - Global tools: evolved, audited pipelines that persist across tasks
  - Temporary tools: task-scoped experimental variants

Persistence format: one JSON file per tool under the storage directory.
"""

import itertools
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


_tool_id_seq = itertools.count(1)


def _next_tool_id() -> str:
    """Process-unique tool id (timestamp + monotonic counter)."""
    return f"tool_{int(time.time() * 1000)}_{next(_tool_id_seq)}"


class ToolType(str, Enum):
    BASE = "base"
    META = "meta"
    GLOBAL = "global"
    TEMPORARY = "temp"


class ToolItem:
    """A single tool entry in the toolkit."""

    def __init__(
        self,
        name: str,
        tool_type: ToolType,
        description: str,
        source_code: str,
        tool_id: Optional[str] = None,
        created_at: Optional[float] = None,
        success_count: int = 0,
        failure_count: int = 0,
    ):
        self.name = name
        self.tool_type = tool_type
        self.description = description
        self.source_code = source_code
        self.tool_id = tool_id or _next_tool_id()
        self.created_at = created_at or time.time()
        self.success_count = success_count
        self.failure_count = failure_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "tool_type": self.tool_type.value,
            "description": self.description,
            "source_code": self.source_code,
            "created_at": self.created_at,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolItem":
        return cls(
            name=d["name"],
            tool_type=ToolType(d["tool_type"]),
            description=d["description"],
            source_code=d["source_code"],
            tool_id=d.get("tool_id"),
            created_at=d.get("created_at"),
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
        )


class ScalableToolkit:
    """
    Scalable Algorithmic Toolkit.

    The toolkit is partitioned into four tool roles: base, meta, global, and
    temporary. Global tools provide stable reuse, temporary tools enable
    experimentation, and meta tools modify the toolkit itself.
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.tools: List[ToolItem] = []
        self._load()

    def _load(self):
        """Load tools from disk."""
        tools_file = self.storage_dir / "toolkit_items.json"
        if tools_file.exists():
            try:
                with open(tools_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tools = [ToolItem.from_dict(d) for d in data]
            except Exception:
                self.tools = []

    def _save(self):
        """Save tools to disk."""
        tools_file = self.storage_dir / "toolkit_items.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(tools_file, "w", encoding="utf-8") as f:
            json.dump([tool.to_dict() for tool in self.tools], f, indent=2, ensure_ascii=False)

    def add(self, tool: ToolItem):
        """Add a tool and persist."""
        self.tools.append(tool)
        self._save()

    def get(self, tool_id: str) -> Optional[ToolItem]:
        """Get a tool by ID."""
        for tool in self.tools:
            if tool.tool_id == tool_id:
                return tool
        return None

    def remove(self, tool_id: str):
        """Remove a tool by ID."""
        self.tools = [t for t in self.tools if t.tool_id != tool_id]
        self._save()

    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[ToolItem]:
        """List tools, optionally filtered by type."""
        if tool_type is None:
            return self.tools
        return [t for t in self.tools if t.tool_type == tool_type]

    def count(self, tool_type: Optional[ToolType] = None) -> int:
        """Count tools, optionally filtered by type."""
        if tool_type is None:
            return len(self.tools)
        return sum(1 for t in self.tools if t.tool_type == tool_type)

    def promote_to_global(self, tool_id: str):
        """Promote a temporary tool to global after it passes the audit."""
        tool = self.get(tool_id)
        if tool and tool.tool_type == ToolType.TEMPORARY:
            tool.tool_type = ToolType.GLOBAL
            self._save()

    def prune(self, tool_id: str):
        """Remove a tool that failed the retention audit."""
        self.remove(tool_id)

    def clear_temporary(self):
        """Remove all temporary tools."""
        for tool in [t for t in self.tools if t.tool_type == ToolType.TEMPORARY]:
            self.remove(tool.tool_id)
