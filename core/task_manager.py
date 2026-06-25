#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FORGE  — FILE: core/task_manager.py                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Forge (formerly Brain Loader v1)
# REPO:       https://github.com/Ehsas317/forge
# WHAT:       It hammers out a project sequentially, task by task, with
#             iterative review. A forge is slow, hot, and precise.
#
# THIS FILE:
#   Task Manager — decomposes project plans into actionable tasks, tracks
#   dependencies, and manages task state throughout the build process.
#
# KEY COMPONENTS:
#   - TaskManager: Main class for task decomposition and tracking
#   - Task: Data class representing a single task
#   - decompose_plan(): Breaks Brain plan into executable tasks
#   - parse_plan(): Restores tasks from saved plan format
#
# HOW TO USE FORGE:
#   1. Install:    pip install -r requirements.txt
#   2. Configure:  Edit config.yaml with your API tokens
#   3. Run:        python main.py "Your project description"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Forge — Task Manager

Handles task decomposition, dependency tracking, and execution state.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("forge.task_manager")


@dataclass
class Task:
    """Represents a single development task."""
    id: str
    type: str  # frontend/backend/devops/security/docs/testing
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending/running/done/failed
    priority: int = 1  # 1=high, 2=medium, 3=low
    estimated_tokens: int = 4096
    output_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Task":
        return cls(**data)


class TaskManager:
    """
    Forge Task Manager

    Decomposes project plans into actionable tasks, tracks dependencies,
    and manages task state throughout the build process.

    Usage:
        manager = TaskManager()
        tasks = manager.decompose_plan(brain_plan)
        for task in tasks:
            if manager.can_execute(task):
                execute(task)
    """

    # Task type patterns for auto-detection
    TYPE_PATTERNS = {
        "frontend": [
            r"\b(react|vue|angular|frontend|ui|component|css|html|jsx|tsx)\b",
            r"\b(screen|page|view|layout|modal|form|button)\b",
        ],
        "backend": [
            r"\b(api|endpoint|server|database|model|schema|migration)\b",
            r"\b(auth|login|register|middleware|controller|service)\b",
        ],
        "devops": [
            r"\b(docker|dockerfile|ci/cd|pipeline|nginx|deploy)\b",
            r"\b(kubernetes|k8s|terraform|ansible|github.action)\b",
        ],
        "security": [
            r"\b(security|auth|oauth|jwt|encrypt|hash|sanitize)\b",
            r"\b(vulnerability|xss|sql.injection|csp|cors)\b",
        ],
        "docs": [
            r"\b(documentation|readme|api.doc|swagger|user.guide)\b",
            r"\b(changelog|contributing|license|wiki)\b",
        ],
        "testing": [
            r"\b(test|spec|jest|pytest|cypress|e2e|unit.test)\b",
            r"\b(integration|coverage|mock|fixture|snapshot)\b",
        ],
    }

    def __init__(self):
        self.tasks: List[Task] = []
        self.completed_ids: set = set()
        self.failed_ids: set = set()

    def decompose_plan(self, plan_text: str) -> List[Dict]:
        """
        Decompose a Brain plan into structured tasks.

        Parses the plan text and extracts task definitions with IDs,
        types, descriptions, and dependencies.
        """
        logger.info("[TaskManager] Decomposing plan...")

        tasks = []
        task_pattern = r'(?:^|\n)\s*(?:Task|Step)\s*[#]?\s*(\w+)[\s:-]+(.+?)(?=\n(?:Task|Step)\s*[#]?\s*\w|\Z)'

        matches = re.findall(task_pattern, plan_text, re.IGNORECASE | re.DOTALL)

        if not matches:
            # Fallback: try simpler pattern
            matches = self._simple_parse(plan_text)

        for idx, (task_id, task_content) in enumerate(matches, 1):
            task_id = task_id.strip() or f"T{idx:03d}"
            task_type = self._detect_type(task_content)
            description = task_content.strip()

            # Extract dependencies
            deps = self._extract_dependencies(description, tasks)

            task = Task(
                id=task_id,
                type=task_type,
                description=description,
                dependencies=deps,
                priority=1 if idx <= 3 else 2,
            )

            tasks.append(task.to_dict())
            logger.debug("[TaskManager] Created task %s (%s)", task_id, task_type)

        logger.info("[TaskManager] Decomposed into %d tasks", len(tasks))
        return tasks

    def _simple_parse(self, plan_text: str) -> List[tuple]:
        """Simple fallback parser for non-structured plans."""
        lines = plan_text.split('\n')
        tasks = []
        current_id = None
        current_content = []

        for line in lines:
            # Look for numbered items or bullet points
            match = re.match(r'^(?:\d+\.\s*|[-*]\s+)(.+)', line)
            if match:
                if current_id:
                    tasks.append((current_id, '\n'.join(current_content)))
                current_id = f"T{len(tasks) + 1:03d}"
                current_content = [match.group(1)]
            elif current_id and line.strip():
                current_content.append(line)

        if current_id:
            tasks.append((current_id, '\n'.join(current_content)))

        return tasks

    def _detect_type(self, content: str) -> str:
        """Auto-detect task type from content."""
        content_lower = content.lower()

        for task_type, patterns in self.TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return task_type

        return "general"

    def _extract_dependencies(self, content: str, existing_tasks: List) -> List[str]:
        """Extract task dependencies from content."""
        deps = []
        # Look for references to other task IDs
        for task in existing_tasks:
            task_id = task.id if hasattr(task, 'id') else task.get('id', '')
            if task_id and task_id in content:
                deps.append(task_id)
        return deps

    def parse_plan(self, plan_data: str) -> List[Dict]:
        """Parse a saved plan file back into tasks."""
        # Try JSON first
        try:
            data = json.loads(plan_data)
            if isinstance(data, list):
                return data
            return data.get("tasks", [])
        except json.JSONDecodeError:
            pass

        # Try markdown format
        return self.decompose_plan(plan_data)

    def can_execute(self, task: Dict) -> bool:
        """Check if a task's dependencies are satisfied."""
        deps = task.get("dependencies", [])
        return all(dep in self.completed_ids for dep in deps)

    def mark_complete(self, task_id: str):
        """Mark a task as completed."""
        self.completed_ids.add(task_id)
        logger.info("[TaskManager] Task %s marked complete", task_id)

    def mark_failed(self, task_id: str):
        """Mark a task as failed."""
        self.failed_ids.add(task_id)
        logger.warning("[TaskManager] Task %s failed", task_id)

    def get_next_task(self) -> Optional[Dict]:
        """Get the next executable task by priority."""
        pending = [
            t for t in self.tasks
            if t.get("status") == "pending" and t.get("id") not in self.failed_ids
        ]

        # Sort by priority
        pending.sort(key=lambda t: t.get("priority", 99))

        for task in pending:
            if self.can_execute(task):
                return task

        return None

    def get_progress(self) -> Dict[str, int]:
        """Get current task progress statistics."""
        total = len(self.tasks)
        completed = len(self.completed_ids)
        failed = len(self.failed_ids)
        pending = total - completed - failed

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "percent": (completed / total * 100) if total > 0 else 0,
        }

    def save_tasks(self, filepath: str = "memory/tasks.json"):
        """Save current tasks to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.tasks, f, indent=2)

    def load_tasks(self, filepath: str = "memory/tasks.json"):
        """Load tasks from disk."""
        with open(filepath, 'r') as f:
            self.tasks = json.load(f)
