"""
Task Manager
Handles markdown-based task/result files and project state.
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a single task in the project."""
    number: int
    worker_model: str
    title: str
    subtasks: List[str]
    pre_instructions: str
    notes: str
    max_tokens: int = 8192
    status: str = "pending"  # pending, in_progress, completed, failed, remediation
    rating: Optional[int] = None
    review_notes: str = ""
    result_file: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ProjectState:
    """Overall project state."""
    project_name: str
    app_idea: str
    total_tasks: int = 0
    completed_tasks: int = 0
    current_task: int = 0
    iteration: int = 0  # Review iteration counter
    status: str = "initialized"  # initialized, planning, executing, reviewing, complete
    brain_model: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class TaskManager:
    """
    Manages all task files, results, and project state.
    Uses markdown as the communication protocol between Brain and Workers.
    """

    def __init__(self, project_name: str, tasks_dir: str = "./tasks", 
                 logs_dir: str = "./logs"):
        self.project_name = project_name
        self.tasks_dir = Path(tasks_dir)
        self.logs_dir = Path(logs_dir)

        # Create directories
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.tasks_dir / "project_state.json"
        self.master_log = self.tasks_dir / "master_log.md"
        self.task_list_file = self.tasks_dir / "task_list.json"

        self.state: Optional[ProjectState] = None
        self.tasks: Dict[int, Task] = {}

        logger.info("[TaskManager] Initialized for project: %s", project_name)

    def initialize_project(self, app_idea: str, brain_model: str) -> ProjectState:
        """Create new project state."""
        self.state = ProjectState(
            project_name=self.project_name,
            app_idea=app_idea,
            brain_model=brain_model,
            status="initialized"
        )
        self._save_state()

        # Initialize master log
        self._write_master_log(f"""# Brain Loader Project Log

**Project:** {self.project_name}
**App Idea:** {app_idea}
**Brain Model:** {brain_model}
**Started:** {self.state.created_at}

---
""")

        logger.info("[TaskManager] Project initialized: %s", app_idea)
        return self.state

    def save_master_plan(self, plan_text: str) -> None:
        """Save the Brain's master plan to a file."""
        plan_file = self.tasks_dir / "master_plan.md"
        with open(plan_file, "w") as f:
            f.write(f"# Master Plan\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(plan_text)

        self._write_master_log(f"Master plan created and saved to {plan_file}")
        self.state.status = "planning"
        self._save_state()
        logger.info("[TaskManager] Master plan saved.")

    def add_task(self, task: Task) -> None:
        """Add a task to the project."""
        self.tasks[task.number] = task
        self._save_task_list()

        # Write task file for worker
        self._write_task_file(task)

        self.state.total_tasks = len(self.tasks)
        self._save_state()

        logger.info("[TaskManager] Added task %d: %s", task.number, task.title)

    def write_task_result(self, task_number: int, result_text: str,
                          rating: Optional[int] = None,
                          review_notes: str = "") -> str:
        """
        Write worker result to markdown file.
        Returns path to result file.
        """
        task = self.tasks.get(task_number)
        if not task:
            raise ValueError(f"Task {task_number} not found")

        result_file = self.tasks_dir / f"result_{task.number:03d}_{task.worker_model}.md"

        content = f"""# Result: Task {task.number} — {task.title}

**Worker Model:** {task.worker_model}
**Status:** {"COMPLETED" if rating and rating >= 6 else "NEEDS_REVIEW"}
**Completed At:** {datetime.now().isoformat()}
**Rating:** {rating}/10 if rating else "Pending"

## Original Subtasks
"""
        for i, st in enumerate(task.subtasks, 1):
            content += f"{i}. {st}\n"

        content += f"""
## Review Notes
{review_notes if review_notes else "No review notes yet."}

---

## Worker Output

{result_text}
"""

        with open(result_file, "w") as f:
            f.write(content)

        # Update task
        task.result_file = str(result_file)
        task.status = "completed" if (rating and rating >= 6) else "remediation"
        task.rating = rating
        task.review_notes = review_notes
        task.completed_at = datetime.now().isoformat()

        self._save_task_list()

        # Log
        self._write_master_log(
            f"Task {task.number} completed. Rating: {rating}/10. "
            f"Result: {result_file}"
        )

        logger.info("[TaskManager] Result written for task %d", task_number)
        return str(result_file)

    def get_task_file_content(self, task_number: int) -> str:
        """Get the content of a task file for worker consumption."""
        task = self.tasks.get(task_number)
        if not task:
            raise ValueError(f"Task {task_number} not found")

        task_file = self.tasks_dir / f"task_{task.number:03d}_{task.worker_model}.md"
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        with open(task_file, "r") as f:
            return f.read()

    def get_result_content(self, task_number: int) -> Optional[str]:
        """Get result content for Brain review."""
        task = self.tasks.get(task_number)
        if not task or not task.result_file:
            return None

        result_path = Path(task.result_file)
        if not result_path.exists():
            return None

        with open(result_path, "r") as f:
            return f.read()

    def get_context_for_task(self, task_number: int, max_chars: int = 4000) -> str:
        """
        Aggregate relevant previous results as context.
        Truncates to fit within context window budget.
        """
        context_parts = []

        for num in range(1, task_number):
            prev_task = self.tasks.get(num)
            if not prev_task or not prev_task.result_file:
                continue

            result_content = self.get_result_content(num)
            if not result_content:
                continue

            # Extract just the worker output section
            output_match = re.search(
                r"## Worker Output\n\n(.+)", 
                result_content, 
                re.DOTALL
            )

            if output_match:
                output = output_match.group(1)[:1500]  # Truncate each
            else:
                output = result_content[:1500]

            context_parts.append(
                f"### Context from Task {num} ({prev_task.worker_model})\n{output}\n"
            )

        full_context = "\n".join(context_parts)

        # Truncate if too long
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "\n\n[Context truncated...]"

        return full_context if full_context else "No previous context available."

    def get_all_results_summary(self) -> str:
        """Get summary of all results for final review."""
        summary_parts = []

        for num in sorted(self.tasks.keys()):
            task = self.tasks[num]
            summary_parts.append(
                f"Task {num}: {task.title} | "
                f"Worker: {task.worker_model} | "
                f"Rating: {task.rating}/10 | "
                f"Status: {task.status}"
            )

        return "\n".join(summary_parts)

    def get_next_pending_task(self) -> Optional[Task]:
        """Get the next task that needs execution."""
        for num in sorted(self.tasks.keys()):
            task = self.tasks[task_number]
            if task.status in ("pending", "remediation"):
                return task
        return None

    def mark_task_in_progress(self, task_number: int) -> None:
        """Mark a task as currently being worked on."""
        task = self.tasks.get(task_number)
        if task:
            task.status = "in_progress"
            self.state.current_task = task_number
            self._save_task_list()
            self._save_state()

    def update_state_status(self, status: str) -> None:
        """Update overall project status."""
        self.state.status = status
        self._save_state()

    def add_remediation_task(self, original_task_num: int, 
                            instructions: str) -> Task:
        """Create a remediation task based on Brain feedback."""
        # Find next available task number
        max_num = max(self.tasks.keys()) if self.tasks else 0
        new_num = max_num + 1

        original = self.tasks.get(original_task_num)
        worker = original.worker_model if original else "codellama_7b"

        remediation = Task(
            number=new_num,
            worker_model=worker,
            title=f"REMEDIATION: Task {original_task_num}",
            subtasks=[
                f"Review and fix issues from Task {original_task_num}",
                instructions,
                "Ensure all previous issues are resolved",
                "Output complete corrected implementation"
            ],
            pre_instructions=f"""This is a REMEDIATION task. 
The previous attempt (Task {original_task_num}) was rated below 6/10.
Brain's feedback: {instructions}

You MUST:
1. Read the previous result carefully
2. Fix ALL identified issues
3. Output complete, corrected code/documentation
4. Do not reference the previous failure — just produce the correct output""",
            notes=f"Remediation for Task {original_task_num}",
            status="pending"
        )

        self.add_task(remediation)

        self._write_master_log(
            f"Remediation task {new_num} created for Task {original_task_num}. "
            f"Reason: {instructions[:200]}"
        )

        return remediation

    def _write_task_file(self, task: Task) -> None:
        """Generate the markdown task file for a worker."""
        task_file = self.tasks_dir / f"task_{task.number:03d}_{task.worker_model}.md"

        # Get context from previous tasks
        context = self.get_context_for_task(task.number)

        content = f"""# Task {task.number}: {task.title}

**Assigned Worker:** {task.worker_model}
**Max Output Tokens:** {task.max_tokens}
**Project:** {self.project_name}

---

## Pre-Instructions (READ CAREFULLY)
{task.pre_instructions}

---

## Context from Previous Tasks
{context}

---

## Your Subtasks
"""
        for i, st in enumerate(task.subtasks, 1):
            content += f"### {i}. {st}\n\n"

        content += f"""
---

## Output Requirements
1. Write your complete response in markdown format
2. Include all code with proper syntax highlighting
3. Explain your reasoning for architectural decisions
4. If you create files, list them with their purposes
5. End with a summary of what was accomplished

## Notes for Brain
{task.notes if task.notes else "No special notes."}

---
*Task generated by Brain Loader at {datetime.now().isoformat()}*
"""

        with open(task_file, "w") as f:
            f.write(content)

    def _write_master_log(self, entry: str) -> None:
        """Append to master log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.master_log, "a") as f:
            f.write(f"\n\n---\n**[{timestamp}]**\n\n{entry}\n")

    def _save_state(self) -> None:
        """Save project state to JSON."""
        if self.state:
            with open(self.state_file, "w") as f:
                json.dump(asdict(self.state), f, indent=2)

    def _save_task_list(self) -> None:
        """Save all tasks to JSON."""
        tasks_data = {str(k): asdict(v) for k, v in self.tasks.items()}
        with open(self.task_list_file, "w") as f:
            json.dump(tasks_data, f, indent=2)

    def load_existing_project(self) -> bool:
        """Try to load existing project state. Returns True if found."""
        if not self.state_file.exists():
            return False

        try:
            with open(self.state_file, "r") as f:
                state_data = json.load(f)
            self.state = ProjectState(**state_data)

            if self.task_list_file.exists():
                with open(self.task_list_file, "r") as f:
                    tasks_data = json.load(f)
                self.tasks = {
                    int(k): Task(**v) for k, v in tasks_data.items()
                }

            logger.info("[TaskManager] Loaded existing project with %d tasks", 
                       len(self.tasks))
            return True

        except Exception as e:
            logger.error("[TaskManager] Failed to load existing project: %s", e)
            return False

    def get_project_summary(self) -> str:
        """Get human-readable project summary."""
        if not self.state:
            return "No active project."

        lines = [
            f"Project: {self.state.project_name}",
            f"Status: {self.state.status}",
            f"Tasks: {self.state.completed_tasks}/{self.state.total_tasks} completed",
            f"Current Task: {self.state.current_task}",
            f"Brain: {self.state.brain_model}",
            "",
            "Task Breakdown:"
        ]

        for num in sorted(self.tasks.keys()):
            t = self.tasks[num]
            status_icon = "✅" if t.status == "completed" else "⏳" if t.status == "pending" else "🔧"
            lines.append(
                f"  {status_icon} Task {num}: {t.title} "
                f"({t.worker_model}) — {t.status}"
            )

        return "\n".join(lines)
