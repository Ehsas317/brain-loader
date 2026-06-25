#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FORGE  — FILE: core/orchestrator.py                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Forge (formerly Brain Loader v1)
# REPO:       https://github.com/Ehsas317/forge
# WHAT:       It hammers out a project sequentially, task by task, with
#             iterative review. A forge is slow, hot, and precise.
#
# THIS FILE:
#   Brain Orchestrator — the central controller of Forge. Maintains project
#   context, decomposes goals into tasks, coordinates worker models, and
#   reviews all output before approval. The brain of the operation.
#
# KEY COMPONENTS:
#   - BrainOrchestrator: Main orchestration class
#   - run(): Main entry point for build execution
#   - _decompose(): Breaks down project goals into actionable tasks
#   - _execute_task(): Runs a single task through the worker pipeline
#   - _review_output(): Brain reviews worker output for quality
#
# HOW TO USE FORGE:
#   1. Install:    pip install -r requirements.txt
#   2. Configure:  Edit config.yaml with your API tokens
#   3. Run:        python main.py "Your project description"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Forge — Brain Orchestrator

Central controller that maintains context and state, decomposes goals
into tasks, coordinates worker models, and reviews all output.
"""

import os
import sys
import yaml
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from core.model_manager import MLXManager, ModelConfig
from core.task_manager import Task, TaskManager
from utils.telegram_notify import TelegramNotifier

logger = logging.getLogger("forge.orchestrator")


@dataclass
class WorkerOutput:
    """Output from a worker model execution."""
    task_id: str
    worker_key: str
    content: str
    tokens_used: int
    generation_time: float
    approved: bool = False
    review_feedback: str = ""


@dataclass
class ProjectState:
    """Current state of the project being built."""
    app_idea: str = ""
    current_phase: str = "planning"  # planning | execution | review | done
    tasks: List[Dict] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    brain_context: str = ""  # Accumulated context from the Brain
    working_tree: str = ""  # Complete file tree of the project


class BrainOrchestrator:
    """
    Forge Brain Orchestrator

    Central controller for the multi-agent build system. The Brain
    handles planning, task decomposition, and quality review while
    worker models handle the actual code generation.

    Usage:
        orchestrator = BrainOrchestrator(config_path="config.yaml")
        orchestrator.run("Build a fitness app")
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.state = ProjectState()
        self.task_manager = TaskManager()

        # Initialize model manager
        model_configs = {
            key: ModelConfig(**cfg)
            for key, cfg in self.config.get("models", {}).items()
        }
        self.model_manager = MLXManager(
            config=model_configs,
            device_ram_gb=25.0  # Leave headroom on 32GB system
        )

        # Worker assignments (task type → model key)
        self.worker_assignments = {
            "frontend": "forge-coder",
            "backend": "forge-coderx",
            "devops": "forge-devops",
            "security": "forge-security",
            "docs": "forge-docs",
            "testing": "forge-qa",
            "planning": "forge-brain",
            "review": "forge-brain",
        }

        # Optional: Telegram notifier
        telegram_cfg = self.config.get("telegram", {})
        if telegram_cfg.get("enabled"):
            self.notifier = TelegramNotifier(
                bot_token=telegram_cfg.get("bot_token", ""),
                chat_id=telegram_cfg.get("chat_id", ""),
            )
        else:
            self.notifier = None

        # Setup directories
        Path("./memory").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)

        logger.info("[Orchestrator] Forge initialized with %d models", len(model_configs))

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def run(self, app_idea: str, resume: bool = False):
        """Main entry point for the build process."""
        if resume:
            self._handle_resume()
        else:
            self.state = ProjectState(app_idea=app_idea)
            self._save_state()

        logger.info("[Orchestrator] Starting build: %s", self.state.app_idea)
        self._notify(f"🔥 Forge starting: {self.state.app_idea}")

        # Phase 1: Planning
        self.state.current_phase = "planning"
        plan = self._create_plan(self.state.app_idea)
        self.state.tasks = self.task_manager.decompose_plan(plan)
        logger.info("[Orchestrator] Plan created with %d tasks", len(self.state.tasks))
        self._notify(f"📋 Plan created: {len(self.state.tasks)} tasks")

        # Phase 2: Execution
        self.state.current_phase = "execution"
        for task in self.state.tasks:
            if task["id"] in self.state.completed_tasks:
                logger.info("[Orchestrator] Skipping completed task: %s", task["id"])
                continue

            self._execute_task(task)
            self.state.completed_tasks.append(task["id"])
            self._save_state()

        # Phase 3: Final Review
        self.state.current_phase = "review"
        self._final_review()

        self.state.current_phase = "done"
        logger.info("[Orchestrator] Build complete!")
        self._notify("✅ Forge build complete!")
        self._save_state()

    def _handle_resume(self):
        """Handle resume with validation."""
        """Handle resuming from a previous state."""
        if not Path("memory/working_tree.md").exists() or not Path("memory/plan.md").exists():
            self.logger.info("No previous state found — starting fresh")
            return False
        try:
            with open("memory/working_tree.md", 'r') as f:
                self.state.working_tree = f.read()
            with open("memory/plan.md", 'r') as f:
                plan_data = f.read()
            # Parse plan and restore tasks
            self.state.tasks = self.task_manager.parse_plan(plan_data)
            logger.info("[Orchestrator] Resumed with %d tasks", len(self.state.tasks))
            return True
        except Exception as e:
            logger.error("[Orchestrator] Failed to resume: %s", e)
            return False

    def _create_plan(self, app_idea: str) -> str:
        """Create a project plan using the Brain model."""
        logger.info("[Orchestrator] Creating plan with Brain model...")

        prompt = f"""You are the Brain of Forge, an expert software architect.

Create a detailed development plan for this app idea:
"{app_idea}"

Break it down into specific, actionable tasks. For each task:
1. Task ID (e.g., T001, T002)
2. Task type (frontend/backend/devops/security/docs/testing)
3. Clear description of what needs to be built
4. Dependencies on other tasks

Format your response as a structured plan."""

        with self.model_manager.model_scope("forge-brain") as model:
            plan = model.generate(prompt, max_tokens=4096)

        # Save plan
        with open("memory/plan.md", 'w') as f:
            f.write(plan)

        return plan

    def _execute_task(self, task: Dict):
        """Execute a single task using the appropriate worker."""
        task_id = task["id"]
        task_type = task.get("type", "general")
        description = task.get("description", "")

        logger.info("[Orchestrator] Executing task %s (%s)", task_id, task_type)
        self._notify(f"🔨 {task_id}: {description[:50]}...")

        # Get the assigned worker model
        worker_key = self.worker_assignments.get(task_type, "forge-coder")

        # Build the prompt
        prompt = self._build_worker_prompt(task)

        # Execute with the worker model
        with self.model_manager.model_scope(worker_key) as model:
            output = model.generate(prompt, max_tokens=4096)

        # Review the output
        review_result = self._review_output(task, output)

        if review_result["approved"]:
            logger.info("[Orchestrator] Task %s approved", task_id)
            self._save_task_output(task_id, output)
        else:
            logger.info("[Orchestrator] Task %s needs revision: %s", task_id, review_result["feedback"])
            # Could implement revision loop here
            self._save_task_output(task_id, output + "\n\n# REVISION NEEDED:\n" + review_result["feedback"])

    def _build_worker_prompt(self, task: Dict) -> str:
        """Build a detailed prompt for a worker model."""
        return f"""You are an expert {task.get('type', 'software')} developer.

Task: {task.get('description', '')}

Context from previous tasks:
{self.state.working_tree[:2000]}

Write complete, production-ready code. Include:
- Full implementation
- Error handling
- Type hints where appropriate
- Docstrings for public functions

Output:"""

    def _review_output(self, task: Dict, output: str) -> Dict:
        """Have the Brain review worker output for quality."""
        prompt = f"""Review this code/output for quality:

Task: {task.get('description', '')}

Output to review:
```
{output[:3000]}
```

Check for:
1. Correctness — does it solve the task?
2. Code quality — clean, maintainable?
3. Completeness — all edge cases handled?
4. Security — any obvious vulnerabilities?

Respond with:
- APPROVED or NEEDS_REVISION
- Brief feedback"""

        with self.model_manager.model_scope("forge-brain") as model:
            review = model.generate(prompt, max_tokens=2048)

        approved = "APPROVED" in review.upper()
        return {"approved": approved, "feedback": review}

    def _final_review(self):
        """Perform final review of the complete project."""
        logger.info("[Orchestrator] Performing final review...")

        # Gather all outputs
        all_outputs = []
        tasks_dir = Path("memory/tasks")
        if tasks_dir.exists():
            for task_file in sorted(tasks_dir.glob("*.md")):
                with open(task_file, 'r') as f:
                    all_outputs.append(f.read())

        combined = "\n\n".join(all_outputs)

        prompt = f"""Perform a final review of this complete project:

{combined[:4000]}

Provide:
1. Overall assessment
2. Any critical issues
3. Suggestions for improvement"""

        with self.model_manager.model_scope("forge-brain") as model:
            final_review = model.generate(prompt, max_tokens=4096)

        with open("memory/review.md", 'w') as f:
            f.write(final_review)

        self._notify("🔍 Final review complete")

    def _save_task_output(self, task_id: str, output: str):
        """Save task output to disk."""
        tasks_dir = Path("memory/tasks")
        tasks_dir.mkdir(exist_ok=True)

        task_file = tasks_dir / f"{task_id}.md"
        with open(task_file, 'w') as f:
            f.write(output)

        logger.info("[Orchestrator] Saved output for %s", task_id)

    def _save_state(self):
        """Save current project state."""
        state_dict = {
            "app_idea": self.state.app_idea,
            "current_phase": self.state.current_phase,
            "completed_tasks": self.state.completed_tasks,
        }
        with open("memory/state.json", 'w') as f:
            json.dump(state_dict, f, indent=2)

    def _notify(self, message: str):
        """Send notification via Telegram if enabled."""
        if self.notifier:
            try:
                self.notifier.send(message)
            except Exception as e:
                logger.warning("Failed to send notification: %s", e)
