"""
Brain Orchestrator
The main engine that runs the Brain → Worker → Brain loop.
"""

import os
import re
import time
import yaml
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from core.model_manager import MLXModelManager, ModelConfig
from core.task_manager import TaskManager, Task, ProjectState

logger = logging.getLogger(__name__)


class BrainOrchestrator:
    """
    Main orchestration engine.

    Flow:
    1. Load Brain → Create master plan
    2. For each task:
       a. Brain writes task file
       b. Offload Brain → Load Worker
       c. Worker executes → Writes result
       d. Offload Worker → Load Brain
       e. Brain reviews → Rates → Notes for future
    3. Final Brain review
    4. If incomplete → New tasks → Repeat from 2
    5. If complete → Telegram notification
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.model_mgr = MLXModelManager(
            gc_sleep=self.config["memory"]["gc_sleep_seconds"],
            aggressive_cleanup=self.config["memory"]["aggressive_cleanup"]
        )

        self.task_mgr = TaskManager(
            project_name=self.config["project"]["name"],
            tasks_dir=self.config["project"]["tasks_dir"],
            logs_dir=self.config["project"]["logs_dir"]
        )

        # Load prompts
        self.brain_system_prompt = self._load_prompt("brain_system.txt")
        self.worker_system_prompt = self._load_prompt("worker_system.txt")
        self.brain_review_prompt = self._load_prompt("brain_review.txt")

        # Model configs
        self.brain_config = self._create_model_config("brain")
        self.worker_configs = {
            key: self._create_worker_config(key, val)
            for key, val in self.config["workers"].items()
        }

        # Telegram notifier (lazy loaded)
        self._telegram = None

        logger.info("[Orchestrator] Initialized. Workers available: %s", 
                   list(self.worker_configs.keys()))

    def run(self, app_idea: str, resume: bool = False) -> None:
        """
        Main entry point. Run the full project pipeline.

        Args:
            app_idea: Description of the app to build
            resume: Try to resume from existing project state
        """
        self._notify(f"🧠 *Brain Loader Started*\n\nApp: _{app_idea}_")

        if resume and self.task_mgr.load_existing_project():
            logger.info("[Orchestrator] Resuming existing project...")
            self._notify("📂 Resuming existing project...")
            self._resume_execution()
        else:
            self._start_new_project(app_idea)

    def _start_new_project(self, app_idea: str) -> None:
        """Start a brand new project from scratch."""
        # Initialize project
        self.task_mgr.initialize_project(app_idea, self.brain_config.path)

        # PHASE 1: Brain creates master plan
        self._notify("📝 Brain is creating the master plan...")
        master_plan = self._brain_create_master_plan(app_idea)

        self.task_mgr.save_master_plan(master_plan)

        # Parse tasks from plan
        tasks = self._parse_master_plan(master_plan)

        if not tasks:
            raise RuntimeError("Brain failed to generate valid tasks!")

        self._notify(
            f"📋 *Master Plan Complete*\n"
            f"Total tasks: *{len(tasks)}*\n"
            f"Ready to execute."
        )

        # Add all tasks to manager
        for task in tasks:
            self.task_mgr.add_task(task)

        # PHASE 2: Execute tasks sequentially
        self._execute_tasks()

    def _resume_execution(self) -> None:
        """Resume from saved state."""
        self._notify("🔄 Resuming task execution...")
        self._execute_tasks()

    def _execute_tasks(self) -> None:
        """Main execution loop."""
        self.task_mgr.update_state_status("executing")

        while True:
            task = self.task_mgr.get_next_pending_task()

            if not task:
                # No more pending tasks — do final review
                self._do_final_review()
                break

            self._execute_single_task(task)

    def _execute_single_task(self, task: Task) -> None:
        """
        Execute one task through the full Brain → Worker → Brain cycle.
        """
        task_num = task.number
        worker_key = task.worker_model

        logger.info("=" * 60)
        logger.info("[Orchestrator] Executing Task %d: %s", task_num, task.title)
        logger.info("=" * 60)

        self.task_mgr.mark_task_in_progress(task_num)
        self._notify(
            f"⚙️ *Task {task_num}/{self.task_mgr.state.total_tasks}*\n"
            f"Title: {task.title}\n"
            f"Worker: `{worker_key}`"
        )

        # STEP 1: Brain is already loaded (from plan creation or previous cycle)
        # Brain adds any notes/adjustments to the task file
        self._brain_enhance_task(task)

        # STEP 2: Offload Brain, Load Worker
        worker_config = self.worker_configs.get(worker_key)
        if not worker_config:
            logger.warning("Unknown worker %s, using llama3_8b", worker_key)
            worker_config = self.worker_configs.get("llama3_8b", self.brain_config)

        logger.info("[Orchestrator] Swapping Brain → %s", worker_key)
        self.model_mgr.offload_model()
        self.model_mgr.load_model(worker_config)

        # STEP 3: Worker executes
        task_content = self.task_mgr.get_task_file_content(task_num)
        worker_prompt = self._build_worker_prompt(task_content, task)

        try:
            worker_output = self.model_mgr.generate(
                prompt=worker_prompt,
                max_tokens=task.max_tokens,
                temperature=worker_config.temperature
            )
        except Exception as e:
            logger.error("[Orchestrator] Worker generation failed: %s", e)
            worker_output = f"ERROR: Worker failed to generate output.\n\nException: {str(e)}"

        # STEP 4: Save worker result
        # (Full review happens after Brain reloads)
        temp_result_path = self.task_mgr.write_task_result(
            task_num, worker_output, rating=None, review_notes=""
        )

        logger.info("[Orchestrator] Worker output saved. Swapping back to Brain...")

        # STEP 5: Offload Worker, Reload Brain
        self.model_mgr.offload_model()
        self.model_mgr.load_model(self.brain_config)

        # STEP 6: Brain reviews
        self._notify(f"🧠 Brain reviewing Task {task_num}...")
        review = self._brain_review_task(task_num)

        rating = self._extract_rating(review)

        # Re-save result with rating and review
        self.task_mgr.write_task_result(task_num, worker_output, rating, review)

        # STEP 7: Brain adds notes for future tasks
        future_notes = self._extract_future_notes(review)
        if future_notes:
            self._apply_future_notes(future_notes)

        # Notification
        status_icon = "✅" if rating >= 6 else "⚠️"
        self._notify(
            f"{status_icon} *Task {task_num} Reviewed*\n"
            f"Rating: *{rating}/10*\n"
            f"{'Passing — moving on.' if rating >= 6 else 'Needs remediation.'}"
        )

        # STEP 8: Handle remediation
        if rating < 6:
            logger.info("[Orchestrator] Task %d rated %d/10. Creating remediation.", 
                       task_num, rating)
            remediation = self.task_mgr.add_remediation_task(task_num, review)
            self._notify(
                f"🔧 Remediation task {remediation.number} created.\n"
                f"Will re-attempt after current queue."
            )

        logger.info("[Orchestrator] Task %d cycle complete.", task_num)

    def _brain_create_master_plan(self, app_idea: str) -> str:
        """
        Load Brain and generate the master task plan.
        """
        logger.info("[Orchestrator] Loading Brain for master planning...")
        self.model_mgr.load_model(self.brain_config)

        prompt = f"""{self.brain_system_prompt}

## Your Current Job
Create a complete development plan for this app:

**App Description:**
{app_idea}

Create between {self.config["tasks"]["min_tasks"]} and {self.config["tasks"]["max_tasks"]} tasks.

Available workers:
- nemotron_15b: Complex architecture, system design (9GB RAM)
- codellama_7b: Code implementation (4.5GB RAM) 
- deepseek_coder_6.7b: Math/logic heavy code (4GB RAM)
- llama3_8b: General tasks, docs, tests (4.5GB RAM)

Output ONLY the task list in the specified format. No extra commentary.
"""

        plan = self.model_mgr.generate(
            prompt=prompt,
            max_tokens=self.config["brain"]["max_tokens_plan"],
            temperature=self.config["brain"]["temperature_plan"]
        )

        logger.info("[Orchestrator] Master plan generated (%d chars)", len(plan))
        return plan

    def _brain_enhance_task(self, task: Task) -> None:
        """
        Brain reviews task before worker execution.
        Adds any last-minute adjustments based on current project state.
        """
        # Get context of what has been done so far
        context = self.task_mgr.get_context_for_task(task.number, max_chars=3000)

        prompt = f"""{self.brain_system_prompt}

## Current Task to Enhance
Task {task.number}: {task.title}
Worker: {task.worker_model}

Current subtasks:
{chr(10).join(f"{i+1}. {st}" for i, st in enumerate(task.subtasks))}

## Context from Previous Work
{context}

## Your Job
Review this task before it goes to the worker. Output ONLY:
1. Any additions/modifications to subtasks (keep original + add)
2. Any critical notes the worker MUST know
3. Cross-references to other tasks

If no changes needed, say "NO_CHANGES".
"""

        enhancement = self.model_mgr.generate(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.4
        )

        if "NO_CHANGES" not in enhancement:
            # Parse enhancements and update task
            # (Simplified — in production, parse structured output)
            task.notes += f"\n\nBrain pre-flight notes:\n{enhancement}"
            self.task_mgr._save_task_list()

    def _brain_review_task(self, task_num: int) -> str:
        """
        Brain reviews completed worker output.
        """
        result_content = self.task_mgr.get_result_content(task_num)
        task = self.task_mgr.tasks.get(task_num)

        if not result_content:
            return "RATING: 1\nNo output generated by worker."

        context = self.task_mgr.get_context_for_task(task_num)

        prompt = f"""{self.brain_review_prompt}

## Task Being Reviewed
Task {task.number}: {task.title}
Worker: {task.worker_model}
Original subtasks:
{chr(10).join(f"{i+1}. {st}" for i, st in enumerate(task.subtasks))}

## Worker Output
{result_content[:8000]}  # Truncate if very long

## Context from Previous Tasks
{context}

Provide your review in the specified format.
"""

        review = self.model_mgr.generate(
            prompt=prompt,
            max_tokens=self.config["brain"]["max_tokens_review"],
            temperature=self.config["brain"]["temperature_review"]
        )

        return review

    def _do_final_review(self) -> None:
        """
        Final review after all tasks complete.
        Brain decides if project is done or needs more work.
        """
        logger.info("[Orchestrator] All tasks processed. Starting final review...")
        self.task_mgr.update_state_status("reviewing")
        self._notify("🔍 *Final Review* — Brain evaluating entire project...")

        summary = self.task_mgr.get_all_results_summary()

        prompt = f"""{self.brain_system_prompt}

## Final Review Phase

All tasks have been executed. Here is the summary:

{summary}

## Your Job
Review the ENTIRE project holistically:
1. Are all features implemented cohesively?
2. Are there gaps or missing pieces?
3. Is the code quality consistent?
4. Would this work as a production app?

If the project is complete and production-ready, output exactly:
PROJECT_COMPLETE

If improvements are needed, output:
PROJECT_INCOMPLETE
Then list the new tasks needed (in the same format as master plan).

Be honest. A half-baked project helps no one.
"""

        max_iterations = self.config["project"]["max_iterations"]
        iteration = self.task_mgr.state.iteration

        while iteration < max_iterations:
            self.task_mgr.state.iteration = iteration
            self.task_mgr._save_state()

            review = self.model_mgr.generate(
                prompt=prompt,
                max_tokens=self.config["brain"]["max_tokens_final"],
                temperature=self.config["brain"]["temperature_final"]
            )

            if "PROJECT_COMPLETE" in review:
                self._finalize_project()
                return

            # Need more tasks
            new_tasks = self._parse_master_plan(review)
            if new_tasks:
                self._notify(
                    f"🔄 *Iteration {iteration + 1}*\n"
                    f"Brain found {len(new_tasks)} improvements needed."
                )
                for task in new_tasks:
                    self.task_mgr.add_task(task)

                # Continue execution loop — increment iteration BEFORE recursing
                iteration += 1
                self.task_mgr.state.iteration = iteration
                self.task_mgr._save_state()
                self._execute_tasks()
                return
            else:
                # Couldn't parse new tasks, force completion
                logger.warning("Could not parse new tasks from review. Forcing completion.")
                self._finalize_project()
                return

            iteration += 1

        # Max iterations reached
        logger.warning("Max iterations reached. Forcing completion.")
        self._finalize_project()

    def _finalize_project(self) -> None:
        """Project is complete. Send final notification."""
        self.task_mgr.update_state_status("complete")

        summary = self.task_mgr.get_project_summary()

        # Save final summary
        final_file = Path(self.task_mgr.tasks_dir) / "FINAL_SUMMARY.md"
        with open(final_file, "w") as f:
            f.write(f"""# 🎉 Project Complete: {self.task_mgr.state.project_name}

**App:** {self.task_mgr.state.app_idea}
**Completed:** {self.task_mgr.state.created_at}
**Total Tasks:** {self.task_mgr.state.total_tasks}
**Final Status:** {self.task_mgr.state.status}

## How to Review on Your MacBook Pro

All files are located in:
```
{os.path.abspath(self.task_mgr.tasks_dir)}
```

### Key Files:
- `master_plan.md` — Original Brain plan
- `master_log.md` — Execution log with timestamps
- `task_###_*.md` — Individual task assignments
- `result_###_*.md` — Worker outputs with ratings
- `project_state.json` — Machine-readable state

### Quick Review:
```bash
# View all results
cd {os.path.abspath(self.task_mgr.tasks_dir)}
cat result_*.md

# View master log
cat master_log.md
```

## Task Breakdown
{summary}
""")

        self._notify(
            f"🎉 *PROJECT COMPLETE!*\n\n"
            f"App: _{self.task_mgr.state.app_idea}_\n"
            f"Tasks: {self.task_mgr.state.total_tasks}\n\n"
            f"📂 Review at:\n"
            f"`{os.path.abspath(self.task_mgr.tasks_dir)}`\n\n"
            f"Open `FINAL_SUMMARY.md` for full details."
        )

        logger.info("[Orchestrator] Project complete!")

    def _parse_master_plan(self, plan_text: str) -> List[Task]:
        """
        Parse Brain's master plan into Task objects.
        Handles the structured markdown format.
        """
        tasks = []

        # Split by task headers
        task_blocks = re.split(r'### Task\s*(\d+)', plan_text)

        # task_blocks[0] is preamble, then pairs of (number, content)
        for i in range(1, len(task_blocks), 2):
            if i + 1 >= len(task_blocks):
                break

            try:
                task_num = int(task_blocks[i].strip())
                content = task_blocks[i + 1]

                # Extract fields with regex
                title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?:\n|$)', content)
                worker_match = re.search(r'\*\*Worker:\*\*\s*(\w+)', content)
                tokens_match = re.search(r'\*\*Max Tokens:\*\*\s*(\d+)', content)
                pre_match = re.search(
                    r'\*\*Pre-Instructions:\*\*\s*(.+?)(?=\*\*Subtasks:|$)', 
                    content, re.DOTALL
                )
                subtasks_match = re.search(
                    r'\*\*Subtasks:\*\*(.+?)(?=\*\*Notes:|$)', 
                    content, re.DOTALL
                )
                notes_match = re.search(
                    r'\*\*Notes:\*\*\s*(.+?)(?=### Task|$)', 
                    content, re.DOTALL
                )

                title = title_match.group(1).strip() if title_match else f"Task {task_num}"
                worker = worker_match.group(1).strip() if worker_match else self.config["tasks"]["default_worker"]
                max_tokens = int(tokens_match.group(1)) if tokens_match else 8192
                pre_instructions = pre_match.group(1).strip() if pre_match else "Follow all subtasks carefully."
                notes = notes_match.group(1).strip() if notes_match else ""

                # Parse subtasks
                subtasks = []
                if subtasks_match:
                    subtask_text = subtasks_match.group(1)
                    # Find numbered items
                    subtasks = re.findall(r'^\d+\.\s*(.+?)(?=^\d+\.|$)', 
                                         subtask_text, re.MULTILINE | re.DOTALL)
                    subtasks = [s.strip() for s in subtasks if s.strip()]

                if not subtasks:
                    # Fallback: split by newlines
                    subtasks = [line.strip() for line in subtask_text.split('\n') 
                               if line.strip() and not line.strip().startswith('**')]

                task = Task(
                    number=task_num,
                    worker_model=worker,
                    title=title,
                    subtasks=subtasks if subtasks else ["Complete the assigned work"],
                    pre_instructions=pre_instructions,
                    notes=notes,
                    max_tokens=max_tokens
                )

                tasks.append(task)

            except Exception as e:
                logger.warning("Failed to parse task block %d: %s", i, e)
                continue

        # Sort by task number
        tasks.sort(key=lambda t: t.number)

        # Renumber if there are gaps
        for i, task in enumerate(tasks, 1):
            task.number = i

        logger.info("[Orchestrator] Parsed %d tasks from master plan", len(tasks))
        return tasks

    def _build_worker_prompt(self, task_content: str, task: Task) -> str:
        """Build the full prompt for a worker model."""
        return f"""{self.worker_system_prompt}

## Your Assignment

{task_content}

---

Now execute all subtasks thoroughly. Output your complete work in markdown.
"""

    def _extract_rating(self, review_text: str) -> int:
        """Extract numerical rating from Brain review."""
        match = re.search(r'RATING:\s*(\d+)(?:\s*/\s*10)?', review_text, re.IGNORECASE)
        if match:
            rating = int(match.group(1))
            return max(1, min(10, rating))  # Clamp 1-10
        return 5  # Default if not found

    def _extract_future_notes(self, review_text: str) -> List[str]:
        """Extract notes about future tasks from review."""
        notes = []

        # Look for patterns like "add X to task Y" or "task Z should..."
        patterns = [
            r'(?:add|include|move)\s+(.+?)\s+(?:to|in)\s+task\s*(\d+)',
            r'task\s*(\d+)\s+(?:should|needs|must)\s+(.+)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, review_text, re.IGNORECASE)
            for match in matches:
                notes.append(match.group(0))

        return notes

    def _apply_future_notes(self, notes: List[str]) -> None:
        """Apply Brain's notes to future tasks."""
        for note in notes:
            # Parse which task number is referenced
            match = re.search(r'task\s*(\d+)', note, re.IGNORECASE)
            if match:
                task_num = int(match.group(1))
                task = self.task_mgr.tasks.get(task_num)
                if task:
                    task.notes += f"\n\n[From previous review]: {note}"
                    self.task_mgr._save_task_list()

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt file."""
        prompt_path = Path("prompts") / filename
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                return f.read()
        return ""  # Return empty if not found

    def _create_model_config(self, config_key: str) -> ModelConfig:
        """Create ModelConfig from config dict."""
        cfg = self.config[config_key]
        return ModelConfig(
            path=cfg["model_path"],
            max_tokens=cfg.get("max_tokens_plan", 4096),
            temperature=cfg.get("temperature_plan", 0.7),
            description="Brain orchestrator model",
            ram_estimate_gb=4.5  # 8B model at 4bit
        )

    def _create_worker_config(self, key: str, cfg: dict) -> ModelConfig:
        """Create ModelConfig for a worker."""
        return ModelConfig(
            path=cfg["model_path"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            description=cfg["description"],
            ram_estimate_gb=cfg["ram_estimate_gb"]
        )

    def _notify(self, message: str) -> None:
        """Send Telegram notification."""
        if self._telegram is None:
            try:
                from utils.telegram_notify import TelegramNotifier
                self._telegram = TelegramNotifier(
                    self.config["telegram"]["token"],
                    self.config["telegram"]["chat_id"]
                )
            except Exception as e:
                logger.warning("Telegram not available: %s", e)
                return

        try:
            self._telegram.send(message)
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
