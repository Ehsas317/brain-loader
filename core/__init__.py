#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FORGE  — FILE: core/__init__.py                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Forge (formerly Brain Loader v1)
# REPO:       https://github.com/Ehsas317/forge
# WHAT:       It hammers out a project sequentially, task by task, with
#             iterative review. A forge is slow, hot, and precise.
#
# THIS FILE:
#   Core package initializer for Forge. Exports the main classes used
#   throughout the orchestration system.
#
# HOW TO USE FORGE:
#   1. Install:    pip install -r requirements.txt
#   2. Configure:  Edit config.yaml with your API tokens
#   3. Run:        python main.py "Your project description"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Forge — Core Package

Exposes the main orchestration classes:
- BrainOrchestrator: Central controller
- MLXManager: Model lifecycle management
- TaskManager: Task decomposition and tracking
"""

from core.orchestrator import BrainOrchestrator
from core.model_manager import MLXManager, ModelConfig
from core.task_manager import TaskManager, Task

__all__ = [
    "BrainOrchestrator",
    "MLXManager",
    "ModelConfig",
    "TaskManager",
    "Task",
]
