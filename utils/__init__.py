#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FORGE  — FILE: utils/__init__.py                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Forge (formerly Brain Loader v1)
# REPO:       https://github.com/Ehsas317/forge
# WHAT:       It hammers out a project sequentially, task by task, with
#             iterative review. A forge is slow, hot, and precise.
#
# THIS FILE:
#   Utils package initializer for Forge. Exports utility classes used
#   across the orchestration system.
#
# HOW TO USE FORGE:
#   1. Install:    pip install -r requirements.txt
#   2. Configure:  Edit config.yaml with your API tokens
#   3. Run:        python main.py "Your project description"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Forge — Utils Package

Utility modules for the Forge orchestrator:
- TelegramNotifier: Real-time build notifications
"""

from utils.telegram_notify import TelegramNotifier

__all__ = ["TelegramNotifier"]
