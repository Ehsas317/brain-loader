#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FORGE  — FILE: main.py                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Forge (formerly Brain Loader v1)
# REPO:       https://github.com/Ehsas317/forge
# WHAT:       It hammers out a project sequentially, task by task, with
#             iterative review. A forge is slow, hot, and precise.
#
# THIS FILE:
#   Entry point for the Forge multi-agent AI orchestrator.
#   Parses CLI args, sets up logging, initializes the BrainOrchestrator,
#   and kicks off the build process.
#
# HOW TO USE FORGE:
#   1. Install:    pip install -r requirements.txt
#   2. Configure:  Edit config.yaml with your API tokens
#   3. Run:        python main.py "Build a React Native fitness app"
#
# HARDWARE TARGET: MacBook Pro M1 Max 32GB (25GB allocated)
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Forge — Main Entry Point

Usage:
    python main.py "Build a React Native fitness app with AI meal planner"
    python main.py --resume
    python main.py --idea "My app idea" --config custom_config.yaml

Hardware: MacBook Pro M1 Max 32GB (25GB allocated)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import BrainOrchestrator


def setup_logging(logs_dir: str = "./logs"):
    """Configure logging to file and console."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    log_file = Path(logs_dir) / f"forge_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    return log_file


def main():
    parser = argparse.ArgumentParser(
        description="Forge — MLX Multi-Agent Orchestrator"
    )
    parser.add_argument(
        "idea",
        nargs="?",
        help="Description of the app/project to build"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing project state"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available worker models and exit"
    )

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("FORGE — MLX Multi-Agent Orchestrator")
    logger.info("Hardware: MacBook Pro M1 Max 32GB (25GB allocated)")
    logger.info("=" * 70)

    # Check config exists
    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        logger.info("Copy config.yaml.example to config.yaml and fill in your tokens.")
        sys.exit(1)

    # Initialize orchestrator
    try:
        orchestrator = BrainOrchestrator(config_path=args.config)
    except Exception as e:
        logger.error("Failed to initialize orchestrator: %s", e)
        sys.exit(1)

    # List models mode
    if args.list_models:
        print("\nAvailable Worker Models:")
        print("-" * 50)
        for key, cfg in orchestrator.worker_configs.items():
            print(f"  {key:20s} — {cfg.description}")
            print(f"    Path: {cfg.path}")
            print(f"    RAM: ~{cfg.ram_estimate_gb}GB | Max tokens: {cfg.max_tokens}")
            print()
        print(f"\nBrain Model: {orchestrator.brain_config.path}")
        sys.exit(0)

    # Determine mode
    if args.resume:
        logger.info("Resuming existing project...")
        orchestrator.run(app_idea="", resume=True)
    elif args.idea:
        logger.info("Starting new project: %s", args.idea)
        orchestrator.run(app_idea=args.idea, resume=False)
    else:
        # Interactive mode
        print("\n🔥 Forge — MLX Multi-Agent Orchestrator")
        print("=" * 50)
        print()
        print("What app/project would you like to build?")
        print("Example: 'A React Native fitness app with AI meal planner'")
        print()

        idea = input("> ").strip()

        if not idea:
            print("No idea provided. Exiting.")
            sys.exit(0)

        logger.info("Starting new project: %s", idea)
        orchestrator.run(app_idea=idea, resume=False)

    logger.info("Forge finished. Log saved to: %s", log_file)
    print(f"\n✅ Done! Check logs at: {log_file}")


if __name__ == "__main__":
    main()
