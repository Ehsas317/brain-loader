#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FORGE  — FILE: core/model_manager.py                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Forge (formerly Brain Loader v1)
# REPO:       https://github.com/Ehsas317/forge
# WHAT:       It hammers out a project sequentially, task by task, with
#             iterative review. A forge is slow, hot, and precise.
#
# THIS FILE:
#   MLX Model Manager — handles loading, generation, and memory management
#   for local LLM models on Apple Silicon. Supports multiple model configs
#   with dynamic loading/unloading to stay within RAM limits.
#
# KEY COMPONENTS:
#   - MLXManager: Main class for model lifecycle management
#   - generate(): Primary text generation with structured output support
#   - ModelConfig: Pydantic model for model configuration
#
# HOW TO USE FORGE:
#   1. Install:    pip install -r requirements.txt
#   2. Configure:  Edit config.yaml with your model paths
#   3. Run:        python main.py "Your project description"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Forge — MLX Model Manager

Handles loading, generation, and memory management for local LLM models.

RECOMMENDED USAGE: Use model_scope() context manager for guaranteed cleanup:
    with manager.model_scope("forge-coder") as m:
        result = m.generate("Write a React component...")
    # Model is automatically unloaded here
"""

import gc
import re
import time
import json
import logging
import weakref
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate as mlx_generate
from mlx_lm.utils import generate_step
from transformers import AutoTokenizer

logger = logging.getLogger("forge.model_manager")


@dataclass
class ModelConfig:
    """Configuration for a single Forge model."""
    description: str
    path: str
    max_tokens: int
    ram_estimate_gb: float
    # Optional overrides
    temperature: float = 0.0
    stop_sequences: list = field(default_factory=lambda: ["</s>", "<|im_end|>"])
    system_prompt: str = ""


class MLXManager:
    """
    Forge MLX Model Manager

    Manages the lifecycle of MLX models including loading, generation,
    and memory management. Designed for Apple Silicon (M1/M2/M3 Max).

    Usage:
        manager = MLXManager(config)
        manager.load_model("forge-coder")
        result = manager.generate("Write a React component...")
        manager.unload_model()  # Free memory

    BEST PRACTICE: Use model_scope() context manager for automatic cleanup:
        with manager.model_scope("forge-coder") as m:
            result = m.generate("...")
    """

    def __init__(self, config: Dict[str, ModelConfig], device_ram_gb: float = 32.0):
        self.config = config
        self.device_ram_gb = device_ram_gb
        self.current_model_key: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.peft_config = None

        # Performance tracking
        self.load_times: Dict[str, float] = {}
        self.gen_stats = {"calls": 0, "tokens": 0, "errors": 0}
        
        # Track instances for cleanup warnings
        self._cleaned_up = False

    def load_model(self, model_key: str) -> None:
        """Load a model by key from config."""
        if model_key not in self.config:
            raise ValueError(f"Unknown model key: {model_key}")

        if self.current_model_key == model_key and self.model is not None:
            logger.info("[MLXManager] Model %s already loaded", model_key)
            return

        # Unload current model to free memory
        if self.model is not None:
            self.unload_model()

        model_cfg = self.config[model_key]
        logger.info("[MLXManager] Loading model: %s (%s)", model_key, model_cfg.description)
        logger.info("[MLXManager] Loading model: %s", model_cfg.path)

        if not Path(model_cfg.path).exists():
            logger.error("[MLXManager] Model path does not exist: %s", model_cfg.path)
            raise FileNotFoundError(f"Model not found: {model_cfg.path}")

        start = time.time()

        try:
            self.model, self.tokenizer = load(model_cfg.path)
            self.current_model_key = model_key

            load_time = time.time() - start
            self.load_times[model_key] = load_time
            logger.info("[MLXManager] Model loaded in %.1fs", load_time)

        except Exception as e:
            logger.error("[MLXManager] Failed to load model %s: %s", model_key, e)
            self.gen_stats["errors"] += 1
            raise

    def unload_model(self) -> None:
        """Unload current model and free memory."""
        if self.model is not None:
            logger.info("[MLXManager] Unloading model: %s", self.current_model_key)
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_model_key = None
            self._cleaned_up = True
            gc.collect()
            mx.clear_cache()
            logger.info("[MLXManager] Model unloaded, memory freed")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ) -> str:
        """Generate text using the currently loaded model."""
        if not self.current_model_key or self.current_model_key not in self.config:
            logger.error("[MLXManager] Invalid or missing model key: %s", self.current_model_key)
            return ""

        model_cfg = self.config[self.current_model_key]

        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        max_tokens = max_tokens or model_cfg.max_tokens

        logger.info(
            "[MLXManager] Generating (max_tokens=%d, model=%s)",
            max_tokens, self.current_model_key
        )

        start = time.time()

        try:
            result = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=model_cfg.temperature,
                verbose=verbose,
            )

            gen_time = time.time() - start
            token_count = len(self.tokenizer.encode(result))

            self.gen_stats["calls"] += 1
            self.gen_stats["tokens"] += token_count

            logger.info(
                "[MLXManager] Generated %d tokens in %.1fs",
                token_count, gen_time
            )
            logger.info("[MLXManager] Generation complete. Output length: %d chars", len(result) if result else 0)
            if not result:
                logger.warning("[MLXManager] Generation returned empty result")

            return result

        except Exception as e:
            logger.error("[MLXManager] Generation failed: %s", e)
            self.gen_stats["errors"] += 1
            raise

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate structured output (JSON) using the current model."""
        schema_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"JSON Response:"
        )

        raw = self.generate(schema_prompt, max_tokens=max_tokens)

        # Extract JSON from response
        try:
            # Try to find JSON block
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[MLXManager] Failed to parse JSON, returning raw")
            return {"raw_response": raw}

    @contextmanager
    def model_scope(self, model_key: str):
        """
        Context manager for loading a model, yielding, then unloading.
        
        This is the RECOMMENDED way to use MLXManager as it guarantees
        cleanup even if exceptions occur.
        
        Usage:
            with manager.model_scope("forge-coder") as m:
                result = m.generate("Write a React component...")
        """
        self.load_model(model_key)
        try:
            yield self
        finally:
            self.unload_model()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        active_mem = mx.metal.get_active_memory() / 1e9
        peak_mem = mx.metal.get_peak_memory() / 1e9
        return {
            "active_gb": active_mem,
            "peak_gb": peak_mem,
            "device_total_gb": self.device_ram_gb,
            "utilization_pct": (active_mem / self.device_ram_gb) * 100,
        }

    def cleanup(self) -> None:
        """Explicit cleanup. Call this when done with the manager."""
        if self.model is not None:
            self.unload_model()
        self._cleaned_up = True

    def __del__(self):
        """
        DEPRECATED: Do not rely on __del__ for cleanup.
        
        Python's __del__ is unreliable — it's not guaranteed to be called,
        and can cause issues with garbage collection. Use model_scope()
        context manager or call cleanup() explicitly instead.
        """
        if getattr(self, 'model', None) is not None and not getattr(self, '_cleaned_up', False):
            logger.warning(
                "[MLXManager] Model %s was not cleaned up properly. "
                "Use model_scope() context manager for guaranteed cleanup.",
                self.current_model_key
            )
