"""
MLX Model Manager
Handles hot-swapping of models with aggressive memory cleanup.
Critical for 25GB RAM budget on M1 Max.
"""

import gc
import time
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load, generate

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a loaded model."""
    path: str
    max_tokens: int
    temperature: float
    description: str
    ram_estimate_gb: float


class MLXModelManager:
    """
    Manages MLX model lifecycle with aggressive offloading.
    Ensures only ONE model exists in unified memory at any time.
    """

    def __init__(self, gc_sleep: float = 3.0, aggressive_cleanup: bool = True):
        self.gc_sleep = gc_sleep
        self.aggressive_cleanup = aggressive_cleanup

        self.current_model = None
        self.current_tokenizer = None
        self.current_config: Optional[ModelConfig] = None

        self._load_history: list = []
        self._total_swaps: int = 0

        logger.info("[MLXManager] Initialized. Aggressive cleanup: %s", aggressive_cleanup)

    def load_model(self, config: ModelConfig) -> None:
        """
        Load a model, automatically offloading any existing model first.

        Args:
            config: ModelConfig with path and generation parameters
        """
        if self.current_model is not None:
            logger.info("[MLXManager] Active model detected. Offloading first...")
            self.offload_model()

        logger.info(
            "[MLXManager] Loading model: %s (est. %.1f GB)",
            config.path, config.ram_estimate_gb
        )

        try:
            self.current_model, self.current_tokenizer = load(config.path)
            self.current_config = config
            self._load_history.append({
                "model": config.path,
                "action": "loaded",
                "swap_num": self._total_swaps + 1
            })
            self._total_swaps += 1

            logger.info("[MLXManager] Successfully loaded: %s", config.path)

        except Exception as e:
            logger.error("[MLXManager] Failed to load %s: %s", config.path, str(e))
            self._emergency_cleanup()
            raise

    def offload_model(self) -> None:
        """
        Aggressively offloads current model from unified memory.
        This is the CRITICAL function for your 25GB budget.
        """
        if self.current_model is None:
            logger.debug("[MLXManager] No model to offload.")
            return

        model_name = self.current_config.path if self.current_config else "unknown"
        logger.info("[MLXManager] Offloading: %s", model_name)

        # Step 1: Delete all references
        try:
            del self.current_model
            del self.current_tokenizer
        except Exception as e:
            logger.warning("[MLXManager] Error deleting references: %s", e)

        self.current_model = None
        self.current_tokenizer = None
        self.current_config = None

        # Step 2: Force Python garbage collection (multiple passes)
        for _ in range(3):
            gc.collect()

        # Step 3: MLX synchronize to ensure GPU/Metal operations complete
        try:
            mx.synchronize()
        except Exception as e:
            logger.warning("[MLXManager] mx.synchronize() error: %s", e)

        # Step 4: Aggressive Metal cache clear (if available in MLX version)
        if self.aggressive_cleanup:
            try:
                # Try newer API first
                if hasattr(mx.metal, 'clear_cache'):
                    mx.metal.clear_cache()
                    logger.info("[MLXManager] Metal cache cleared via clear_cache()")
                # Fallback for older MLX versions
                elif hasattr(mx, 'clear_cache'):
                    mx.clear_cache()
                    logger.info("[MLXManager] Cache cleared via mx.clear_cache()")
            except Exception as e:
                logger.debug("[MLXManager] Cache clear not available: %s", e)

        # Step 5: Sleep to let memory pressure settle
        logger.info("[MLXManager] Sleeping %.1fs for memory settlement...", self.gc_sleep)
        time.sleep(self.gc_sleep)

        self._load_history.append({
            "model": model_name,
            "action": "offloaded",
            "swap_num": self._total_swaps
        })

        logger.info("[MLXManager] Offload complete.")

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, 
                 verbose: bool = False) -> str:
        """
        Generate text using the currently loaded model.

        Args:
            prompt: The input prompt
            max_tokens: Override default max tokens
            temperature: Override default temperature
            verbose: Print generation progress

        Returns:
            Generated text string
        """
        if self.current_model is None or self.current_config is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        tokens = max_tokens or self.current_config.max_tokens
        temp = temperature if temperature is not None else self.current_config.temperature

        logger.info(
            "[MLXManager] Generating with %s (max_tokens=%d, temp=%.2f)",
            self.current_config.path, tokens, temp
        )

        try:
            result = generate(
                model=self.current_model,
                tokenizer=self.current_tokenizer,
                prompt=prompt,
                max_tokens=tokens,
                temp=temp,
                verbose=verbose
            )

            logger.info(
                "[MLXManager] Generation complete. Output length: %d chars",
                len(result)
            )
            return result

        except Exception as e:
            logger.error("[MLXManager] Generation failed: %s", str(e))
            raise

    def get_status(self) -> Dict[str, Any]:
        """Return current manager status."""
        return {
            "loaded_model": self.current_config.path if self.current_config else None,
            "total_swaps": self._total_swaps,
            "history": self._load_history[-5:]  # Last 5 operations
        }

    def _emergency_cleanup(self) -> None:
        """Emergency memory cleanup on failure."""
        logger.critical("[MLXManager] EMERGENCY CLEANUP initiated!")
        self.current_model = None
        self.current_tokenizer = None
        self.current_config = None
        gc.collect()
        try:
            mx.synchronize()
            if hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()
        except:
            pass
        time.sleep(5)
        logger.critical("[MLXManager] Emergency cleanup complete.")

    def __del__(self):
        """Ensure cleanup on destruction."""
        if self.current_model is not None:
            self.offload_model()
