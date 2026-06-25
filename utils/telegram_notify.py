#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FORGE  — FILE: utils/telegram_notify.py                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Forge (formerly Brain Loader v1)
# REPO:       https://github.com/Ehsas317/forge
# WHAT:       It hammers out a project sequentially, task by task, with
#             iterative review. A forge is slow, hot, and precise.
#
# THIS FILE:
#   Telegram Notifier — sends real-time build progress updates via Telegram.
#   Get your bot token from @BotFather and configure in config.yaml.
#
# HOW TO USE FORGE:
#   1. Install:    pip install -r requirements.txt
#   2. Configure:  Edit config.yaml with your API tokens
#   3. Run:        python main.py "Your project description"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Forge — Telegram Notifier

Sends real-time build progress updates via Telegram.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger("forge.telegram")


class TelegramNotifier:
    """
    Forge Telegram Notifier

    Sends build progress updates to a Telegram chat.
    Requires a bot token (from @BotFather) and chat ID.

    Usage:
        notifier = TelegramNotifier(bot_token="...", chat_id="...")
        notifier.send("🔥 Forge starting new build...")
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._enabled = bool(bot_token and chat_id)

        if not self._enabled:
            logger.warning("Telegram notifier disabled — missing token or chat ID")

    def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to the configured Telegram chat."""
        if not self._enabled:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
            }

            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()

            logger.debug("Telegram message sent: %s", message[:50])
            return True

        except httpx.HTTPStatusError as e:
            logger.error("Telegram API error: %s", e.response.text)
            return False
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def send_file(self, file_path: str, caption: str = "") -> bool:
        """Send a file to Telegram."""
        if not self._enabled:
            return False

        try:
            url = f"{self.base_url}/sendDocument"

            with open(file_path, "rb") as f:
                files = {"document": f}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                }
                response = httpx.post(url, data=data, files=files, timeout=30.0)
                response.raise_for_status()

            logger.debug("Telegram file sent: %s", file_path)
            return True

        except Exception as e:
            logger.error("Failed to send Telegram file: %s", e)
            return False

    def __repr__(self):
        status = "enabled" if self._enabled else "disabled"
        return f"<TelegramNotifier {status}>"
