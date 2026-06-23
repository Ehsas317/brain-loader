"""
Telegram Notification Utility
Sends updates to your phone about project progress.
"""

import asyncio
import logging
from typing import Optional

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Simple Telegram bot wrapper for project notifications.

    Setup:
    1. Message @BotFather on Telegram, create a bot, get token
    2. Message @userinfobot to get your chat ID
    3. Put both in config.yaml
    """

    def __init__(self, token: str, chat_id: str):
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot"
            )

        self.token = token
        self.chat_id = str(chat_id)
        self.bot = Bot(token=token)

        logger.info("[Telegram] Notifier initialized for chat %s", chat_id)

    def send(self, message: str) -> bool:
        """
        Send a message. Handles asyncio internally.

        Args:
            message: Markdown-formatted message (max 4096 chars)

        Returns:
            True if sent successfully
        """
        # Truncate if too long for Telegram
        if len(message) > 4000:
            message = message[:3997] + "..."

        try:
            # Create new event loop if needed (for thread safety)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode="Markdown",
                    disable_web_page_preview=True
                )
            )

            logger.debug("[Telegram] Sent: %s...", message[:50])
            return True

        except Exception as e:
            logger.error("[Telegram] Failed to send message: %s", e)
            return False

    def send_file(self, file_path: str, caption: str = "") -> bool:
        """Send a file (e.g., final summary)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            with open(file_path, "rb") as f:
                loop.run_until_complete(
                    self.bot.send_document(
                        chat_id=self.chat_id,
                        document=f,
                        caption=caption[:1024]
                    )
                )
            return True
        except Exception as e:
            logger.error("[Telegram] Failed to send file: %s", e)
            return False
