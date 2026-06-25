"""
Telegram Notification Utility
Sends updates to your phone about project progress.
"""

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
        Send a message. Safe for both sync and async contexts.

        Args:
            message: Markdown-formatted message (max 4096 chars)

        Returns:
            True if sent successfully
        """
        # Truncate if too long for Telegram
        if len(message) > 4000:
            message = message[:3997] + "..."

        try:
            import asyncio

            # Check if we're already in an async context (running event loop)
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context — use create_task if possible,
                # or run_coroutine_threadsafe for thread safety
                if loop.is_running():
                    # Use asyncio.run_coroutine_threadsafe for safe execution
                    # in a running loop from a different thread
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_async(message), loop
                    )
                    # Wait with timeout to avoid blocking forever
                    future.result(timeout=30)
                    return True
            except RuntimeError:
                # No running loop — safe to use run_until_complete
                pass

            # No running event loop — use standard approach
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._send_async(message))
            finally:
                loop.close()

            logger.debug("[Telegram] Sent: %s...", message[:50])
            return True

        except Exception as e:
            logger.error("[Telegram] Failed to send message: %s", e)
            return False

    async def _send_async(self, message: str) -> None:
        """Async helper for sending messages."""
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

    def send_file(self, file_path: str, caption: str = "") -> bool:
        """Send a file (e.g., final summary)."""
        try:
            import asyncio

            async def _send_file_async():
                with open(file_path, "rb") as f:
                    await self.bot.send_document(
                        chat_id=self.chat_id,
                        document=f,
                        caption=caption[:1024]
                    )

            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        _send_file_async(), loop
                    )
                    future.result(timeout=30)
                    return True
            except RuntimeError:
                pass

            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_send_file_async())
            finally:
                loop.close()
            return True

        except Exception as e:
            logger.error("[Telegram] Failed to send file: %s", e)
            return False
