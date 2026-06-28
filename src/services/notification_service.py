import os
import logging
import asyncio
import aiosmtplib
from email.message import EmailMessage
from typing import List, Dict

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.office365.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", 587))
        self.smtp_user = os.environ.get("SMTP_USER", "alerts@industrial-ai.local")
        self.smtp_pass = os.environ.get("SMTP_PASS", "")

    async def _send_single_email(self, target_email: str, subject: str, content: str):
        """Fires a single email asynchronously with strict timeouts."""
        message = EmailMessage()
        message["From"] = self.smtp_user
        message["To"] = target_email
        message["Subject"] = subject
        message.set_content(content)

        try:
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                start_tls=True,
                username=self.smtp_user,
                password=self.smtp_pass,
                timeout=10.0 # Prevents hung sockets
            )
        except Exception as e:
            logger.error(f"SMTP drop for {target_email}: {e}")

    async def broadcast_alert(self, users: List[Dict[str, str]], asset_type: str, alert_text: str):
        """Concurrently dispatches emails to all targets."""
        if not users:
            logger.warning("Notification aborted: No active team members matching criteria.")
            return

        subject = f"[CRITICAL AI ALERT] Predictive Failure Detected: Asset {asset_type}"
        
        # Gather all network calls and execute them in parallel
        tasks = [
            self._send_single_email(user["email"], subject, alert_text)
            for user in users
        ]
        
        # return_exceptions=True prevents one failed email from crashing the whole broadcast
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Broadcast complete: {len(users)} emails dispatched.")