"""
Alert system for the trading platform.
Sends notifications (e.g., via email) when key thresholds are breached.
"""

import smtplib
from email.mime.text import MIMEText
from core.config import config
from core.logger import log_manager

logger = log_manager.get_logger(__name__)

class AlertManager:
    def __init__(self):
        self.smtp_server = config.get("monitoring.smtp_server")
        self.smtp_port = config.get("monitoring.smtp_port")
        self.username = config.get("monitoring.email_username")
        self.password = config.get("monitoring.email_password")
        self.from_email = config.get("monitoring.from_email")
        self.to_emails = config.get("monitoring.to_emails", [])

    def send_alert(self, subject: str, message: str):
        """
        Send an email alert.
        
        Args:
            subject: Subject of the email.
            message: Body of the email.
        """
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())
            logger.info(f"Alert sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
