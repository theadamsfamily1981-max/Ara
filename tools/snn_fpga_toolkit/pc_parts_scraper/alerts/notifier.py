"""
Alert System for PC Parts Scraper
Sends notifications when rare parts are found
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.models import PCPart, Alert
from utils.helpers import format_price, truncate_text, load_config

logger = logging.getLogger(__name__)


class AlertNotifier:
    """
    Manages sending alerts through various channels
    """

    def __init__(self, config: Dict[str, Any] = None, db_manager=None):
        self.config = config or load_config()
        self.db = db_manager
        self.alert_config = self.config.get('alerts', {})

    async def check_and_alert(self, parts: List[PCPart]):
        """
        Check parts against alert criteria and send notifications
        """
        if not self.alert_config.get('enabled', False):
            return

        for part in parts:
            # Check if part matches any alert criteria
            alerts_to_send = self._check_alert_criteria(part)

            for alert_type in alerts_to_send:
                # Check if we've already sent this alert
                if self.db and self.db.has_alert_been_sent(part.id, alert_type):
                    continue

                # Send the alert
                await self._send_alert(part, alert_type)

    def _check_alert_criteria(self, part: PCPart) -> List[str]:
        """Check if a part matches any alert criteria"""
        alerts = []

        # Check for instant alert keywords
        instant_keywords = self.alert_config.get('instant_alert_keywords', [])
        text = f"{part.title} {part.description or ''}".lower()

        for keyword in instant_keywords:
            if keyword.lower() in text:
                alerts.append('instant_keyword')
                break

        # Check for rare item
        if part.is_rare and part.rarity_score >= 70:
            alerts.append('rare_item')

        # Check price thresholds
        price_alerts = self.alert_config.get('price_alerts', {})

        if part.category == 'gpu' and part.price:
            if part.price <= price_alerts.get('vintage_gpu_max', 50):
                alerts.append('price_alert')

        if part.category == 'cpu' and part.price:
            if part.price <= price_alerts.get('vintage_cpu_max', 30):
                alerts.append('price_alert')

        if part.category == 'motherboard' and part.price:
            if part.price <= price_alerts.get('rare_motherboard_max', 100):
                alerts.append('price_alert')

        return alerts

    async def _send_alert(self, part: PCPart, alert_type: str):
        """Send alert through all enabled channels"""
        message = self._format_message(part, alert_type)

        channels = self.alert_config.get('channels', {})

        # Discord
        if channels.get('discord', {}).get('enabled'):
            try:
                await self._send_discord(message, channels['discord'])
                self._record_alert(part.id, alert_type, 'discord', message)
            except Exception as e:
                logger.error(f"Discord alert failed: {e}")

        # Slack
        if channels.get('slack', {}).get('enabled'):
            try:
                await self._send_slack(message, channels['slack'])
                self._record_alert(part.id, alert_type, 'slack', message)
            except Exception as e:
                logger.error(f"Slack alert failed: {e}")

        # Email
        if channels.get('email', {}).get('enabled'):
            try:
                await self._send_email(message, channels['email'])
                self._record_alert(part.id, alert_type, 'email', message)
            except Exception as e:
                logger.error(f"Email alert failed: {e}")

        # Console (always enabled for debugging)
        logger.info(f"ALERT [{alert_type}]: {part.title} - {format_price(part.price)}")

    def _format_message(self, part: PCPart, alert_type: str) -> Dict[str, str]:
        """Format alert message for different channels"""
        # Create message content
        if alert_type == 'instant_keyword':
            emoji = "🚨"
            title = "RARE ITEM FOUND!"
        elif alert_type == 'rare_item':
            emoji = "⭐"
            title = "Rare PC Part Discovered"
        elif alert_type == 'price_alert':
            emoji = "💰"
            title = "Great Price Alert"
        else:
            emoji = "📦"
            title = "New Item Alert"

        # Build message
        plain_text = f"""
{emoji} {title}

{part.title}

Price: {format_price(part.price) if part.price else 'Not listed'}
Category: {part.category or 'Unknown'}
Condition: {part.condition or 'Unknown'}
Source: {part.source_name}
Rarity Score: {part.rarity_score:.1f}/100

{part.source_url}
"""

        # Markdown version
        markdown = f"""
## {emoji} {title}

**{truncate_text(part.title, 100)}**

- **Price:** {format_price(part.price) if part.price else 'Not listed'}
- **Category:** {part.category or 'Unknown'}
- **Condition:** {part.condition or 'Unknown'}
- **Source:** {part.source_name}
- **Rarity Score:** {part.rarity_score:.1f}/100

[View Item]({part.source_url})
"""

        return {
            'plain': plain_text.strip(),
            'markdown': markdown.strip(),
            'title': title,
            'url': part.source_url,
            'thumbnail': part.thumbnail_url,
        }

    async def _send_discord(self, message: Dict[str, str], config: Dict[str, Any]):
        """Send alert to Discord webhook"""
        try:
            from discord_webhook import DiscordWebhook, DiscordEmbed

            webhook = DiscordWebhook(url=config['webhook_url'])

            embed = DiscordEmbed(
                title=message['title'],
                description=message['markdown'],
                color='03b2f8'
            )

            if message.get('thumbnail'):
                embed.set_thumbnail(url=message['thumbnail'])

            embed.set_timestamp()
            webhook.add_embed(embed)

            response = webhook.execute()
            logger.info(f"Discord alert sent: {response.status_code}")

        except ImportError:
            logger.warning("discord-webhook not installed")

    async def _send_slack(self, message: Dict[str, str], config: Dict[str, Any]):
        """Send alert to Slack webhook"""
        try:
            from slack_sdk.webhook import WebhookClient

            webhook = WebhookClient(config['webhook_url'])

            response = webhook.send(
                text=message['title'],
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": message['markdown']
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "View Item"
                                },
                                "url": message['url']
                            }
                        ]
                    }
                ]
            )

            logger.info(f"Slack alert sent: {response.status_code}")

        except ImportError:
            logger.warning("slack-sdk not installed")

    async def _send_email(self, message: Dict[str, str], config: Dict[str, Any]):
        """Send alert via email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"PC Parts Alert: {message['title']}"
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['recipients'])

            # Add plain text and HTML parts
            part1 = MIMEText(message['plain'], 'plain')
            msg.attach(part1)

            # Send email
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.sendmail(
                    config['username'],
                    config['recipients'],
                    msg.as_string()
                )

            logger.info("Email alert sent")

        except Exception as e:
            logger.error(f"Email failed: {e}")

    def _record_alert(self, part_id: int, alert_type: str, channel: str, message: Dict):
        """Record sent alert in database"""
        if self.db:
            alert = Alert(
                part_id=part_id,
                alert_type=alert_type,
                channel=channel,
                sent_at=datetime.utcnow(),
                message=message['plain'][:500]
            )
            self.db.save_alert(alert)


class PriceDropNotifier:
    """
    Specialized notifier for price drop alerts
    """

    def __init__(self, config=None, db_manager=None):
        self.config = config or load_config()
        self.db = db_manager
        self.base_notifier = AlertNotifier(config, db_manager)

    async def check_price_drops(self, min_drop_percent: float = 10):
        """Check for and alert on price drops"""
        if not self.db:
            return

        drops = self.db.get_price_drops(min_drop_percent)

        for drop in drops:
            part = drop['part']

            # Create custom message for price drop
            message = f"""
💰 PRICE DROP ALERT!

{part.title}

Previous: {format_price(drop['previous_price'])}
Current: {format_price(drop['current_price'])}
Drop: {drop['drop_percent']:.1f}%

Source: {part.source_name}
{part.source_url}
"""

            logger.info(message)

            # Could send through channels here
