"""
telegram_scraper.py
Module for scraping messages from Telegram channels.
"""
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import os
import json
from typing import List, Dict
import pandas as pd
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("telegram_scraper")

class TelegramScraper:
    """
    Scrapes messages, images, and documents from specified Telegram channels.
    Optionally accepts a custom channel list.
    """
    def __init__(self, api_id: str, api_hash: str, session_name: str = 'anon', channel_list: list = None):
        """
        Initializes the scraper with Telegram API credentials and optional channel list.
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(session_name, api_id, api_hash)
        self._channel_list = channel_list

    def get_channel_list(self) -> List[str]:
        """
        Returns a predefined or user-supplied list of Telegram channel usernames.
        """
        if self._channel_list:
            logger.info(f"Using custom channel list: {self._channel_list}")
            return self._channel_list
        default_channels = [
            '@Leyueqa',
            '@sinayelj',
            '@Shewabrand',
            '@helloomarketethiopia',
            '@modernshoppingcenter',
        ]
        logger.info(f"Using default channel list: {default_channels}")
        return default_channels

    async def fetch_messages(self, channel: str, limit: int = 1000) -> List[Dict]:
        """
        Fetches messages from a given Telegram channel, logs metrics and errors.
        Returns a list of message dicts.
        """
        messages = []
        media_count = 0
        error_count = 0
        try:
            logger.info(f"Connecting to Telegram for channel: {channel}")
            async with self.client:
                logger.info(f"Fetching channel entity for {channel}...")
                entity = await self.client.get_entity(channel)
                channel_title = getattr(entity, 'title', None)
                channel_username = getattr(entity, 'username', channel.strip('@'))
                logger.info(f"Fetching messages from {channel} (limit={limit})...")
                async for msg in self.client.iter_messages(channel, limit=limit):
                    media_path = None
                    if msg.media:
                        media_dir = os.path.join('data', 'raw', channel.strip('@'), 'media')
                        os.makedirs(media_dir, exist_ok=True)
                        try:
                            media_path = await msg.download_media(file=media_dir)
                            media_count += 1
                        except Exception as e:
                            logger.warning(f"Could not download media for message {msg.id}: {e}")
                            error_count += 1
                    msg_dict = {
                        'Channel Title': channel_title,
                        'Channel Username': channel_username,
                        'ID': msg.id,
                        'Message': msg.text,
                        'Date': str(msg.date),
                        'Media Path': media_path
                    }
                    # Validate message dict
                    if not msg_dict['Message']:
                        logger.debug(f"Skipping empty message (ID: {msg.id})")
                        continue
                    messages.append(msg_dict)
            logger.info(f"Successfully fetched {len(messages)} messages from {channel}. Media files: {media_count}, Errors: {error_count}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to fetch messages from {channel}: {e}", exc_info=True)
        return messages

    def save_data(self, data: List[Dict], out_path: str):
        """
        Saves data to a JSON file. Logs errors and validates output.
        """
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # Validate data before saving
            if not isinstance(data, list):
                raise ValueError("Data to save must be a list of dicts.")
            if data and not isinstance(data[0], dict):
                raise ValueError("Each item in data must be a dict.")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Data successfully saved to {out_path} (records: {len(data)})")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save data to {out_path}: {e}", exc_info=True)

    def save_data_csv(self, data: List[Dict], out_path: str):
        """
        Saves data to a CSV file. Logs errors and validates output.
        """
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if not isinstance(data, list):
                raise ValueError("Data to save must be a list of dicts.")
            if data and not isinstance(data[0], dict):
                raise ValueError("Each item in data must be a dict.")
            df = pd.DataFrame(data)
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            logger.info(f"Data successfully saved to {out_path} (records: {len(data)})")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save data to {out_path}: {e}", exc_info=True) 