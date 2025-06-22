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

class TelegramScraper:
    """
    Scrapes messages, images, and documents from specified Telegram channels.
    """
    def __init__(self, api_id: str, api_hash: str, session_name: str = 'anon'):
        """
        Initializes the scraper with Telegram API credentials.

        Args:
            api_id (str): Your Telegram API ID.
            api_hash (str): Your Telegram API hash.
 
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(session_name, api_id, api_hash)

    def get_channel_list(self) -> List[str]:
        """
        Returns a predefined list of Telegram channel usernames.

        Returns:
            List[str]: A list of channel usernames.
        """
        return [
            '@Leyueqa',
            '@sinayelj',
            '@Shewabrand',
            '@helloomarketethiopia',
            '@modernshoppingcenter',
        ]

    async def fetch_messages(self, channel: str, limit: int = 1000) -> List[Dict]:
        """
        Fetches messages from a given Telegram channel.

        Args:
            channel (str): The username or link of the Telegram channel.
            limit (int): The maximum number of messages to fetch.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a message.
        """
        messages = []
        try:
            print(f"Connecting to Telegram...")
            async with self.client:
                print(f"Fetching channel entity for {channel}...")
                entity = await self.client.get_entity(channel)
                channel_title = getattr(entity, 'title', None)
                channel_username = getattr(entity, 'username', channel.strip('@'))
                print(f"Fetching messages from {channel}...")
                async for msg in self.client.iter_messages(channel, limit=limit):
                    media_path = None
                    if msg.media:
                        media_dir = os.path.join('data', 'raw', channel.strip('@'), 'media')
                        os.makedirs(media_dir, exist_ok=True)
                        try:
                            media_path = await msg.download_media(file=media_dir)
                        except Exception as e:
                            print(f"Could not download media for message {msg.id}: {e}")

                    msg_dict = {
                        'Channel Title': channel_title,
                        'Channel Username': channel_username,
                        'ID': msg.id,
                        'Message': msg.text,
                        'Date': str(msg.date),
                        'Media Path': media_path
                    }
                    messages.append(msg_dict)
            print(f"Successfully fetched {len(messages)} messages from {channel}.")
        except Exception as e:
            print(f"[ERROR] Failed to fetch messages from {channel}: {e}")
        return messages

    def save_data(self, data: List[Dict], out_path: str):
        """
        Saves data to a JSON file.

        Args:
            data (List[Dict]): The data to save.
            out_path (str): The path to the output JSON file.
        """
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Data successfully saved to {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save data to {out_path}: {e}")

    def save_data_csv(self, data: List[Dict], out_path: str):
        """
        Saves data to a CSV file.

        Args:
            data (List[Dict]): The data to save.
            out_path (str): The path to the output CSV file.
        """
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df = pd.DataFrame(data)
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"Data successfully saved to {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save data to {out_path}: {e}") 