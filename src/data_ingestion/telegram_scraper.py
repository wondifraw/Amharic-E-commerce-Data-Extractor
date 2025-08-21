"""
Telegram Data Scraper for Ethiopian E-commerce Channels
"""
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from loguru import logger
import yaml
from dotenv import load_dotenv
load_dotenv()
class TelegramScraper:
    """Scrapes messages from Ethiopian Telegram e-commerce channels"""
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        self.config = self._load_config(config_path)
        self.client = None
        self.scraped_data = []
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    async def initialize_client(self):
        """Initialize Telegram client"""
        api_id = os.getenv('TELEGRAM_API_ID') or self.config['telegram']['api_id']
        api_hash = os.getenv('TELEGRAM_API_HASH') or self.config['telegram']['api_hash']
        phone = os.getenv('TELEGRAM_PHONE_NUMBER') or self.config['telegram']['phone_number']
        
        if not all([api_id, api_hash, phone]) or api_id == "YOUR_API_ID" or phone == "YOUR_PHONE_NUMBER":
            raise ValueError("Missing Telegram credentials. Set TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE_NUMBER in .env file")
        
        if not str(phone).startswith('+'):
            phone = '+' + str(phone)
        
        self.client = TelegramClient('session_name', int(api_id), api_hash)
        await self.client.start(phone=phone)
        logger.info("Telegram client initialized successfully")
    
    async def scrape_channel(self, channel_username: str, limit: int = 2000) -> List[Dict]:
        """Scrape messages from a specific channel"""
        messages = []
        
        try:
            entity = await self.client.get_entity(channel_username)
            logger.info(f"Scraping channel: {channel_username}")
            
            async for message in self.client.iter_messages(entity, limit=limit):
                if message.text:
                    message_data = {
                        'id': message.id,
                        'channel': channel_username,
                        'text': message.text,
                        'date': message.date.isoformat() if message.date else None,
                        'views': getattr(message, 'views', 0),
                        'forwards': getattr(message, 'forwards', 0),
                        'replies': getattr(message.replies, 'replies', 0) if message.replies else 0,
                        'sender_id': message.sender_id,
                        'has_media': bool(message.media),
                        'media_type': self._get_media_type(message.media),
                        'message_link': f"https://t.me/{channel_username.replace('@', '')}/{message.id}"
                    }
                    messages.append(message_data)
            
            logger.info(f"Scraped {len(messages)} messages from {channel_username}")
            
        except Exception as e:
            logger.error(f"Error scraping channel {channel_username}: {str(e)}")
        
        return messages
    
    def _get_media_type(self, media) -> Optional[str]:
        """Determine media type"""
        if isinstance(media, MessageMediaPhoto):
            return "photo"
        elif isinstance(media, MessageMediaDocument):
            return "document"
        return None

    async def scrape_all_channels(self, limit_per_channel: int = 2000) -> pd.DataFrame:
        """Scrape all configured channels"""
        all_messages = []
        
        for channel in self.config['channels']:
            messages = await self.scrape_channel(channel, limit_per_channel)
            all_messages.extend(messages)
        
        df = pd.DataFrame(all_messages)
        logger.info(f"Total messages scraped: {len(df)}")
        return df
    
    async def save_raw_data(self, df: pd.DataFrame, filename: str = None):
        """Save scraped data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"telegram_data_{timestamp}.csv"
        
        output_path = os.path.join(self.config['data']['raw_data_path'], filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Data saved to {output_path}")
        return output_path
    
    async def close(self):
        """Close the Telegram client"""
        if self.client:
            await self.client.disconnect()


async def main():
    """Main function to run the scraper"""
    try:
        scraper = TelegramScraper()
        await scraper.initialize_client()
        df = await scraper.scrape_all_channels(limit_per_channel=2000)
        await scraper.save_raw_data(df)
        await scraper.close()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Use demo_scraper.py for testing without Telegram API")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        logger.info("Use demo_scraper.py for testing without Telegram API")
if __name__ == "__main__":
    asyncio.run(main())