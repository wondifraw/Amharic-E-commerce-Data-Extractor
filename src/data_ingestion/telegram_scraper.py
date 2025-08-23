"""Telegram data scraper for Amharic e-commerce channels."""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
import logging

from config.config import telegram_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramScraper:
    """Scraper for collecting messages from Telegram channels."""
    
    def __init__(self):
        self.client = None
        self.config = telegram_config
        
    async def initialize_client(self) -> bool:
        """Initialize Telegram client."""
        try:
            self.client = TelegramClient('session', self.config.api_id, self.config.api_hash)
            await self.client.start()
            
            if not await self.client.is_user_authorized():
                await self.client.send_code_request(self.config.phone)
                code = input('Enter the code: ')
                await self.client.sign_in(self.config.phone, code)
                
            logger.info("Telegram client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            return False
    
    async def scrape_channel(self, channel: str, limit: int = 100) -> List[Dict]:
        """Scrape messages from a single channel."""
        messages = []
        
        try:
            entity = await self.client.get_entity(channel)
            
            async for message in self.client.iter_messages(entity, limit=limit):
                if message.text:
                    msg_data = {
                        'channel': channel,
                        'message_id': message.id,
                        'text': message.text,
                        'date': message.date,
                        'views': getattr(message, 'views', 0),
                        'forwards': getattr(message, 'forwards', 0),
                        'sender_id': message.sender_id,
                        'media_type': 'text' if not message.media else str(type(message.media).__name__)
                    }
                    messages.append(msg_data)
                    
            logger.info(f"Scraped {len(messages)} messages from {channel}")
            
        except Exception as e:
            logger.error(f"Error scraping {channel}: {e}")
            
        return messages
    
    async def scrape_all_channels(self, limit_per_channel: int = 100) -> pd.DataFrame:
        """Scrape messages from all configured channels."""
        all_messages = []
        
        if not await self.initialize_client():
            return pd.DataFrame()
        
        for channel in self.config.channels:
            messages = await self.scrape_channel(channel, limit_per_channel)
            all_messages.extend(messages)
            await asyncio.sleep(1)  # Rate limiting
        
        await self.client.disconnect()
        
        df = pd.DataFrame(all_messages)
        logger.info(f"Total messages scraped: {len(df)}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str) -> None:
        """Save scraped data to file."""
        try:
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

async def main():
    """Main function to run the scraper."""
    scraper = TelegramScraper()
    df = await scraper.scrape_all_channels(limit_per_channel=200)
    
    if not df.empty:
        scraper.save_data(df, 'data/raw/telegram_messages.csv')
    
if __name__ == "__main__":
    asyncio.run(main())