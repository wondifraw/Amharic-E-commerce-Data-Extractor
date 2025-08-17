"""Professional Telegram scraper for e-commerce data collection."""

import asyncio
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from telethon import TelegramClient
from telethon.errors import FloodWaitError, ChannelPrivateError
from dotenv import load_dotenv
# Config handled externally


@dataclass
class ScrapingConfig:
    """Configuration for Telegram scraping."""
    api_id: str
    api_hash: str
    session_name: str = 'scraping_session'
    max_messages: int = 10000
    media_download: bool = True
    rate_limit_delay: float = 1.0
    cache_file: str = 'data/cache/scraped_message_ids.json'
    output_file: str = 'data/raw/telegram_messages.csv'
    media_dir: str = 'data/raw/media'


class TelegramScraper:
    """Professional Telegram scraper with caching and error handling."""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.client = TelegramClient(config.session_name, config.api_id, config.api_hash)
        self.cache: Dict[str, Set[int]] = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def load_cache(self) -> None:
        """Load cached message IDs to avoid duplicates."""
        cache_path = Path(self.config.cache_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = {k: set(v) for k, v in data.items()}
            except (json.JSONDecodeError, FileNotFoundError):
                self.cache = {}
        else:
            self.cache = {}
            
    def save_cache(self) -> None:
        """Save cached message IDs."""
        cache_path = Path(self.config.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert sets to lists for JSON serialization
        cache_data = {k: list(v) for k, v in self.cache.items()}
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
            
    async def scrape_channel(self, channel_username: str, writer: csv.writer) -> int:
        """Scrape messages from a single channel."""
        try:
            entity = await self.client.get_entity(channel_username)
            channel_title = getattr(entity, 'title', channel_username)
            
            if channel_username not in self.cache:
                self.cache[channel_username] = set()
                
            message_count = 0
            media_dir = Path(self.config.media_dir)
            media_dir.mkdir(parents=True, exist_ok=True)
            
            async for message in self.client.iter_messages(entity, limit=self.config.max_messages):
                if message.id in self.cache[channel_username]:
                    continue
                    
                media_path = None
                if self.config.media_download and message.media:
                    try:
                        filename = f"{channel_username}_{message.id}.jpg"
                        media_path = media_dir / filename
                        await self.client.download_media(message.media, str(media_path))
                    except Exception as e:
                        self.logger.warning(f"Failed to download media for message {message.id}: {e}")
                        
                writer.writerow([
                    channel_title,
                    channel_username,
                    message.id,
                    message.message or '',
                    message.date.isoformat() if message.date else '',
                    str(media_path) if media_path else ''
                ])
                
                self.cache[channel_username].add(message.id)
                message_count += 1
                
                if message_count % 100 == 0:
                    self.save_cache()
                    
            self.logger.info(f"Scraped {message_count} new messages from {channel_username}")
            return message_count
            
        except ChannelPrivateError:
            self.logger.error(f"Channel {channel_username} is private or doesn't exist")
            return 0
        except FloodWaitError as e:
            self.logger.warning(f"Rate limited. Waiting {e.seconds} seconds...")
            await asyncio.sleep(e.seconds)
            return 0
        except Exception as e:
            self.logger.error(f"Error scraping {channel_username}: {e}")
            return 0
            
    async def scrape_channels(self, channels: List[str]) -> Dict[str, int]:
        """Scrape multiple channels concurrently."""
        await self.client.start()
        self.load_cache()
        
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        with open(output_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write header if file is empty
            if output_path.stat().st_size == 0:
                writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
                
            for channel in channels:
                count = await self.scrape_channel(channel, writer)
                results[channel] = count
                await asyncio.sleep(self.config.rate_limit_delay)
                
        self.save_cache()
        return results
        
    async def close(self):
        """Close the Telegram client."""
        await self.client.disconnect()


def create_scraper_from_env() -> TelegramScraper:
    """Create scraper instance from environment variables."""
    load_dotenv()
    
    config = ScrapingConfig(
        api_id=os.getenv('TG_API_ID'),
        api_hash=os.getenv('TG_API_HASH')
    )
    
    if not config.api_id or not config.api_hash:
        raise ValueError("TG_API_ID and TG_API_HASH must be set in environment variables")
        
    return TelegramScraper(config)


async def main():
    """Main scraping function."""
    scraper = create_scraper_from_env()
    
    channels = [
        '@classybrands',
        '@Shageronlinestore',
        '@ZemenExpress',
        '@sinayelj',
        '@modernshoppingcenter'
    ]
    
    try:
        results = await scraper.scrape_channels(channels)
        print("Scraping completed:")
        for channel, count in results.items():
            print(f"  {channel}: {count} messages")
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())