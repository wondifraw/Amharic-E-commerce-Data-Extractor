#!/usr/bin/env python3
"""
Automated Telegram Scraper with Scheduling
Production-ready script for continuous data collection
"""

import os
import sys
import asyncio
import schedule
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.env_config import config

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from telegram_scraper import TelegramScraper

class AutomatedScraper:
    def __init__(self):
        # Configuration
        self.channels = [
            '@classybrands',
            '@Shageronlinestore', 
            '@ZemenExpress',
            '@sinayelj',
            '@modernshoppingcenter'
        ]
        
        self.output_file = '../../data/telegram_data.csv'
        self.log_file = '../../logs/scraper.log'
        self.incremental_limit = 100
        
        # Setup logging
        os.makedirs('../../logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Initialize scraper
        self.api_id = os.getenv('TG_API_ID')
        self.api_hash = os.getenv('TG_API_HASH')
        self.scraper = None
        
    async def incremental_scrape(self):
        """Scrape only new messages since last run"""
        try:
            logging.info("Starting incremental scrape...")
            
            if not self.scraper:
                self.scraper = TelegramScraper(self.api_id, self.api_hash)
            
            last_timestamp = self.get_last_timestamp()
            
            new_data = await self.scraper.scrape_channels(
                channels=self.channels,
                limit=self.incremental_limit,
                since_date=last_timestamp
            )
            
            if new_data:
                self.append_data(new_data)
                logging.info(f"Collected {len(new_data)} new messages")
            else:
                logging.info("No new messages found")
                
        except Exception as e:
            logging.error(f"Scraping failed: {str(e)}")
            
    def get_last_timestamp(self):
        """Get timestamp of last collected message"""
        try:
            if os.path.exists(self.output_file):
                df = pd.read_csv(self.output_file)
                if not df.empty:
                    return pd.to_datetime(df['Date']).max()
        except:
            pass
        return datetime.now() - timedelta(days=1)
    
    def append_data(self, new_data):
        """Append new data to existing file"""
        new_df = pd.DataFrame(new_data)
        
        if os.path.exists(self.output_file):
            existing_df = pd.read_csv(self.output_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['ID'], keep='last')
        else:
            combined_df = new_df
            
        combined_df.to_csv(self.output_file, index=False, encoding='utf-8')
        logging.info(f"Data saved to {self.output_file}")

    def run_sync(self):
        """Synchronous wrapper for async scraper"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.incremental_scrape())
            loop.close()
        except Exception as e:
            logging.error(f"Scheduler error: {str(e)}")

def main():
    """Main function to run the automated scraper"""
    scraper = AutomatedScraper()
    
    # Schedule daily scraping at 9 AM
    schedule.every().day.at("09:00").do(scraper.run_sync)
    
    logging.info("Automated scraper started - Daily at 09:00")
    logging.info(f"Monitoring {len(scraper.channels)} channels")
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()