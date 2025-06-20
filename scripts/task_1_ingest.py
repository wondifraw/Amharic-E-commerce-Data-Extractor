"""
task_1_ingest.py
Main script to run the data ingestion and preprocessing pipeline.

This script orchestrates the process of fetching data from Telegram channels,
preprocessing it, and saving both raw and processed versions.
"""
import asyncio
import os
import argparse
from src.ingestion.telegram_scraper import TelegramScraper
from src.preprocessing.amharic_text import preprocess_messages

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME = "telegram_session"

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
MESSAGE_LIMIT_PER_CHANNEL = 200

async def main(api_id: str, api_hash: str):
    """
    Main function to execute the data ingestion and preprocessing pipeline.
    """
    print("--- Starting Data Ingestion and Preprocessing Pipeline ---")
    
    scraper = TelegramScraper(api_id, api_hash, SESSION_NAME)
    channels = scraper.get_channel_list()
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    for channel in channels:
        print(f"\n--- Processing Channel: {channel} ---")
        
        raw_messages = await scraper.fetch_messages(channel, limit=MESSAGE_LIMIT_PER_CHANNEL)
        
        if not raw_messages:
            print(f"No messages fetched for {channel}. Skipping.")
            continue
            
        raw_path = os.path.join(RAW_DATA_DIR, f"{channel.strip('@')}_raw.json")
        scraper.save_data(raw_messages, raw_path)
        
        print(f"Preprocessing {len(raw_messages)} messages for {channel}...")
        processed_messages = preprocess_messages(raw_messages)
        
        processed_path = os.path.join(PROCESSED_DATA_DIR, f"{channel.strip('@')}_processed.json")
        scraper.save_data(processed_messages, processed_path)
        
    print("\n--- Pipeline execution finished successfully! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Telegram data ingestion pipeline.")
    parser.add_argument("--api_id", type=str, default=API_ID, help="Your Telegram API ID.")
    parser.add_argument("--api_hash", type=str, default=API_HASH, help="Your Telegram API Hash.")
    args = parser.parse_args()

    if not args.api_id or not args.api_hash:
        print("[ERROR] Telegram API ID and Hash must be provided.")
        print("Example: python scripts/task_1_ingest.py --api_id YOUR_ID --api_hash YOUR_HASH")
    else:
        asyncio.run(main(args.api_id, args.api_hash)) 