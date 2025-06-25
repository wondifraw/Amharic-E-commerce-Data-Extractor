"""
task_1_ingest.py
Main script to run the data ingestion and preprocessing pipeline.

This script orchestrates the process of fetching data from Telegram channels,
preprocessing it, and saving both raw and processed versions with the simplified structure.
"""
import asyncio
import os
import argparse
import logging
import pandas as pd
from src.telegram_data_ingestion.telegram_scraper import TelegramScraper
from src.amharic_text_processing.amharic_text import preprocess_messages, load_and_preprocess_csv

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
# SESSION_NAME = "telegram_session"

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
MESSAGE_LIMIT_PER_CHANNEL = 200

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("task_1_ingest")

async def main(api_id: str, api_hash: str, channel_list=None):
    """
    Main function to execute the data ingestion and preprocessing pipeline.
    Works with the simplified data structure: Channel Title, Channel Username, ID, Message, Date, Media Path
    """
    logger.info("--- Starting Data Ingestion and Preprocessing Pipeline ---")
    
    scraper = TelegramScraper(api_id, api_hash)
    channels = channel_list if channel_list else scraper.get_channel_list()
    logger.info(f"Channels to process: {channels}")
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    total_raw, total_processed, total_skipped = 0, 0, 0
    all_raw_messages = []
    
    for channel in channels:
        logger.info(f"--- Processing Channel: {channel} ---")
        try:
            raw_messages = await scraper.fetch_messages(channel, limit=MESSAGE_LIMIT_PER_CHANNEL)
        except Exception as e:
            logger.error(f"Failed to fetch messages for {channel}: {e}")
            total_skipped += 1
            continue
            
        if not raw_messages:
            logger.warning(f"No messages fetched for {channel}. Skipping.")
            total_skipped += 1
            continue
            
        # Validate message structure
        required_columns = ['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path']
        valid_messages = []
        for msg in raw_messages:
            if all(col in msg for col in required_columns):
                valid_messages.append(msg)
            else:
                logger.warning(f"Message missing required columns: {msg}")
        
        if not valid_messages:
            logger.warning(f"No valid messages for {channel}. Skipping.")
            total_skipped += 1
            continue
        
        # Save raw messages for this channel
        raw_path = os.path.join(RAW_DATA_DIR, f"{channel.strip('@')}_raw.csv")
        try:
            df_raw = pd.DataFrame(valid_messages)
            df_raw.to_csv(raw_path, index=False, encoding='utf-8')
            logger.info(f"Saved {len(valid_messages)} raw messages for {channel}")
        except Exception as e:
            logger.error(f"Failed to save raw messages for {channel}: {e}")
        
        total_raw += len(valid_messages)
        all_raw_messages.extend(valid_messages)
        
        # Preprocess messages
        logger.info(f"Preprocessing {len(valid_messages)} messages for {channel}...")
        try:
            processed_messages = preprocess_messages(valid_messages)
            # Validate processed output
            if not isinstance(processed_messages, list):
                raise ValueError("Processed messages should be a list.")
            if processed_messages and not isinstance(processed_messages[0], dict):
                raise ValueError("Each processed message should be a dict.")
        except Exception as e:
            logger.error(f"Preprocessing failed for {channel}: {e}")
            total_skipped += 1
            continue
        
        # Save processed messages for this channel
        processed_path = os.path.join(PROCESSED_DATA_DIR, f"{channel.strip('@')}_processed.csv")
        try:
            df_processed = pd.DataFrame(processed_messages)
            df_processed.to_csv(processed_path, index=False, encoding='utf-8')
            logger.info(f"Saved {len(processed_messages)} processed messages for {channel}")
        except Exception as e:
            logger.error(f"Failed to save processed messages for {channel}: {e}")
        
        total_processed += len(processed_messages)
    
    # Save combined raw data
    if all_raw_messages:
        combined_raw_path = os.path.join(RAW_DATA_DIR, "telegram_raw_messages.csv")
        try:
            df_combined = pd.DataFrame(all_raw_messages)
            df_combined.to_csv(combined_raw_path, index=False, encoding='utf-8')
            logger.info(f"Saved combined raw data: {len(all_raw_messages)} messages")
        except Exception as e:
            logger.error(f"Failed to save combined raw data: {e}")
    
    logger.info(f"\n--- Pipeline execution finished! ---")
    logger.info(f"Channels processed: {len(channels)} | Total raw messages: {total_raw} | Total processed: {total_processed} | Channels skipped: {total_skipped}")
    print(f"\n--- Pipeline execution finished! ---\nChannels processed: {len(channels)} | Total raw messages: {total_raw} | Total processed: {total_processed} | Channels skipped: {total_skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Telegram data ingestion pipeline.")
    parser.add_argument("--api_id", type=str, default=API_ID, help="Your Telegram API ID.")
    parser.add_argument("--api_hash", type=str, default=API_HASH, help="Your Telegram API Hash.")
    parser.add_argument("--channels", type=str, nargs="*", help="Optional: List of Telegram channels to process.")
    args = parser.parse_args()

    if not args.api_id or not args.api_hash:
        print("[ERROR] Telegram API ID and Hash must be provided.")
        print("Example: python scripts/task_1_ingest.py --api_id YOUR_ID --api_hash YOUR_HASH")
    else:
        asyncio.run(main(args.api_id, args.api_hash, args.channels)) 