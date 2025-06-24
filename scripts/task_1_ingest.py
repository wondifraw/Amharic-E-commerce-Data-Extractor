"""
task_1_ingest.py
Main script to run the data ingestion and preprocessing pipeline.

This script orchestrates the process of fetching data from Telegram channels,
preprocessing it, and saving both raw and processed versions.
"""
import asyncio
import os
import argparse
import logging
from src.telegram_data_ingestion.telegram_scraper import TelegramScraper
from src.amharic_text_processing.amharic_text import preprocess_messages

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME = "telegram_session"

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
MESSAGE_LIMIT_PER_CHANNEL = 200

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("task_1_ingest")

async def main(api_id: str, api_hash: str, channel_list=None):
    """
    Main function to execute the data ingestion and preprocessing pipeline.
    Logs metrics and errors, validates output, and prints a summary report.
    """
    logger.info("--- Starting Data Ingestion and Preprocessing Pipeline ---")
    
    scraper = TelegramScraper(api_id, api_hash, SESSION_NAME)
    channels = channel_list if channel_list else scraper.get_channel_list()
    logger.info(f"Channels to process: {channels}")
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    total_raw, total_processed, total_skipped = 0, 0, 0
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
        raw_path = os.path.join(RAW_DATA_DIR, f"{channel.strip('@')}_raw.json")
        try:
            scraper.save_data(raw_messages, raw_path)
        except Exception as e:
            logger.error(f"Failed to save raw messages for {channel}: {e}")
        logger.info(f"Fetched {len(raw_messages)} messages for {channel}.")
        total_raw += len(raw_messages)
        logger.info(f"Preprocessing {len(raw_messages)} messages for {channel}...")
        try:
            processed_messages = preprocess_messages(raw_messages)
            # Validate processed output
            if not isinstance(processed_messages, list):
                raise ValueError("Processed messages should be a list.")
            if processed_messages and not isinstance(processed_messages[0], dict):
                raise ValueError("Each processed message should be a dict.")
        except Exception as e:
            logger.error(f"Preprocessing failed for {channel}: {e}")
            total_skipped += 1
            continue
        processed_path = os.path.join(PROCESSED_DATA_DIR, f"{channel.strip('@')}_processed.json")
        try:
            scraper.save_data(processed_messages, processed_path)
        except Exception as e:
            logger.error(f"Failed to save processed messages for {channel}: {e}")
        logger.info(f"Saved {len(processed_messages)} processed messages for {channel}.")
        total_processed += len(processed_messages)
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