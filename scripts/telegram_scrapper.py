from telethon import TelegramClient
import csv
import os
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables once
load_dotenv('.env')
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
# phone = os.getenv('phone')

CACHE_FILE = 'data/scraped_message_id.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            return json.load(file)  
    return {}  


def save_cache(data):
    with open(CACHE_FILE, 'w') as file:  
        json.dump(data, file)


# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir, cache):
    entity = await client.get_entity(channel_username)
    channel_title = entity.title  # Extract the channel's title

    async for message in client.iter_messages(entity, limit=10000):
        
        try:
            if message.id in cache[channel_username]:
                continue
        except:
            cache[channel_username] = []

        media_path = None
        if message.media and hasattr(message.media, 'photo'):
            # Create a unique filename for the photo
            filename = f"{channel_username}_{message.id}.jpg"
            media_path = os.path.join(media_dir, filename)
            # Download the media to the specified directory if it's a photo
            await client.download_media(message.media, media_path)
        
        # Write the channel title along with other data
        writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])

        
        cache[channel_username].append(message.id)


        save_cache(cache)

# Initialize the client once
client = TelegramClient('scraping_session', api_id, api_hash)

async def main():
    await client.start()
    
    # Create a directory for media files
    media_dir = 'data/photos'
    os.makedirs(media_dir, exist_ok=True)

    # Open the CSV file and prepare the writer
    with open('data/telegram_data.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        cache = load_cache()

        if os.stat('data/telegram_data.csv').st_size==0:
            writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])  # Include channel title in the header
        
        # List of channels to scrape
        channels = [
            # '@Shageronlinestore',  # Existing channel
            # '@ZemenExpress',
            # '@sinayelj',
            # '@modernshoppingcenter',
            # '@Shewabrand',
            # '@helloomarketethiopia',
            # 't.me/machesmarket',
            # 't.me/+OHreoW14-IQxODFk',
            # 't.me/ratazbrand1',
            # 't.me/AnchorRealEstateMarketing',
            # '@ethio_brand_collection',
            # '@EthioBrandWomen',
            # '@ratazbrand1',
            '@classybrands'
                 # You can add more channels here
            
        ]

        tasks = [scrape_channel(client, channel, writer, media_dir, cache) for channel in channels]

        await asyncio.gather(*tasks)

        print("Scraping completed.")        
        
        # # Iterate over channels and scrape data into the single CSV file
        # for channel in channels:
        #     await scrape_channel(client, channel, writer, media_dir, cache)
        #     print(f"Scraped data from {channel}")

        

with client:
    client.loop.run_until_complete(main())
