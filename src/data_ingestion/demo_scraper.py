"""
Demo scraper that creates sample data without requiring Telegram API
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import random
from loguru import logger


class DemoTelegramScraper:
    """Creates sample Ethiopian e-commerce data for demonstration"""
    
    def __init__(self):
        self.channels = [
            "@ZemenExpress",
            "@sinayelj",
            "@Shewabrand",
            "@helloomarketethiopia",
            "@ethio_brand_collection",
            "@ratazbrand1",
            "@classybrands",
            "@ethio_telecom_official"
        ]
        
        self.sample_messages = [
            "ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።",
            "አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር",
            "Baby bottle for sale 150 birr in Bole area",
            "መርካቶ ላይ ጫማ 300 ብር የሚሸጥ ነው",
            "ፒያሳ አካባቢ ስልክ ETB 5000 የሚሸጥ",
            "ሰሚት ላይ መጽሐፍ 50 ብር",
            "ሃያ ሁለት አካባቢ ልብስ ዋጋ 180 ብር",
            "ካዛንቺስ ላይ ጫማ በ 250 ብር",
            "ጀሞ አካባቢ የሚሸጥ ስልክ 4500 ብር",
            "አዲስ አበባ ውስጥ baby bottle 140 birr"
        ]
    
    def generate_sample_data(self, total_messages: int = 100) -> pd.DataFrame:
        """Generate sample Ethiopian e-commerce data"""
        logger.info(f"Generating {total_messages} sample messages")
        
        data = []
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(total_messages):
            channel = random.choice(self.channels)
            message = random.choice(self.sample_messages)
            
            # Add some variation to messages
            if random.random() > 0.7:
                price = random.randint(50, 5000)
                message = message.replace("150", str(price)).replace("200", str(price))
            
            data.append({
                'id': i + 1,
                'channel': channel,
                'text': message,
                'date': (start_date + timedelta(hours=random.randint(0, 720))).isoformat(),
                'views': random.randint(50, 1000),
                'forwards': random.randint(0, 50),
                'replies': random.randint(0, 20),
                'sender_id': 12345 + hash(channel) % 1000,
                'has_media': random.random() > 0.7,
                'media_type': 'photo' if random.random() > 0.5 else None,
                'message_link': f"https://t.me/{channel.replace('@', '')}/{i+1}"
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample messages")
        return df
    
    def save_sample_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save sample data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sample_telegram_data_{timestamp}.csv"
        
        output_path = os.path.join("data/raw", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Sample data saved to {output_path}")
        return output_path


def main():
    """Generate sample data for demonstration"""
    scraper = DemoTelegramScraper()
    df = scraper.generate_sample_data(100)
    output_path = scraper.save_sample_data(df)
    
    print(f"Sample data generated: {output_path}")
    print(f"Total messages: {len(df)}")
    print(f"Channels: {df['channel'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()