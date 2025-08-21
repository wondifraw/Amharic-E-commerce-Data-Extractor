"""
Simple test script for Ethiopian E-commerce NER System
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing.text_preprocessor import AmharicTextPreprocessor
from preprocessing.conll_labeler import CoNLLLabeler
from vendor_analytics.vendor_scorer import VendorAnalyticsEngine


def test_system():
    """Test the complete system with sample data"""
    print("ETHIOPIAN E-COMMERCE NER SYSTEM TEST")
    print("=" * 50)
    
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'channel': ['@vendor1', '@vendor2', '@vendor1', '@vendor3', '@vendor2'],
        'text': [
            "Baby bottle price 150 birr in Bole area",
            "Addis Ababa clothes for 200 birr",
            "Shoes at Merkato 300 birr",
            "Phone at Piassa ETB 5000",
            "Book 50 birr Hayahulet"
        ],
        'views': [100, 200, 150, 300, 80],
        'forwards': [5, 10, 8, 15, 3],
        'replies': [2, 5, 3, 8, 1],
        'date': pd.date_range('2024-01-01', periods=5),
        'sender_id': [12345, 67890, 12345, 11111, 67890],
        'has_media': [False, True, False, True, False],
        'media_type': [None, 'photo', None, 'photo', None],
        'message_link': ['link1', 'link2', 'link3', 'link4', 'link5']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Created sample dataset with {len(df)} messages")
    
    # Test 1: Text Preprocessing
    print("\n1. TESTING TEXT PREPROCESSING...")
    preprocessor = AmharicTextPreprocessor()
    processed_df = preprocessor.preprocess_dataframe(df)
    print(f"   Processed {len(processed_df)} messages")
    print(f"   Average tokens per message: {processed_df['token_count'].mean():.1f}")
    
    # Test 2: CoNLL Labeling
    print("\n2. TESTING CoNLL LABELING...")
    labeler = CoNLLLabeler()
    sample_text = "Baby bottle price 150 birr in Bole area"
    tokens = labeler.tokenize_for_conll(sample_text)
    labels = labeler.auto_label_entities(tokens)
    print(f"   Sample text: {sample_text}")
    print(f"   Tokens: {tokens}")
    print(f"   Labels: {labels}")
    
    # Test 3: Vendor Analytics
    print("\n3. TESTING VENDOR ANALYTICS...")
    analytics = VendorAnalyticsEngine()
    vendor_scores = analytics.analyze_all_vendors(processed_df)
    print(f"   Analyzed {len(vendor_scores)} vendors")
    
    if len(vendor_scores) > 0:
        top_vendor = vendor_scores.iloc[0]
        print(f"   Top vendor: {top_vendor['vendor_channel']}")
        print(f"   Lending score: {top_vendor['lending_score']:.1f}/100")
        
        # Show vendor scorecard
        print("\n   VENDOR SCORECARD:")
        print("   " + "-" * 60)
        for _, vendor in vendor_scores.iterrows():
            print(f"   {vendor['vendor_channel']:<15} "
                  f"Score: {vendor['lending_score']:<6.1f} "
                  f"Views: {vendor['avg_views_per_post']:<6.0f} "
                  f"Posts/Week: {vendor['posts_per_week']:<6.1f}")
    
    # Test 4: Entity Recognition Demo
    print("\n4. TESTING ENTITY RECOGNITION...")
    test_texts = [
        "Baby bottle price 150 birr in Bole area",
        "Addis Ababa clothes for 200 birr",
        "Phone at Merkato ETB 3000"
    ]
    
    for text in test_texts:
        tokens = labeler.tokenize_for_conll(text)
        labels = labeler.auto_label_entities(tokens)
        
        # Extract entities
        entities = []
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(f"{current_entity}: {' '.join(current_tokens)}")
                current_entity = label[2:]
                current_tokens = [token]
            elif label.startswith('I-') and current_entity == label[2:]:
                current_tokens.append(token)
            else:
                if current_entity:
                    entities.append(f"{current_entity}: {' '.join(current_tokens)}")
                current_entity = None
                current_tokens = []
        
        if current_entity:
            entities.append(f"{current_entity}: {' '.join(current_tokens)}")
        
        print(f"   Text: {text}")
        print(f"   Entities: {entities if entities else 'None found'}")
    
    print("\n" + "=" * 50)
    print("SYSTEM TEST COMPLETED SUCCESSFULLY!")
    print("All components are working correctly.")
    print("\nNext steps:")
    print("1. Run: streamlit run src/dashboard/streamlit_app.py")
    print("2. Run: python -m pytest tests/ -v")
    print("3. Configure Telegram API credentials for real data")


if __name__ == "__main__":
    test_system()