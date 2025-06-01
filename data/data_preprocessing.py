"""
Script to preprocess Amazon product reviews data from .txt format.
"""
import os
import json
import pandas as pd
from tqdm import tqdm
import argparse
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Amazon reviews data from .txt format')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the .txt reviews file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Number of samples to use (default: 100000)')
    return parser.parse_args()

def parse_txt_reviews(file_path, max_samples=100000):
    """
    Parse Amazon reviews from custom .txt format.
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines to separate products
    product_blocks = content.split('\n\n')
    
    print(f"Found {len(product_blocks)} product blocks")
    
    for i, block in enumerate(tqdm(product_blocks[:max_samples], desc="Parsing reviews")):
        if not block.strip():
            continue
            
        # Parse each product block
        product_data = {}
        
        for line in block.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'product/productId':
                    product_data['asin'] = value
                elif key == 'product/title':
                    product_data['title'] = value
                elif key == 'product/price':
                    product_data['price'] = value
                elif key == 'review/score':
                    try:
                        product_data['overall'] = float(value)
                    except:
                        product_data['overall'] = 0
                elif key == 'review/summary':
                    product_data['summary'] = value
                elif key == 'review/text':
                    product_data['reviewText'] = value
        
        # Only add if we have the essential fields
        if all(k in product_data for k in ['asin', 'title', 'reviewText', 'summary', 'overall']):
            data.append(product_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"Successfully parsed {len(df)} reviews")
    
    return df

def clean_text(text):
    """
    Clean and tokenize text.
    """
    if isinstance(text, str):
        # Remove quotes and clean
        text = text.strip('"').strip("'")
        # Lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Join tokens back
        text = ' '.join(tokens)
    else:
        text = ''
    return text

def preprocess_data(df):
    """
    Preprocess the reviews data.
    """
    # Clean text
    print("Cleaning text...")
    df['reviewText'] = df['reviewText'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)
    df['title'] = df['title'].apply(clean_text)
    
    # Remove rows with empty reviews
    df = df.dropna(subset=['reviewText', 'summary', 'title'])
    
    # Create a combined text field for training
    df['combined_text'] = df['summary'] + " " + df['reviewText']
    
    # Filter reviews based on length
    df = df[df['combined_text'].str.split().str.len() > 10]
    
    # Convert ratings to float
    df['overall'] = pd.to_numeric(df['overall'], errors='coerce')
    df = df.dropna(subset=['overall'])
    
    return df

def create_training_data(df, output_dir):
    """
    Create training data in format suitable for TinyLlama fine-tuning.
    """
    # Create conversation format for training
    conversations = []
    
    # Group by product (asin)
    product_groups = df.groupby('asin')
    
    # Create conversations with product recommendations
    for asin, group in tqdm(product_groups, desc="Creating conversations"):
        # Only use groups with enough reviews
        if len(group) < 1:
            continue
            
        # Get positive reviews (rating >= 4)
        positive_reviews = group[group['overall'] >= 4.0]
        if len(positive_reviews) < 1:
            continue
        
        # Get product title from the group
        product_title = group['title'].iloc[0]
        
        # Sample some reviews for training examples
        sampled_reviews = positive_reviews.sample(min(3, len(positive_reviews)))
        
        for _, review in sampled_reviews.iterrows():
            # Create user message (query based on summary)
            user_query = f"I'm looking for {review['summary']}. Can you recommend something?"
            
            # Create assistant response (recommendation with product title and ID)
            product_id = review['asin']
            recommendation = f"Based on your preferences, I recommend this product (ID: {product_id}). {product_title.title()}. It has received positive reviews, with users highlighting: {review['reviewText'][:150]}..."
            
            # Create a conversation sample
            conversation = {
                "messages": [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": recommendation}
                ]
            }
            conversations.append(conversation)
    
    # Create some negative examples
    for i in range(len(conversations) // 10):  # Create 10% negative examples
        user_query = "I'm looking for something specific but unusual."
        response = "I don't have enough information to make a specific product recommendation based on your preferences. Could you provide more details about what features or characteristics are important to you?"
        
        conversation = {
            "messages": [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": response}
            ]
        }
        conversations.append(conversation)
    
    # Shuffle conversations
    random.shuffle(conversations)
    
    # Split into train/val/test
    train_data, test_data = train_test_split(conversations, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    # Save data
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Create product metadata (for the lookup we discussed earlier)
    product_metadata = []
    for asin, group in product_groups:
        positive_reviews = group[group['overall'] >= 4.0]
        if len(positive_reviews) > 0:
            product_metadata.append({
                'asin': asin,
                'title': group['title'].iloc[0]
            })
    
    # Save the lightweight product lookup
    with open(os.path.join(output_dir, 'product_lookup.json'), 'w') as f:
        json.dump(product_metadata, f, indent=2)
    
    print(f"Created {len(train_data)} training examples, {len(val_data)} validation examples, and {len(test_data)} test examples.")
    print(f"Created product lookup with {len(product_metadata)} products.")
    
    return train_data, val_data, test_data, product_metadata

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return
    
    # Parse reviews from txt file
    df = parse_txt_reviews(args.input_file, max_samples=args.samples)
    
    if df.empty:
        print("No data found. Check your input file format.")
        return
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Create training data
    create_training_data(df, args.output_dir)

if __name__ == "__main__":
    main()
