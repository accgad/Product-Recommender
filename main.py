"""
Main script to run the product recommender project.
"""
import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Product Recommender Chatbot")
    parser.add_argument('--action', type=str, required=True,
                       choices=['download', 'preprocess', 'train', 'serve'],
                       help='Action to perform')
    parser.add_argument('--category', type=str, default='Electronics',
                       help='Product category for data (default: Electronics)')
    parser.add_argument('--samples', type=int, default=100000,
                       help='Number of samples to use (default: 100000)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    return parser.parse_args()

def download_and_preprocess(category, samples):
    """Download and preprocess data."""
    print(f"Downloading and preprocessing {category} data...")
    script_path = os.path.join('data', 'data_preprocessing.py')
    cmd = [
        sys.executable, script_path,
        '--category', category,
        '--samples', str(samples)
    ]
    subprocess.run(cmd, check=True)

def train_model(epochs):
    """Train the model."""
    print(f"Training model for {epochs} epochs...")
    script_path = os.path.join('model', 'fine_tuning.py')
    cmd = [
        sys.executable, script_path,
        '--num_epochs', str(epochs)
    ]
    subprocess.run(cmd, check=True)

def serve_api():
    """Start the Flask API server."""
    print("Starting API server...")
    script_path = os.path.join('api', 'app.py')
    cmd = [sys.executable, script_path]
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()
    
    if args.action == 'download':
        download_and_preprocess(args.category, args.samples)
    elif args.action == 'preprocess':
        download_and_preprocess(args.category, args.samples)
    elif args.action == 'train':
        train_model(args.epochs)
    elif args.action == 'serve':
        serve_api()

if __name__ == "__main__":
    main()
