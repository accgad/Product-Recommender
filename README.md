# Product Recommender Chatbot

A conversational product recommender system using NLP and fine-tuned TinyLlama model.

## Project Overview

This project implements a chatbot-based product recommender that:
- Uses a fine-tuned TinyLlama model to understand user preferences
- Processes natural language queries to recommend relevant products
- Provides a conversational interface for better user experience
- Runs on a Flask backend with a simple web frontend

## Project Structure

```
product_recommender/
├── data/                        # Data processing
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data for training
│   └── data_preprocessing.py    # Data preprocessing script
├── model/                       # Model components
│   ├── config.py                # Model configuration
│   ├── fine_tuning.py           # TinyLlama fine-tuning script
│   ├── inference.py             # Inference logic
│   └── utils.py                 # Helper functions
├── api/                         # API components
│   ├── app.py                   # Flask application
│   ├── routes.py                # API endpoints
│   └── utils.py                 # API utilities
├── frontend/                    # Frontend components
│   ├── static/                  # Static files (CSS, JS)
│   └── templates/               # HTML templates
│       └── index.html           # Simple chat interface
├── scripts/                     # Utility scripts
│   ├── download_data.py         # Script to download data
│   └── train_model.py           # Training script
├── requirements.txt             # Project dependencies
├── main.py                      # Main script to run components
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

The project can be run step by step:

### 1. Download and Preprocess Data

```
python main.py --action download --category Electronics --samples 100000
```

This will:
- Download Amazon product reviews data for the Electronics category
- Preprocess the data for fine-tuning
- Create training files in the `data/processed` directory

### 2. Fine-tune the TinyLlama Model

```
python main.py --action train --epochs 3
```

This will:
- Fine-tune TinyLlama on the processed data
- Save the fine-tuned model to `model/output/final_model`

### 3. Run the API Server

```
python main.py --action serve
```

This will:
- Start the Flask API server
- Load the fine-tuned model
- Serve the web interface at http://localhost:5000

## How It Works

### Data Processing

1. Downloads Amazon product reviews dataset
2. Processes the reviews to create training examples
3. Formats data for conversational fine-tuning

### Model Fine-tuning

1. Loads the TinyLlama base model
2. Fine-tunes using LoRA (Low-Rank Adaptation) for efficiency
3. Trains the model to understand product preferences and make recommendations

### Inference

1. Takes user input through the chatbot interface
2. Processes the input through the fine-tuned model
3. Extracts product recommendations from the model output
4. Provides detailed information about recommended products

### API and Frontend

1. Flask API handles communication between the frontend and model
2. Simple web interface provides a chat-like experience
3. Shows product details alongside recommendations

## Customization

- **Different Dataset**: Change the `--category` parameter to use different Amazon product categories
- **Model Size**: Modify `model/config.py` to use a different base model
- **Training Parameters**: Adjust learning rate, batch size, etc. in `model/config.py`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Flask 2.2+
- 8+ GB RAM
- GPU recommended for training (CPU can be used but will be slow)

## Limitations

- Training requires significant computational resources
- The model is limited by the quality and scope of the training data
- Product details are limited to what's available in the dataset

## Future Improvements

- Implement multi-turn conversational capabilities
- Add product images and more detailed information
- Improve recommendation relevance with user feedback
- Add search functionality across product database
- Implement user profiles and personalized recommendations
