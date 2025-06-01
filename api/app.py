"""
Flask API for the product recommender chatbot.
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import sys
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.inference import ProductRecommender
from model.config import MODEL_CONFIG


# Initialize Flask app
app = Flask(__name__, 
            static_folder="../frontend/static",
            template_folder="../frontend/templates")
CORS(app)  # Enable CORS for all routes

# Global recommender instance
recommender = None

# Store conversation history (in memory for simplicity)
# For production, use a database
conversations = {}


def initialize_recommender():
    """Initialize the recommender model."""
    global recommender
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             MODEL_CONFIG["output_dir"], "final_model")
    
    try:
        recommender = ProductRecommender(model_path)
        return True
    except Exception as e:
        print(f"Error initializing recommender: {e}")
        return False


@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global recommender
    is_model_loaded = recommender is not None
    return jsonify({
        'status': 'ok',
        'model_loaded': is_model_loaded
    })


@app.route('/api/chat/start', methods=['POST'])
def start_conversation():
    """Start a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = []
    
    return jsonify({
        'conversation_id': conversation_id,
        'message': 'Conversation started'
    })


@app.route('/api/chat/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get conversation history."""
    if conversation_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404
    
    return jsonify({
        'conversation_id': conversation_id,
        'messages': conversations[conversation_id]
    })


@app.route('/api/chat/<conversation_id>/message', methods=['POST'])
def send_message(conversation_id):
    """Send a message to the chatbot."""
    global recommender
    
    # Check if conversation exists
    if conversation_id not in conversations:
        return jsonify({'error': 'Conversation not found'}), 404
    
    # Check if model is loaded
    if recommender is None:
        success = initialize_recommender()
        if not success:
            return jsonify({'error': 'Model not initialized'}), 500
    
    # Get user message
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400
    
    user_message = data['message']
    
    # Add user message to conversation
    conversations[conversation_id].append({
        'role': 'user',
        'content': user_message
    })
    
    try:
        # Generate response
        result = recommender.generate_response(
            conversations[conversation_id],
            user_message
        )
        
        # Create response
        response = {
            'role': 'assistant',
            'content': result['response']
        }
        
        # Add product details if available
        if result['product_id'] and result['product_details']:
            response['product'] = {
                'id': result['product_id'],
                'details': result['product_details']
            }
        
        # Add response to conversation
        conversations[conversation_id].append(response)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({'error': 'Failed to generate response'}), 500


@app.route('/api/products', methods=['GET'])
def search_products():
    """Search for products."""
    global recommender
    
    # Check if model is loaded
    if recommender is None:
        success = initialize_recommender()
        if not success:
            return jsonify({'error': 'Model not initialized'}), 500
    
    # Get search query
    query = request.args.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # For now, just return some random products from the metadata
    # In a real implementation, you would use a search index or database
    products = []
    for product_id, product in list(recommender.product_metadata.items())[:10]:
        products.append({
            'id': product_id,
            'details': product
        })
    
    return jsonify({
        'query': query,
        'products': products
    })


def main():
    """Run the Flask app."""
    # Initialize recommender
    initialize_recommender()
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
