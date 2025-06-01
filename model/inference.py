"""
Inference logic for the product recommender chatbot.
"""
import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from config, with fallbacks
try:
    from model.config import MODEL_CONFIG, DATA_PATHS, CHAT_TEMPLATE
except ImportError:
    print("Warning: Config not found, using default values")
    MODEL_CONFIG = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "output_dir": "model/output"
    }
    DATA_PATHS = {
        "metadata": "data/processed/product_lookup.json"
    }
    CHAT_TEMPLATE = "<|user|>\n{user_input}\n<|assistant|>\n"


class ProductRecommender:
    def __init__(self, model_path=None, metadata_path=None):
        """Initialize the product recommender with the fine-tuned model."""
        # Use default paths if not provided
        if model_path is None:
            model_path = os.path.join(MODEL_CONFIG["output_dir"], "final_model")
        
        if metadata_path is None:
            metadata_path = DATA_PATHS["metadata"]
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please fine-tune the model first.")
        
        # Load product metadata (from your preprocessing script's product_lookup.json)
        self.product_lookup = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    product_data = json.load(f)
                    # Convert list to dict for faster lookup
                    if isinstance(product_data, list):
                        self.product_lookup = {item['asin']: item['title'] for item in product_data}
                    else:
                        self.product_lookup = product_data
                print(f"Loaded product lookup for {len(self.product_lookup)} products")
            except Exception as e:
                print(f"Warning: Could not load product metadata from {metadata_path}: {e}")
        else:
            print(f"Warning: Product metadata not found at {metadata_path}")
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}")
        
        # Check if this is a PEFT model by looking for adapter files
        is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_peft_model:
            print("Detected PEFT model. Loading base model and adapter...")
            
            # Load tokenizer from base model (not from adapter path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIG["base_model"], 
                trust_remote_code=True
            )
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIG["base_model"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load PEFT adapter
            self.model = PeftModel.from_pretrained(base_model, model_path)
            
            # Optional: Merge adapter with base model for faster inference
            print("Merging adapter with base model...")
            self.model = self.model.merge_and_unload()
            
        else:
            print("Loading full fine-tuned model...")
            # Load tokenizer from saved model path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Put model in evaluation mode
        self.model.eval()
        print("Model loaded successfully")
    
    def extract_product_ids(self, text):
        """Extract product IDs from model response."""
        # Look for product IDs in various formats
        patterns = [
            r"ID: ([A-Z0-9]{10})",  # Format: ID: B0000630MQ
            r"Product ID: ([A-Z0-9]{10})",  # Format: Product ID: B0000630MQ
            r"\(ID: ([A-Z0-9]{10})\)",  # Format: (ID: B0000630MQ)
            r"([A-Z0-9]{10})"  # Just the ID itself
        ]
        
        found_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            found_ids.extend(matches)
        
        # Remove duplicates while preserving order
        unique_ids = []
        for id_val in found_ids:
            if id_val not in unique_ids:
                unique_ids.append(id_val)
        
        return unique_ids
    
    def get_product_name(self, product_id):
        """Get product name from lookup table."""
        return self.product_lookup.get(product_id, None)
    
    def enhance_response(self, response):
        """Enhance response by replacing IDs with product names where possible."""
        enhanced_response = response
        product_ids = self.extract_product_ids(response)
        
        replaced_products = []
        for product_id in product_ids:
            product_name = self.get_product_name(product_id)
            if product_name:
                # Replace ID with name in the response
                enhanced_response = enhanced_response.replace(
                    f"ID: {product_id}", 
                    f"'{product_name}' (ID: {product_id})"
                )
                enhanced_response = enhanced_response.replace(
                    f"Product ID: {product_id}", 
                    f"'{product_name}' (Product ID: {product_id})"
                )
                replaced_products.append({
                    'id': product_id,
                    'name': product_name
                })
        
        return enhanced_response, replaced_products
    
    def recommend(self, user_input, max_length=512, temperature=0.7, top_p=0.9):
        """Generate product recommendation based on user input."""
        try:
            # Format input using chat template
            formatted_input = CHAT_TEMPLATE.format(user_input=user_input)
            
            # Tokenize input
            inputs = self.tokenizer(formatted_input, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response part only (after assistant tag)
            assistant_markers = ["<|assistant|>", "<|assistant|>\n"]
            for marker in assistant_markers:
                assistant_pos = response.find(marker)
                if assistant_pos != -1:
                    response = response[assistant_pos + len(marker):].strip()
                    break
            
            # Extract product IDs
            product_ids = self.extract_product_ids(response)
            
            # Enhance response with product names
            enhanced_response, replaced_products = self.enhance_response(response)
            
            return {
                "original_response": response,
                "enhanced_response": enhanced_response,
                "product_ids": product_ids,
                "products": replaced_products
            }
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return {
                "original_response": "I apologize, but I'm having trouble generating a recommendation right now.",
                "enhanced_response": "I apologize, but I'm having trouble generating a recommendation right now.",
                "product_ids": [],
                "products": []
            }
    
    def generate_response(self, conversation_history, new_user_input, max_length=512):
        """Generate response based on conversation history."""
        # For now, we'll just use the last user message
        # In a more advanced implementation, we could use the entire conversation history
        return self.recommend(new_user_input, max_length=max_length)


# Simple test function
def test_recommender(model_path=None):
    """Test the product recommender with some sample queries."""
    try:
        recommender = ProductRecommender(model_path)
        
        test_inputs = [
            "I'm looking for a good battery charger for my camera",
            "I need something to charge my AA batteries quickly",
            "Can you recommend a reliable battery charger?",
            "I want a charger that doesn't overheat my batteries"
        ]
        
        print("Testing Product Recommender:")
        print("=" * 50)
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n{i}. User: {user_input}")
            result = recommender.recommend(user_input)
            print(f"   Bot: {result['enhanced_response']}")
            
            if result['products']:
                print("   Found products:")
                for product in result['products']:
                    print(f"   - {product['name']} (ID: {product['id']})")
            
            print("-" * 30)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have fine-tuned the model first by running:")
        print("python model/fine_tuning.py")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_recommender()
