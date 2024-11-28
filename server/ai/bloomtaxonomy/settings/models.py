import os

# Possible models : 
## llama3-8b-8192
## llama3-70b-8192
## gemma-7b-it
## mixtral-8x7b-32768

MODEL_NAME = os.getenv('MODEL_NAME', 'llama3-70b-8192')  # Default model