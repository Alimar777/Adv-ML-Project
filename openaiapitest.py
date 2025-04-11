# Import the necessary modules
import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the current directory
load_dotenv()  # This will look for a .env file in the current directory and load it

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("API key loaded successfully")
else:
    print("API key not found. Please check your .env file.")


