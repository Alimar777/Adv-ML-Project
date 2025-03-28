import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import os
from paths import VIDEO_PATH  # Ensure paths.py is correctly set

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load summarization and rephrasing models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
rephraser = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)

# Define BLIP output directory
BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "BLIP")
os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

# Frame capture interval
FRAME_INTERVAL = 1
