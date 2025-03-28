import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from paths import VIDEO_PATH

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Output directory for BLIP captions
BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "BLIP")
os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

# Frame capture interval in seconds
FRAME_INTERVAL = 1
