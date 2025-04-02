import os
import cv2
import torch
import csv
import time
import statistics
import nltk
from collections import Counter
from transformers import pipeline
from PIL import Image
from config import *

# === ANSI Color Codes ===
DARK_GREEN = "\033[38;2;0;128;0m"
CYAN = "\033[38;2;0;255;255m"
YELLOW = "\033[38;2;255;255;0m"
RED = "\033[38;2;255;0;0m"
MAGENTA = "\033[38;2;255;0;255m"
RESET = "\033[0m"

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = interval * fps if fps > 0 else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))
        frame_count += 1

    cap.release()
    return frames, total_frames, video_duration

'''def generate_caption(image, processor, model, device):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(caption_ids[0], skip_special_tokens=True)'''

def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    generated_ids = blip_model.generate(pixel_values=inputs["pixel_values"], max_new_tokens=60)
    return processor.decode(generated_ids[0], skip_special_tokens=True)

def save_results(video_filename, video_output_dir, csv_data, total_stats):
    csv_filename = os.path.join(video_output_dir, f"{video_filename[:-4]}_captions.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Image File", "Caption", "Processing Time (s)"])
        writer.writerows(csv_data)
        writer.writerow([])
        writer.writerow(["Video Stats"])
        for key, value in total_stats.items():
            writer.writerow([key, value])

def print_summary(final_summary, total_stats):
    print("\n=== Final Scene Summary ===")
    print(final_summary)
    print("\n=== Processing Time Stats ===")
    for key, value in total_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")