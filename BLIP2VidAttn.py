import cv2
import torch
import os
import csv
import time
import statistics
import numpy as np
import nltk
from collections import Counter
from paths import *  # Ensure this is correctly set
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import matplotlib.pyplot as plt

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

nltk.download('punkt')
nltk.download('stopwords')

interval = 1  # Capture frame every 1 second

# Function to extract frames from a video at intervals
def extract_frames(video_path, interval=interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = interval * fps  # Capture every 'interval' seconds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0  # In seconds

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))  # Store frame number and frame
        frame_count += 1

    cap.release()
    return frames, total_frames, video_duration

# Function to extract attention maps
def get_attention_map(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.vision_model(**inputs)  # Extract vision transformer embeddings

    attention = outputs.last_hidden_state  # Shape: [1, num_patches+1, hidden_dim]
    attention = attention[:, 1:, :]  # Remove CLS token
    attention = attention.mean(dim=-1).squeeze(0).cpu().numpy()  # Average over hidden dim
    
    return attention  # Shape: [num_patches]

# Function to visualize attention on image
def visualize_attention(image, attention_map, save_path):
    h, w, _ = image.shape
    grid_size = int(np.sqrt(len(attention_map)))  # ViT splits image into patches
    
    attention_map = attention_map.reshape(grid_size, grid_size)
    attention_map = cv2.resize(attention_map, (w, h))  # Resize to match original image

    # Normalize attention map
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

    # Convert attention to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlayed_image)

# Function to generate captions and save attention visualization
def generate_caption(image, save_attention_path):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)

    caption_ids = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)

    # Extract attention map
    attention_map = get_attention_map(image)
    visualize_attention(image, attention_map, save_attention_path)

    return caption

# Initialize summarizer before timing
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Function to summarize captions
def summarize_captions(captions):
    start_summary_time = time.time()
    summary = summarizer(" ".join(captions), max_length=100, min_length=15, do_sample=False)[0]["summary_text"]
    end_summary_time = time.time()
    return summary, end_summary_time - start_summary_time

# Define BLIP output directory
BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "BLIP")  
os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

# Process videos 01.mp4 to 10.mp4
for i in range(1, 11):
    video_filename = f"{i:02d}.mp4"
    video_path = os.path.join(VIDEO_PATH, video_filename)
    if not os.path.exists(video_path):
        print(f"Video file {video_filename} not found. Skipping...")
        continue

    video_output_dir = os.path.join(BLIP_OUTPUT_DIR, video_filename[:-4])
    os.makedirs(video_output_dir, exist_ok=True)

    print(f"\nProcessing {video_filename}...\n")

    frames, total_frames, video_duration = extract_frames(video_path, interval)
    captions, csv_data, frame_times = [], [], []
    total_start_time = time.time()

    for frame_number, frame in frames:
        start_time = time.time()
        
        attention_save_path = os.path.join(video_output_dir, f"frame_{frame_number}_attention.jpg")
        caption = generate_caption(frame, attention_save_path)
        
        end_time = time.time()

        captions.append(caption)
        frame_times.append(end_time - start_time)
        csv_data.append([frame_number, f"frame_{frame_number}.jpg", caption, end_time - start_time])

    mean_time = statistics.mean(frame_times)
    median_time = statistics.median(frame_times)
    mode_time = statistics.mode(frame_times) if len(frame_times) > 1 else frame_times[0]

    final_summary, summary_time = summarize_captions(captions)
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    csv_filename = os.path.join(video_output_dir, f"{video_filename[:-4]}_captions.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Image File", "Caption", "Processing Time (s)"])
        writer.writerows(csv_data)
        writer.writerow([])
        writer.writerow(["Video Stats"])
        writer.writerow(["Total Frames", total_frames])
        writer.writerow(["Video Duration (s)", video_duration])
        writer.writerow(["Mean Processing Time (s)", mean_time])
        writer.writerow(["Median Processing Time (s)", median_time])
        writer.writerow(["Mode Processing Time (s)", mode_time])
        writer.writerow(["Summary Processing Time (s)", summary_time])
        writer.writerow(["Total Processing Time (s)", total_processing_time])
        writer.writerow([])
        writer.writerow(["Summary", final_summary])

    print("\n=== Final Video Summary ===")
    print(final_summary)
    print("\n=== Processing Time Stats ===")
    print(f"Total Frames: {total_frames}")
    print(f"Video Duration: {video_duration:.2f} seconds")
    print(f"Mean Frame Processing Time: {mean_time:.4f} seconds")
    print(f"Median Frame Processing Time: {median_time:.4f} seconds")
    print(f"Mode Frame Processing Time: {mode_time:.4f} seconds")
    print(f"Summary Processing Time: {summary_time:.4f} seconds")
    print(f"Total Processing Time: {total_processing_time:.4f} seconds")
