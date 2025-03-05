import cv2
import torch
import os
import csv
import nltk
from collections import Counter
from paths import *  # Ensure this is correctly set
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image

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

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))  # Store frame number and frame
        frame_count += 1

    cap.release()
    return frames

# Function to generate captions
def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(caption_ids[0], skip_special_tokens=True)

# Function to summarize captions
def summarize_captions(captions):
    words = nltk.word_tokenize(" ".join(captions))
    word_freq = Counter(words)

    stopwords = set(nltk.corpus.stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stopwords]

    important_words = [word for word, freq in word_freq.items() if freq > 1 and word.lower() not in stopwords]

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    summary = summarizer(" ".join(captions), max_length=100, min_length=15, do_sample=False)[0]["summary_text"]

    return summary

# Define BLIP output directory
BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "BLIP")  
os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

# Process videos 01.mp4 to 10.mp4
for i in range(1, 11):
    video_filename = f"{i:02d}.mp4"  # Format to 2-digit numbers (01, 02, ..., 10)
    video_path = os.path.join(VIDEO_PATH, video_filename)

    if not os.path.exists(video_path):
        print(f"Video file {video_filename} not found. Skipping...")
        continue

    # Create directory for this video's outputs
    video_name = os.path.splitext(video_filename)[0]  # Extract name without extension
    video_output_dir = os.path.join(BLIP_OUTPUT_DIR, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    print(f"\nProcessing {video_filename}...\n")

    frames = extract_frames(video_path, interval)

    captions = []  # Store captions for summary
    csv_data = []  # Store frame data for CSV

    # Process each frame
    for frame_number, frame in frames:
        caption = generate_caption(frame)
        captions.append(caption)

        # Save raw frame image
        frame_filename = f"frame_{frame_number}.jpg"
        frame_filepath = os.path.join(video_output_dir, frame_filename)
        cv2.imwrite(frame_filepath, frame)

        # Store frame info for CSV
        csv_data.append([frame_number, frame_filename, caption])

        print(f"Frame {frame_number}: {caption}")

    # Generate summary
    final_summary = summarize_captions(captions)

    # Save captions to CSV
    csv_filename = os.path.join(video_output_dir, f"{video_name}_captions.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Image File", "Caption"])
        writer.writerows(csv_data)
        writer.writerow([])
        writer.writerow(["Summary", final_summary])

    print("\n=== Final Video Summary ===")
    print(final_summary)
