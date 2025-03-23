import cv2
import torch
import os
import csv
import time
import statistics
import nltk
from collections import Counter
from paths import *  # Ensure this is correctly set
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load summarization and rephrasing models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
rephraser = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)

# Download necessary NLTK resources
nltk.download('punkt')

# Frame capture settings
interval = 1  # Capture every 1 second

# Function to extract frames from video
def extract_frames(video_path, interval=interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = interval * fps  # Capture every 'interval' seconds
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

# Function to generate captions using BLIP
def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(caption_ids[0], skip_special_tokens=True)

# Function to clean captions and remove hallucinated entities
def clean_captions(captions):
    words = [word.lower() for caption in captions for word in nltk.word_tokenize(caption)]
    entity_counts = Counter(words)

    # Keep only frequently occurring words (min 2 appearances)
    min_occurrences = 2
    valid_entities = {word for word, count in entity_counts.items() if count >= min_occurrences}

    # Filter captions that contain valid entities
    refined_captions = [caption for caption in captions if any(word in caption.lower() for word in valid_entities)]
    return refined_captions

# Function to summarize refined captions
def summarize_captions(captions):
    refined_captions = clean_captions(captions)
    text_to_summarize = " ".join(refined_captions)
    
    if not text_to_summarize:
        return "No meaningful summary could be generated."
    
    summary = summarizer(text_to_summarize, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]
    return summary

# Function to rephrase summary into a more natural flow
def rephrase_summary(summary):
    prompt = (f"Convert this summary into a natural, flowing narrative with descriptive language: {summary}. "
              "Make it sound engaging and coherent, like a story.")

    response = rephraser(prompt, max_length=120, do_sample=True, temperature=0.7)[0]["generated_text"]
    return response

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
        caption = generate_caption(frame)
        end_time = time.time()

        captions.append(caption)
        frame_times.append(end_time - start_time)
        csv_data.append([frame_number, f"frame_{frame_number}.jpg", caption, end_time - start_time])

    mean_time = statistics.mean(frame_times)
    median_time = statistics.median(frame_times)
    mode_time = statistics.mode(frame_times) if len(frame_times) > 1 else frame_times[0]

    # Summarize and rephrase captions
    summary = summarize_captions(captions)
    final_summary = rephrase_summary(summary)
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    # Save results to CSV
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
        writer.writerow(["Total Processing Time (s)", total_processing_time])
        writer.writerow([])
        writer.writerow(["Summary", final_summary])

    # Print final results
    print("\n=== Final Story-Like Summary ===")
    print(final_summary)
    print("\n=== Processing Time Stats ===")
    print(f"Total Frames: {total_frames}")
    print(f"Video Duration: {video_duration:.2f} seconds")
    print(f"Mean Frame Processing Time: {mean_time:.4f} seconds")
    print(f"Median Frame Processing Time: {median_time:.4f} seconds")
    print(f"Mode Frame Processing Time: {mode_time:.4f} seconds")
    print(f"Total Processing Time: {total_processing_time:.4f} seconds")
