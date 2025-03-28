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

# Download necessary NLTK resources
nltk.download('punkt')

def extract_frames(video_path, interval=1):
    """Extract frames from a video at a specified interval."""
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

def generate_caption(image, processor, model, device):
    """Generate a caption for an image using BLIP."""
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(caption_ids[0], skip_special_tokens=True)

def clean_captions(captions):
    """Filter hallucinated entities and refine captions."""
    words = [word.lower() for caption in captions for word in nltk.word_tokenize(caption)]
    entity_counts = Counter(words)
    valid_entities = {word for word, count in entity_counts.items() if count >= 2}
    return [caption for caption in captions if any(word in caption.lower() for word in valid_entities)]

def summarize_captions(captions, summarizer):
    """Summarize cleaned captions using a summarization model."""
    refined_captions = clean_captions(captions)
    text_to_summarize = " ".join(refined_captions)
    if not text_to_summarize:
        return "No meaningful summary could be generated."
    return summarizer(text_to_summarize, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]

def rephrase_summary(summary, rephraser):
    """Rephrase the summary to make it more natural and engaging."""
    prompt = (f"Convert this summary into a natural, flowing narrative with descriptive language: {summary}. "
              "Make it sound engaging and coherent, like a story.")
    return rephraser(prompt, max_length=120, do_sample=True, temperature=0.7)[0]["generated_text"]

def save_results(video_filename, video_output_dir, csv_data, total_stats):
    """Save results to a CSV file."""
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
    """Print the final results in a readable format."""
    print("\n=== Final Story-Like Summary ===")
    print(final_summary)
    print("\n=== Processing Time Stats ===")
    for key, value in total_stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
