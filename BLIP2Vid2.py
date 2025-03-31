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

# === Global Settings ===
VIDEO_SELECTION_MODE = "single"  # Options: "all", "single"
SELECTED_VIDEO_INDEX = 3  # Only used if VIDEO_SELECTION_MODE == "single"
DETAIL_MODE = True  # Print captions for each frame in single mode
SUMMARY_MODEL = "gpt2"  # Options: "bart", "gpt2", "llama"

# === Prompt Engineering (for GPT2/LLaMA transitions) ===
PROMPT_STYLE = (
    "\n\nDescribe the visual change between the two scenes below "
    "in one concise sentence. \n\nPrevious: {prev}\nCurrent: {curr}\nTransition:"
)

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


nltk.download('punkt')
nltk.download('stopwords')

interval = 1  # Capture frame every 1 second

# Load the appropriate summarizer based on model switch
def load_summarizer():
    global tokenizer, model  # So transition function can access them
    if SUMMARY_MODEL == "bart":
        return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

    elif SUMMARY_MODEL == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

        def summarize_gpt2(text):
            inputs = tokenizer.encode("Summarize: " + text, return_tensors="pt", truncation=True).to(device)
            attention_mask = torch.ones_like(inputs)  # GPT2 has no pad token, but this avoids the warning
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id  # prevents generation error
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summarize_gpt2

    elif SUMMARY_MODEL == "llama":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)

        def summarize_llama(text):
            inputs = tokenizer.encode("Summarize: " + text, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summarize_llama

    else:
        raise ValueError("Unsupported SUMMARY_MODEL")

summarizer = load_summarizer()

# Function to extract frames from a video at intervals
def extract_frames(video_path, interval=interval):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = interval * fps
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

# Function to generate captions
def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)

    # Run BLIP encoder + generate
    generated_ids = blip_model.generate(pixel_values=inputs["pixel_values"], max_new_tokens=60)
    return processor.decode(generated_ids[0], skip_special_tokens=True)





# Function to summarize full video captions
def summarize_captions(captions):
    start_summary_time = time.time()
    joined = " ".join(captions)

    if SUMMARY_MODEL == "bart":
        summary = summarizer(joined, max_length=100, min_length=15, do_sample=False)[0]["summary_text"]
    else:
        summary = summarizer(joined)

    end_summary_time = time.time()
    return summary, end_summary_time - start_summary_time

# Function to summarize transition between two captions (for GPT2/LLaMA)
def summarize_transition(prev_caption, curr_caption):
    if SUMMARY_MODEL not in ["gpt2", "llama"]:
        raise ValueError("Transition summarization is only supported for gpt2 or llama.")

    prompt = PROMPT_STYLE.format(prev=prev_caption, curr=curr_caption)
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(inputs, max_length=60, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define BLIP output directory
BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "BLIP")  
os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

video_indices = range(1, 11) if VIDEO_SELECTION_MODE == "all" else [SELECTED_VIDEO_INDEX]

for i in video_indices:
    video_filename = f"{i:02d}.mp4"
    video_path = os.path.join(VIDEO_PATH, video_filename)
    if not os.path.exists(video_path):
        print(f"Video file {video_filename} not found. Skipping...")
        continue

    video_output_dir = os.path.join(BLIP_OUTPUT_DIR, video_filename[:-4])
    os.makedirs(video_output_dir, exist_ok=True)

    summary_output_dir = os.path.join(video_output_dir, "Summaries", SUMMARY_MODEL)
    os.makedirs(summary_output_dir, exist_ok=True)

    print(f"\nProcessing {video_filename}...\n")

    frames, total_frames, video_duration = extract_frames(video_path, interval)
    captions, csv_data, frame_times = [], [], []
    total_start_time = time.time()

    transitions = []  # Store transitions for saving later

    for frame_number, frame in frames:
        start_time = time.time()
        caption = generate_caption(frame)
        end_time = time.time()

        captions.append(caption)
        frame_times.append(end_time - start_time)
        csv_data.append([frame_number, f"frame_{frame_number}.jpg", caption, end_time - start_time])

        if VIDEO_SELECTION_MODE == "single" and DETAIL_MODE:
            print(f"Frame {frame_number}: {caption}")

        if SUMMARY_MODEL in ["gpt2", "llama"] and len(captions) > 1:
            prev_caption = captions[-2]
            curr_caption = captions[-1]
            transition = summarize_transition(prev_caption, curr_caption)
            transitions.append((prev_caption, curr_caption, transition))
            print(f"Transition: {transition}")

    mean_time = statistics.mean(frame_times)
    median_time = statistics.median(frame_times)
    mode_time = statistics.mode(frame_times) if len(frame_times) > 1 else frame_times[0]

    if transitions and SUMMARY_MODEL in ["gpt2", "llama"]:
        transition_sentences = [t[2] for t in transitions]
        final_summary, summary_time = summarize_captions(transition_sentences)
    else:
        final_summary, summary_time = summarize_captions(captions)

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    # Save detailed caption CSV
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

    # Save summary and metadata for this model
    summary_filename = os.path.join(summary_output_dir, f"{video_filename[:-4]}_summary.csv")
    with open(summary_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Summary Model", SUMMARY_MODEL])
        writer.writerow(["Total Captions", len(captions)])
        writer.writerow(["Summary Text", final_summary])
        writer.writerow(["Summary Processing Time (s)", summary_time])
        writer.writerow(["Total Video Processing Time (s)", total_processing_time])

    # Save raw captions if desired
    captions_txt_path = os.path.join(summary_output_dir, f"{video_filename[:-4]}_captions.txt")
    with open(captions_txt_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(captions))

    # Save transition file if any transitions were generated
    if transitions:
        transitions_csv_path = os.path.join(summary_output_dir, f"{video_filename[:-4]}_transitions.csv")
        with open(transitions_csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Previous Caption", "Current Caption", "Transition Summary"])
            writer.writerows(transitions)

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
