import cv2
import torch
import os
import nltk
from collections import Counter
from paths import *  # Ensure this is correctly set
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Function to detect scenes using histogram differences
def detect_scenes(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    prev_hist = None
    scene_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if diff < threshold:  # Significant scene change
                scene_frames.append(frame)

        prev_hist = hist

    cap.release()
    return scene_frames

# Function to generate captions
def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(caption_ids[0], skip_special_tokens=True)

# Function to summarize captions
def summarize_captions(captions):
    words = nltk.word_tokenize(" ".join(captions))
    stopwords = set(nltk.corpus.stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stopwords]
    word_freq = Counter(filtered_words)

    important_words = [word for word, freq in word_freq.items() if freq > 1]

    # Use Hugging Face summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    summary = summarizer(" ".join(captions), max_length=30, min_length=15, do_sample=False)[0]["summary_text"]

    return summary

# Process video
video_path = os.path.join(VIDEO_PATH, '03.mp4')  # Ensure correct path
scene_frames = detect_scenes(video_path)

captions = []  # Store captions

# Generate captions only for key scenes
for i, frame in enumerate(scene_frames):
    caption = generate_caption(frame)
    captions.append(caption)

    # Resize frame to 25% of its original size
    height, width, _ = frame.shape
    resized_frame = cv2.resize(frame, (width // 4, height // 4))

    # Overlay caption on resized frame
    cv2.putText(resized_frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(f"Scene {i+1}", resized_frame)
    print(f"Scene {i+1}: {caption}")

    # Wait for key press to proceed to the next scene
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Summarize final captions
final_summary = summarize_captions(captions)
print("\n=== Final Video Summary ===")
print(final_summary)
