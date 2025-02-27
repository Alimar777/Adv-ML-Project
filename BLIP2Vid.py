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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

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
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

# Function to generate captions
def generate_caption(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    caption_ids = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(caption_ids[0], skip_special_tokens=True)

# Function to summarize captions
def summarize_captions(captions):
    # Tokenize words and count occurrences
    words = nltk.word_tokenize(" ".join(captions))
    word_freq = Counter(words)

    # Remove common words (stopwords)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stopwords]

    # Find key themes based on word frequency
    important_words = [word for word, freq in word_freq.items() if freq > 1 and word.lower() not in stopwords]
    
    # Use Hugging Face summarization model (optional)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(" ".join(captions), max_length=50, min_length=15, do_sample=False)[0]["summary_text"]

    return summary

# Process video
video_path = os.path.join(VIDEO_PATH, '01.mp4')  # Ensure correct path
frames = extract_frames(video_path, interval)

captions = []  # Store captions

# Generate captions and display resized frames
for i, frame in enumerate(frames):
    caption = generate_caption(frame)
    captions.append(caption)

    # Resize frame to 25% of its original size
    height, width, _ = frame.shape
    resized_frame = cv2.resize(frame, (width // 4, height // 4))

    # Display the resized frame with caption overlay
    #cv2.putText(resized_frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    #cv2.imshow(f"Frame {i+1}", resized_frame)
    print(f"Frame {i+1}: {caption}")

    # Wait for key press to proceed to the next frame (press any key to continue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # Close the previous frame window

# Summarize captions to avoid redundancy
final_summary = summarize_captions(captions)
print("\n=== Final Video Summary ===")
print(final_summary)
