from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import random
import os




# Load COCO 2014 captions dataset
dataset = load_dataset("shunk031/MSCOCO", name="2014", coco_task="captions", split="train[:1%]", trust_remote_code=True)


# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Helper: caption an image
def get_blip_caption(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Generate scene pairs and differences
scene1_list, scene2_list, diff_list = [], [], []
for _ in range(10):  # Generate 10 examples
    i1, i2 = random.sample(range(len(dataset)), 2)
    
    # The image key might be different depending on the dataset structure
    if "image" in dataset[i1]:
        img1 = dataset[i1]["image"]
        img2 = dataset[i2]["image"]
    elif "pixel_values" in dataset[i1]:
        img1 = dataset[i1]["pixel_values"]
        img2 = dataset[i2]["pixel_values"]
    else:
        # Print available keys to help debug
        print(f"Available keys in dataset: {dataset[i1].keys()}")
        raise ValueError("Can't find image data in the dataset")
    
    cap1 = get_blip_caption(img1)
    cap2 = get_blip_caption(img2)
    
    # Naive difference summary (later can be improved using embedding diffs or templates)
    diff = f"{cap1} vs {cap2}"
    scene1_list.append(cap1)
    scene2_list.append(cap2)
    diff_list.append(diff)

# Print samples
for s1, s2, d in zip(scene1_list, scene2_list, diff_list):
    print("Scene 1:", s1)
    print("Scene 2:", s2)
    print("Difference:", d)
    print("-" * 60)