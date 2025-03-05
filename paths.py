import os

#Paths for videos
VID_DIR = 'Benchmark Videos'
VIDEO_PATH = os.path.join(os.path.dirname(__file__), VID_DIR)

OUTPUT_DIR = 'Output'
OUTPUT_PATH = os.path.join(VIDEO_PATH, OUTPUT_DIR)

BLIP_DIR = 'BLIP'
OUTPUT_PATH = os.path.join(OUTPUT_PATH,BLIP_DIR)

BLIPID3_DIR = 'BLIPID3'
OUTPUT_PATH = os.path.join(OUTPUT_PATH, BLIPID3_DIR)

CLIP_DIR = 'CLIP'
OUTPUT_PATH = os.path.join(OUTPUT_PATH, CLIP_DIR)

ID3_DIR = 'ID3'
OUTPUT_PATH = os.path.join(OUTPUT_PATH, ID3_DIR)