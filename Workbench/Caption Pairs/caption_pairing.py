import os
import csv
import glob
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === Ensure working directory is the script's location ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# === Settings ===
CAPTIONS_DIR = os.path.join(SCRIPT_DIR, "captions")  # CSVs like 01_captions.csv
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "transitions_dataset.csv")
SIMILARITY_THRESHOLD = 0.75  # tweak as needed

# === Load model ===
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_transition_pairs(captions, video_label):
    transitions = []
    for i in range(len(captions) - 1):
        prev_caption = captions[i]
        curr_caption = captions[i + 1]
        sim = util.pytorch_cos_sim(
            model.encode(prev_caption, convert_to_tensor=True),
            model.encode(curr_caption, convert_to_tensor=True)
        ).item()
        if sim < SIMILARITY_THRESHOLD:
            transitions.append({
                "video": video_label,
                "prev_caption": prev_caption,
                "curr_caption": curr_caption,
                "human_transition": ""
            })
    return transitions

def main():
    all_transitions = []

    for csv_file in sorted(glob.glob(os.path.join(CAPTIONS_DIR, "*_captions.csv"))):
        df = pd.read_csv(csv_file)
        if "Caption" not in df.columns:
            print(f"Skipping {csv_file}: no 'Caption' column found.")
            continue

        captions = df["Caption"].dropna().tolist()
        video_label = os.path.basename(csv_file).split("_")[0]
        transitions = get_transition_pairs(captions, video_label)
        all_transitions.extend(transitions)

    if all_transitions:
        df_out = pd.DataFrame(all_transitions)
        df_out.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Saved {len(df_out)} transitions to: {OUTPUT_PATH}")
    else:
        print("⚠️ No transition pairs found.")

if __name__ == "__main__":
    main()
