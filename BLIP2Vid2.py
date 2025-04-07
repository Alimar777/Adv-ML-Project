# BLIP2Vid2.py
import cv2
import torch
import os
import statistics
from paths import *
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils import *

# === Global Settings ===
VIDEO_SELECTION_MODE        = "single"
SELECTED_VIDEO_INDEX        = 3
DETAIL_MODE                 = True
SUMMARY_MODEL               = "majority"   # bart |  distilbart | gpt2 | tinyllama | majority
TRANSITION_MODEL            = "majority"       #  gpt2 | tinyllama | majority
TRANSITION_PROMPT_TEMPLATE  = (
    "Describe the visual change between the two scenes below in ONE "
    "concise sentence.\n\nPrevious: {prev}\nCurrent: {curr}\nTransition:"
)
GENERATE_GROUP_TRANSITIONS  = True


STORY_OUTPUT_MODE = "verbose"   #  story | verbose 




def load_summarizer(device, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    if model_name == "bart":
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
        return summarizer, None, None

    elif model_name == "distilbart":
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1)
        return summarizer, None, None

    elif model_name == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    elif model_name == "tinyllama":
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
    
    elif model_name == "majority":
        def summarizer_stub(text):
            return text.split("\n")[0]  # fallback stub
        return summarizer_stub, None, None
    else:
        raise ValueError("Unsupported SUMMARY_MODEL")

    def summarize_transformer(text):
        prompt = (
            "You are a summarizer. Follow these steps carefully.\n\n"
            "1. Read the following scene descriptions exactly as written.\n"
            "2. Identify only the nouns and adjectives present in the descriptions.\n"
            "3. Do NOT add anything that is not mentioned in the text.\n"
            "4. Combine the descriptions into one clear sentence.\n\n"
            "Captions:\n" + text + "\n\nSummary:"
        )

        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(model.device)
        attention_mask = torch.ones_like(inputs)
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=30,
            temperature=1,
            do_sample=False,  # Set to TRUE when modifying temperature != 1
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summarize_transformer, tokenizer, model

def summarize_captions(captions, summarizer, summary_model):
    import time
    start_summary_time = time.time()
    joined = "\n".join(captions)

    if summary_model in ["bart", "distilbart"]:
        prompt = (
            "Summarize the following descriptions into a single coherent scene:\n"
            + joined
        )
        input_len = len(prompt.split())
        max_len = min(100, max(10, int(input_len * 1.2)))
        summary = summarizer(prompt, max_length=max_len, min_length=6, do_sample=False)[0]["summary_text"]
    else:
        # Using one of the GPT2/LLama/tinyLlama approaches
        summary = summarizer(joined)

    end_summary_time = time.time()
    return summary, end_summary_time - start_summary_time

def summarize_transition(prev_caption, curr_caption,
                         summarizer_fn, model_choice,
                         prompt_template=TRANSITION_PROMPT_TEMPLATE):
    """Return a one‑sentence bridge between two scene captions."""
    # Fast path for the 'majority' stub
    if model_choice == "majority":
        return f"The scene shifts from “{prev_caption}” to “{curr_caption}.”"

    prompt = prompt_template.format(prev=prev_caption, curr=curr_caption)

    if model_choice in ["bart", "distilbart"]:
        max_len = min(60, max(10, int(len(prompt.split()) * 1.2)))
        return summarizer_fn(prompt,
                             max_length=max_len,
                             min_length=6,
                             do_sample=False)[0]["summary_text"].strip()

    # gpt‑style models (gpt2, llama, tinyllama …)
    result = summarizer_fn(prompt)
    if "Transition:" in result:
        result = result.split("Transition:")[-1]
    return result.strip().split(".")[0] + "."

def output_story_set(group_summaries, group_transitions, mode=STORY_OUTPUT_MODE):
    """
    Print the narrative in either:
      • 'story'    first group summary, then every bridge on its own line
      • 'verbose'  group summary, then its bridge, each on separate lines
    """
    if not group_summaries:
        print("ERROR:  No groups to narrate.")
        return

    if mode.lower() == "story":
        print(f"\n{MAGENTA}=== Final Story ==={RESET}")
        print(group_summaries[0])                     # first scene
        for _, _, bridge in group_transitions:        # each bridge
            print(bridge)

    elif mode.lower() == "verbose":
        print(f"\n{MAGENTA}=== Verbose Story ==={RESET}")
        for i, summary in enumerate(group_summaries):
            print(summary)                            # group summary
            if i < len(group_transitions):            # its bridge (if any)
                print(group_transitions[i][2])
    else:
        raise ValueError("STORY_OUTPUT_MODE must be 'story' or 'verbose'.")



def summarize_group(group, summarizer, summary_model):
    """
    Return a one‑sentence description of an entire caption group.
    Falls back to the group's only caption if it’s a singleton.
    """
    if len(group) == 1:
        return group[0]
    summary, _ = summarize_captions(group, summarizer, summary_model)
    return summary.strip()


def compare_summarizers(group_text, bart_summarizer, llama_summarizer, group_idx, distilbart_summarizer=None):
    if SUMMARY_MODEL == "majority":
        summary = get_most_frequent_caption(group_text)
        print(f"{CYAN}Group {group_idx+1} Majority Summary:{RESET} {summary} (Size: {len(group_text)})")
        return
    
    joined = "\n".join(group_text)
    input_len = len(joined.split())
    max_len = min(100, max(10, int(input_len * 1.2)))

    bart_result = bart_summarizer(joined, max_length=max_len, min_length=6, do_sample=False)[0]["summary_text"]

    if distilbart_summarizer:
        distilbart_result = distilbart_summarizer(joined, max_length=max_len, min_length=6, do_sample=False)[0]["summary_text"]
    else:
        distilbart_result = None

    llama_result = llama_summarizer(joined)

    print(f"\n{YELLOW}Summarizing Group {group_idx+1}:{RESET}")
    print(f"Majority: {get_most_frequent_caption(group_text)}")
    print(f"{CYAN}BART Summary:{RESET}  {bart_result}")
    if distilbart_result:
        print(f"{DARK_GREEN}DistilBART Summary:{RESET} {distilbart_result}")
    print(f"{MAGENTA}LLAMA Summary:{RESET} {llama_result}")


# Load BART-CNN summarizer (for comparison)
from transformers import pipeline as hf_pipeline
bart_summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
distilbart_summarizer = hf_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1)


def main():
    """
    Main video-processing workflow:
    1) Extract frames from each video at a chosen interval.
    2) Generate a caption for each frame using BLIP.
    3) Check semantic similarity between consecutive captions on-the-fly:
       - If difference is large, we mark a "significant change" and start a new group.
    4) (Optionally) Summarize transitions if using GPT2/LLama.
    5) Summarize all captions at the end.
    6) Save results, plus optional cluster visualizations.
    """

    global summarizer, tokenizer, model
    global trans_summarizer, trans_tokenizer, trans_model

    # Set up device, BLIP processor/model, summarizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    # Caption / group summariser
    summarizer, tokenizer, model = load_summarizer(device, SUMMARY_MODEL)

    # Transition generator
    trans_summarizer, trans_tokenizer, trans_model = load_summarizer(device, TRANSITION_MODEL)


    # 1) Model for on‐the‐fly semantic checks
    from sentence_transformers import SentenceTransformer, util
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Adjust this threshold if you want more or fewer "significant changes"
    similarity_threshold = 0.61

    interval = 1
    BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "BLIP")
    os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

    # Whether to do all videos or just a single chosen one
    video_indices = range(1, 11) if VIDEO_SELECTION_MODE == "all" else [SELECTED_VIDEO_INDEX]

    for i in video_indices:
        video_filename = f"{i:02d}.mp4"
        video_path = os.path.join(VIDEO_PATH, video_filename)
        if not os.path.exists(video_path):
            print(f"{RED}Video file {video_filename} not found. Skipping...{RESET}")
            continue

        # Prepare output directories
        video_output_dir = os.path.join(BLIP_OUTPUT_DIR, video_filename[:-4])
        os.makedirs(video_output_dir, exist_ok=True)
        summary_output_dir = os.path.join(video_output_dir, "Summaries", SUMMARY_MODEL)
        os.makedirs(summary_output_dir, exist_ok=True)

        print(f"{CYAN}\nProcessing {video_filename}...{RESET}\n")

        # Extract frames
        frames, total_frames, video_duration = extract_frames(video_path, interval)

        # Placeholders for results
        captions, csv_data, frame_times = [], [], []
        transitions = []

        # 2) Track caption groups
        grouped_captions = []
        current_group = []
        last_embedding = None

        import time
        total_start_time = time.time()

        for frame_number, frame in frames:
            start_time = time.time()
            # Generate a BLIP caption for this frame
            caption = generate_caption(frame, processor, blip_model, device)
            end_time = time.time()

            captions.append(caption)
            frame_times.append(end_time - start_time)
            csv_data.append([frame_number, f"frame_{frame_number}.jpg", caption, end_time - start_time])

            if VIDEO_SELECTION_MODE == "single" and DETAIL_MODE:
                print(f"{DARK_GREEN}Frame {frame_number}:{RESET} {caption}")

            # 3) Compute new caption embedding
            new_embedding = embedding_model.encode(caption, convert_to_tensor=True)

            # 4) Compare similarity with the previous embedding
            if last_embedding is not None:
                sim_score = util.pytorch_cos_sim(new_embedding, last_embedding).item()
                if sim_score < similarity_threshold:
                    print(f"{YELLOW}Significant change detected at frame {frame_number}!"
                          f" Similarity={sim_score:.2f}{RESET}")
                    # Close out the old group
                    grouped_captions.append(current_group)
                    current_group = []

            # Add the current caption to our active group
            current_group.append(caption)
            last_embedding = new_embedding

            # If you want transitions for GPT2/LLama
            if SUMMARY_MODEL in ["gpt2", "llama"] and len(captions) > 1:
                transition = summarize_transition(captions[-2], captions[-1])
                transitions.append((captions[-2], captions[-1], transition))
                print_transition(transition)

        # End of frames — close out the last group
        if current_group:
            grouped_captions.append(current_group)

        print(f"\n{MAGENTA}=== Grouped Caption Sets (Semantic Clusters) ==={RESET}")
        for idx, group in enumerate(grouped_captions):
            print(f"\n{CYAN}Group {idx+1} ({len(group)} captions):{RESET}")
            for caption in group:
                print(f"  - {caption}")

            if len(group) > 1:
                unique_captions = set(group)
                if len(unique_captions) == 1:
                    repeated_caption = unique_captions.pop()
                    print(f"{DARK_GREEN}Group {idx+1} is uniform. Skipping summarization. Returning repeated caption:{RESET} {repeated_caption}")
                else:
                    compare_summarizers(group, bart_summarizer, summarizer, idx, distilbart_summarizer)
            else:
                print(f"{YELLOW}Skipping summarization for Group {idx+1} (only one caption).{RESET}")

        
        group_summaries = [
            summarize_group(g, summarizer, SUMMARY_MODEL) for g in grouped_captions
        ]

        # Optionally build transitions BETWEEN those summaries
        group_transitions = []
        if GENERATE_GROUP_TRANSITIONS and len(group_summaries) > 1:
            for prev, curr in zip(group_summaries[:-1], group_summaries[1:]):
                if TRANSITION_MODEL in ["gpt2", "tinyllama"]:
                    bridge = summarize_transition(prev, curr,trans_summarizer, TRANSITION_MODEL)

                else:
                    # Fallback: plain template so we don't need a generative model
                    bridge = f"The scene shifts from “{prev}” to “{curr}.”"
                group_transitions.append((prev, curr, bridge))

            # Pretty‑print
            print(f"\n{MAGENTA}=== Group-level Transitions ==={RESET}")
            for i, (_, _, t) in enumerate(group_transitions, 1):
                print(f"{i}. {t}")

        if group_transitions:
            save_transitions_csv(
                summary_output_dir,
                video_filename.replace(".mp4", "_group"),
                group_transitions
            )

        # Summarization logic
        mean_time = statistics.mean(frame_times)
        median_time = statistics.median(frame_times)
        mode_time = statistics.mode(frame_times) if len(frame_times) > 1 else frame_times[0]

        # If we have transitions from GPT2/LLama summarization:
        if transitions and SUMMARY_MODEL in ["gpt2", "llama"]:
            # For demonstration, building a "final summary" by combining transitions
            final_summary = " ".join(sorted(set(t[2] for t in transitions)))
            summary_time = 0.0
        else:
            # Otherwise use your summarizer for a single final summary of *all* captions
            final_summary, summary_time = summarize_captions(captions, summarizer, SUMMARY_MODEL)


        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time

        total_stats = {
            "Total Frames": total_frames,
            "Video Duration (s)": video_duration,
            "Mean Frame Processing Time": mean_time,
            "Median Frame Processing Time": median_time,
            "Mode Frame Processing Time": mode_time,
            "Summary Processing Time": summary_time,
            "Total Processing Time": total_processing_time
        }

        # Save results and print summary
        save_results(video_filename, video_output_dir, csv_data, total_stats)
        save_summary_csv(summary_output_dir, video_filename, SUMMARY_MODEL, final_summary,
                         summary_time, total_processing_time, len(captions))
        save_captions_txt(summary_output_dir, video_filename, captions)
        if transitions:
            save_transitions_csv(summary_output_dir, video_filename, transitions)
            print_unique_transitions(transitions)

        print_video_summary(final_summary, total_stats)

        output_story_set(group_summaries, group_transitions, STORY_OUTPUT_MODE)

        # Condense all the final captions into clusters (existing function in utils)
        cluster_map = generate_cluster_map(captions, threshold = similarity_threshold)

        # Visualize cluster similarity (optional)
        visualize_clusters(captions, cluster_map, summary_output_dir, video_filename, method="both",threshold=similarity_threshold)
        animate_caption_clusters(captions, cluster_map, summary_output_dir, video_filename, method='pca')
        animate_caption_clusters(captions, cluster_map, summary_output_dir, video_filename, method='tsne', save_frames = True)

if __name__ == "__main__":
    main()
