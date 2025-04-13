import cv2
import torch
import os
import statistics
from paths import *
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils import *
from bridgeBART_builder import run_bridgebart_training_pipeline
from dotenv import load_dotenv
load_dotenv()

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")




# === Global Settings ===
VIDEO_SELECTION_MODE        = "all" 
SELECTED_VIDEO_INDEX        = 3
DETAIL_MODE                 = True

# Summaries at the group/overall level:
SUMMARY_MODEL               = "majority"   # bart | distilbart | gpt2 | tinyllama | flan-t5 | majority

# Transitions between group summaries:
TRANSITION_MODEL            = "bridgeBART"    # gpt2 | tinyllama | majority | flan-t5 | bridgeBART

# More explicit instructive prompt for transitions:
TRANSITION_PROMPT_TEMPLATE = """Caption 1: {prev}
Caption 2: {curr}
What's the transition?"""


GENERATE_GROUP_TRANSITIONS  = True
STORY_OUTPUT_MODE           = "verbose"   # story | verbose

FINAL_SUMMARIZER_MODELS = ["bart", "distilbart", "flan-t5", "gpt-3.5", "gpt-4"]

LLM_SWITCH = True

def load_summarizer(device, model_name):
    """
    Return a pipeline (or function) for summarizing text,
    plus optional tokenizer/model references for generative models.

    For flan-t5, we switch to a "text2text-generation" pipeline
    to better handle instruction-like prompts.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    # Bart
    if model_name == "bart":
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        # This pipeline returns outputs in out[0]["summary_text"]
        return summarizer, None, None

    # DistilBart
    elif model_name == "distilbart":
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=0 if torch.cuda.is_available() else -1
        )
        # This pipeline returns out[0]["summary_text"]
        return summarizer, None, None

    # Flan‑T5: use text2text-generation so we get "generated_text"
    elif model_name == "flan-t5":
        summarizer = pipeline(
            "text2text-generation",  # <-- NOTE: changed from "summarization"
            model="google/flan-t5-base",  # or "google/flan-t5-large" if you have enough VRAM
            device=0 if torch.cuda.is_available() else -1
        )
        # This pipeline returns out[0]["generated_text"]
        return summarizer, None, None

    # GPT-2
    elif model_name == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # TinyLlama
    elif model_name == "tinyllama":
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)

    # 'majority' fallback
    elif model_name == "majority":
        def summarizer_stub(text):
            # If 'majority', just return the first line or a quick majority approach
            return text.split("\n")[0]
        return summarizer_stub, None, None
    
    elif model_name == "bridgeBART":
        from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "Models", "bridgeBART")

        summarizer = pipeline(
            "text2text-generation",  # treat as a T5-style generation
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"Loaded bridgeBART model from {model_path}")
        return summarizer, None, None


    else:
        raise ValueError("Unsupported SUMMARY_MODEL")

    def summarize_transformer(text):
        """
        GPT2/TinyLlama approach: create a prompt, call model.generate.
        """
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
            do_sample=False,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summarize_transformer, tokenizer, model


def summarize_captions(captions, summarizer, summary_model):
    """
    Summarize a list of captions into a single short summary.
    Distinguish whether we're using a pipeline that returns
    "summary_text" vs. "generated_text".
    """
    import time
    start_summary_time = time.time()
    joined = "\n".join(captions)

    if summary_model in ["bart", "distilbart"]:
        # Summarization pipeline -> "summary_text"
        prompt = (
            "Summarize the following descriptions into a single coherent scene:\n"
            + joined
        )
        input_len = len(prompt.split())
        max_len = min(100, max(10, int(input_len * 1.2)))
        output = summarizer(prompt, max_length=max_len, min_length=6, do_sample=False)
        summary = output[0]["summary_text"]

    elif summary_model == "flan-t5":
        # text2text-generation pipeline -> "generated_text"
        prompt = (
            "Summarize the following descriptions into a single coherent scene:\n"
            + joined
        )
        input_len = len(prompt.split())
        max_len = min(100, max(10, int(input_len * 1.2)))
        output = summarizer(prompt, max_length=max_len, min_length=6, do_sample=False)
        summary = output[0]["generated_text"]

    else:
        # Using GPT2/TinyLlama or 'majority'
        summary = summarizer(joined)

    end_summary_time = time.time()
    return summary, (end_summary_time - start_summary_time)


def summarize_transition(prev_caption, curr_caption,
                         summarizer_fn, model_choice,
                         prompt_template=TRANSITION_PROMPT_TEMPLATE):
    """
    Return a one-sentence "bridge" describing the change from prev_caption to curr_caption.
    If using a pipeline, we check whether it's "summarization" or "text2text-generation".
    """
    if model_choice == "majority":
        # fallback for 'majority'
        print(f"Mjority")
        return f"The scene shifts from “{prev_caption}” to “{curr_caption}.”"

    prompt = prompt_template.format(prev=prev_caption, curr=curr_caption)

    if model_choice in ["bart", "distilbart"]:
        max_len = min(60, max(10, int(len(prompt.split()) * 1.2)))
        output = summarizer_fn(prompt, max_length=max_len, min_length=6, do_sample=False)
        return output[0]["summary_text"].strip()

    elif model_choice in ["flan-t5", "bridgeBART"]:
        max_len = min(60, max(10, int(len(prompt.split()) * 1.2)))
        output = summarizer_fn(prompt, max_length=max_len, min_length=6, do_sample=False)
        return output[0]["generated_text"].strip()


    else:
        # GPT2, TinyLlama, or a custom function
        result = summarizer_fn(prompt)
        # If there's a 'Transition:' substring, strip it out
        if "Transition:" in result:
            result = result.split("Transition:")[-1]
        # Return the first sentence
        return result.strip().split(".")[0] + "."


def output_story_set(group_summaries, group_transitions, mode=STORY_OUTPUT_MODE, grouped_captions=None):
    """
    Print the narrative in either:
    - 'story': first group summary, then each transition
    - 'verbose': includes:
        1. Original BLIP captions (in order)
        2. Original BLIP captions (grouped) — already printed elsewhere
        3. Group summaries only
        4. First summary + transitions
        5. Summary + transition interleaved per group
    """
    if not group_summaries:
        print("ERROR: No groups to narrate.")
        return

    if mode.lower() == "story":
        print(f"\n{MAGENTA}=== Final Story ==={RESET}")
        print(group_summaries[0])
        for _, _, bridge in group_transitions:
            print(bridge)

    elif mode.lower() == "verbose":
        print(f"\n{MAGENTA}=== Verbose Story ==={RESET}")

        # 1. All original BLIP captions in order
        print(f"\n{CYAN}-- Original BLIP Captions (In Order) --{RESET}")
        if grouped_captions:
            all_captions = [cap for group in grouped_captions for cap in group]
            for i, cap in enumerate(all_captions, 1):
                print(f"{i:02d}. {cap}")

        # 2. Grouped captions already printed earlier in main()

        # 3. Group summaries only
        print(f"\n{CYAN}-- Group Summaries --{RESET}")
        for i, summary in enumerate(group_summaries, 1):
            print(f"Group {i}: {summary}")

        # 4. First summary + all transitions
        print(f"\n{CYAN}-- Summary + Transitions --{RESET}")
        if group_summaries:
            print(f"Summary 1: {group_summaries[0]}")
            for j, (_, _, bridge) in enumerate(group_transitions, 1):
                print(f"Transition {j}: {bridge}")

        # 5. Interleaved format (original verbose style)
        print(f"\n{CYAN}-- Summary + Transition for Each Group --{RESET}")
        for i, summary in enumerate(group_summaries):
            print(f"Group {i+1} Summary: {summary}")
            if i < len(group_transitions):
                print(f"Group {i+1} → {i+2} Transition: {group_transitions[i][2]}")

    else:
        raise ValueError("STORY_OUTPUT_MODE must be 'story' or 'verbose'.")



def summarize_group(group, summarizer, summary_model):
    """
    Return a one-sentence description of an entire caption group.
    Falls back to the single caption if there's only one.
    """
    if len(group) == 1:
        return group[0]

    if summary_model == "majority":
        return get_most_frequent_caption(group)

    summary, _ = summarize_captions(group, summarizer, summary_model)
    return summary.strip()



def compare_summarizers(group_text, bart_summarizer, llama_summarizer, group_idx, distilbart_summarizer=None):
    """
    For demonstration: show how bart, distilbart, or llama summarize the same text.
    If SUMMARY_MODEL == 'majority', we just do a majority approach.
    """
    if SUMMARY_MODEL == "majority":
        summary = get_most_frequent_caption(group_text)
        print(f"{CYAN}Group {group_idx+1} Majority Summary:{RESET} {summary} (Size: {len(group_text)})")
        return

    joined = "\n".join(group_text)
    input_len = len(joined.split())
    max_len = min(100, max(10, int(input_len * 1.2)))

    # BART
    bart_result = bart_summarizer(joined, max_length=max_len, min_length=6, do_sample=False)[0]["summary_text"]

    # DistilBART
    if distilbart_summarizer:
        distilbart_result = distilbart_summarizer(joined, max_length=max_len, min_length=6, do_sample=False)[0]["summary_text"]
    else:
        distilbart_result = None

    # "llama_summarizer" is presumably a custom function or pipeline
    llama_result = llama_summarizer(joined)

    print(f"\n{YELLOW}Summarizing Group {group_idx+1}:{RESET}")
    print(f"Majority: {get_most_frequent_caption(group_text)}")
    print(f"{CYAN}BART Summary:{RESET}  {bart_result}")
    if distilbart_result:
        print(f"{DARK_GREEN}DistilBART Summary:{RESET} {distilbart_result}")
    print(f"{MAGENTA}LLAMA Summary:{RESET} {llama_result}")


# For demonstration with compare_summarizers, we load a BART and DistilBart pipeline:
from transformers import pipeline as hf_pipeline
bart_summarizer = hf_pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if torch.cuda.is_available() else -1
)
distilbart_summarizer = hf_pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=0 if torch.cuda.is_available() else -1
)
def get_perplexity(sentence):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "gpt2" 
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity, loss
def get_final_summary(prompt, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Given a prompt and model name, run summarization using local or OpenAI models.
    """
    if model_name in ["gpt-3.5", "gpt-4"]:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes narratives."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo" if model_name == "gpt-3.5" else "gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        description = response.choices[0].message.content.strip()
        perplexity, loss = get_perplexity(description)
        return description, perplexity, loss

    else:
        summarizer_fn, _, _ = load_summarizer(device, model_name)

        if model_name in ["bart", "distilbart"]:
            output = summarizer_fn(prompt, max_length=80, min_length=10, do_sample=False)
            description = output[0]["summary_text"].strip()

            perplexity, loss = get_perplexity(description)
            return description, perplexity, loss
        elif model_name in ["flan-t5", "bridgeBART"]:
            output = summarizer_fn(prompt, max_length=80, min_length=10, do_sample=False)
            description = output[0]["generated_text"].strip()
            perplexity, loss = get_perplexity(description)
            return description, perplexity, loss
        else:
            output = summarizer_fn(prompt)
            description = output.strip().split(".")[0] + "."
            perplexity, loss = get_perplexity(description)
            return description, perplexity, loss


def sklearn_metrics(scores, threshold):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    pred_labels = [1 if score >= threshold else 0 for score in scores]
    true_labels = [1] * len(pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return precision, accuracy, recall, f1


def calculate_similarity_transformer(predicted_label, actual_caption): #captures context, semantic
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  
    embedding1 = model.encode(predicted_label, convert_to_tensor=True)
    embedding2 = model.encode(actual_caption, convert_to_tensor=True)
    similarity_score = util.cos_sim(embedding1, embedding2)
    print(f"Similarity Score: {similarity_score.item():.4f}")
    return similarity_score.item()


def main():
    """
    Main workflow:
    1) Extract frames at intervals.
    2) Generate frame-level captions using BLIP.
    3) Check semantic similarity to group frames.
    4) Summarize transitions if using a generative model for transitions.
    5) Summarize all captions at the end.
    6) Save results + optional cluster visualizations.
    """
    run_bridgebart_training_pipeline(force_train=False)

    global summarizer, tokenizer, model
    global trans_summarizer, trans_tokenizer, trans_model
    print("In main")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # For group-level or final summary
    summarizer, tokenizer, model = load_summarizer(device, SUMMARY_MODEL)

    # For transitions between group summaries
    trans_summarizer, trans_tokenizer, trans_model = load_summarizer(device, TRANSITION_MODEL)

    from sentence_transformers import SentenceTransformer, util
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    similarity_threshold = 0.61
    interval = 1

    BLIP_OUTPUT_DIR = os.path.join(VIDEO_PATH, "Output", "BLIP")
    os.makedirs(BLIP_OUTPUT_DIR, exist_ok=True)

    video_indices = range(1, 11) if VIDEO_SELECTION_MODE == "all" else [SELECTED_VIDEO_INDEX]
    all_perplexities = []
    all_losses = []
    all_summaries = []
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

        # Placeholders
        captions, csv_data, frame_times = [], [], []
        transitions = []

        grouped_captions = []
        current_group = []
        last_embedding = None

        import time
        total_start_time = time.time()

        for frame_number, frame in frames:
            start_time = time.time()
            # BLIP caption
            caption = generate_caption(frame, processor, blip_model, device)
            end_time = time.time()

            captions.append(caption) #blip
            frame_times.append(end_time - start_time)
            csv_data.append([frame_number, f"frame_{frame_number}.jpg", caption, end_time - start_time])

            if VIDEO_SELECTION_MODE == "single" and DETAIL_MODE:
                print(f"{DARK_GREEN}Frame {frame_number}:{RESET} {caption}")

            # Compute similarity
            new_embedding = embedding_model.encode(caption, convert_to_tensor=True)
            if last_embedding is not None:
                sim_score = util.pytorch_cos_sim(new_embedding, last_embedding).item()
                if sim_score < similarity_threshold:
                    print(f"{YELLOW}Significant change detected at frame {frame_number}! Similarity={sim_score:.2f}{RESET}")
                    grouped_captions.append(current_group)
                    current_group = []
            current_group.append(caption)
            last_embedding = new_embedding


        if current_group:#blip
            grouped_captions.append(current_group)

        print(f"\n{MAGENTA}=== Grouped Caption Sets (Semantic Clusters) ==={RESET}")
        for idx, group in enumerate(grouped_captions):
            print(f"\n{CYAN}Group {idx+1} ({len(group)} captions):{RESET}")
            for cap in group:
                print(f"  - {cap}")

            # Summaries for group
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

        # === Build transitions BETWEEN group summaries ===
        group_transitions = []
        if GENERATE_GROUP_TRANSITIONS and len(group_summaries) > 1:
            for prev, curr in zip(group_summaries[:-1], group_summaries[1:]):
                # If TRANSITION_MODEL is generative, call summarize_transition
                if TRANSITION_MODEL in ["gpt2", "tinyllama", "flan-t5", "bart", "distilbart", "bridgeBART"]:
                    print(f" Transition model {TRANSITION_MODEL}")
                    bridge = summarize_transition(prev, curr, trans_summarizer, TRANSITION_MODEL)
                else:
                    # "majority" fallback
                    bridge = f"The scene shifts from “{prev}” to “{curr}.”"
                group_transitions.append((prev, curr, bridge))

            print(f"\n{MAGENTA}=== Group-level Transitions ==={RESET}")
            for i, (_, _, t) in enumerate(group_transitions, 1):
                print(f"{i}. {t}")

            if group_transitions:
                save_transitions_csv(
                    summary_output_dir,
                    video_filename.replace(".mp4", "_group"),
                    group_transitions
                )

        # Summarize entire set
        mean_time = statistics.mean(frame_times)
        median_time = statistics.median(frame_times)
        mode_time = statistics.mode(frame_times) if len(frame_times) > 1 else frame_times[0]

        # If we have transitions from GPT2/LLama summarization:
        if transitions and SUMMARY_MODEL in ["gpt2", "llama"]:
            final_summary = " ".join(sorted(set(t[2] for t in transitions)))
            summary_time = 0.0
        else:
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

        # Save results, print final summary
        save_results(video_filename, video_output_dir, csv_data, total_stats)
        save_summary_csv(
            summary_output_dir, video_filename, SUMMARY_MODEL, final_summary,
            summary_time, total_processing_time, len(captions)
        )
        save_captions_txt(summary_output_dir, video_filename, captions)
        if transitions:
            save_transitions_csv(summary_output_dir, video_filename, transitions)
            print_unique_transitions(transitions)

        print_video_summary(final_summary, total_stats)

        # Print out the story with group transitions
        output_story_set(group_summaries, group_transitions, STORY_OUTPUT_MODE, grouped_captions)
        
        if LLM_SWITCH:
            # Create text versions of each story variant
            original_blip_text = "\n".join([cap for group in grouped_captions for cap in group])
            group_summary_text = "\n".join(group_summaries)
            first_plus_transitions = group_summaries[0] + "\n" + "\n".join(t[2] for t in group_transitions)

            full_narrative = ""
            for i, summary in enumerate(group_summaries):
                full_narrative += f"Summary {i+1}: {summary}\n"
                if i < len(group_transitions):
                    full_narrative += f"Transition {i+1}: {group_transitions[i][2]}\n"

            story_versions = {
                "Original BLIP Captions": original_blip_text,
                "Group Summaries Only": group_summary_text,
                "First Summary + Transitions": first_plus_transitions,
                "Full Interleaved Summary + Transitions": full_narrative
            }
########################################################################################################
            # Run final summarization

            perplexities = []
            losses = []
            model_names = []
            summaries = []
            for model_name in FINAL_SUMMARIZER_MODELS:
                print(f"\n{MAGENTA}=== Final Summaries using {model_name} ==={RESET}")
                for label, text in story_versions.items():
                    prompt = f"Summarize the following narrative in one concise paragraph:\n\n{text.strip()}"
                    try:
                        summary, perplexity, loss = get_final_summary(prompt, model_name)
                        if label == "Full Interleaved Summary + Transitions":
                            print(f"Perplexity {perplexity}, and loss {loss} of {model_name}")
                            perplexities.append(perplexity)
                            model_names.append(model_name)
                            losses.append(loss)
                            summaries.append(summary)
                        print(f"\n{CYAN}{label}:{RESET}\n{summary}")
                    except Exception as e:
                        print(f"\n{RED}Failed to summarize with {model_name} for {label}: {e}{RESET}")


        # Optionally cluster final captions
        cluster_map = generate_cluster_map(captions, threshold=similarity_threshold)
        visualize_clusters(captions, cluster_map, summary_output_dir, video_filename,
                           method="both", threshold=similarity_threshold)
        animate_caption_clusters(captions, cluster_map, summary_output_dir, video_filename, method='pca')
        animate_caption_clusters(captions, cluster_map, summary_output_dir, video_filename, method='tsne', save_frames=True)
    
        #list of lists for each model on each vid
        all_perplexities.append(perplexities)
        all_losses.append(losses)
        all_summaries.append(summaries)
    perplexity_avgs = []
    loss_avgs = []
    model_summaries = []
    for i, name in enumerate(model_names):
        model_perplexity = [y[i] for y in all_perplexities]
        model_loss = [y[i] for y in all_losses]
        print(f"avg per{model_perplexity}")
        avg_perplexity =np.mean([p.detach().cpu().numpy() for p in model_perplexity])
        avg_loss = np.mean([p.detach().cpu().numpy() for p in model_loss])
        model_summary =  [y[i] for y in all_summaries]
        print(f"For Model {name}: Perplex: {avg_perplexity}, Loss: {avg_loss}\n Summary: {model_summary}")
        perplexity_avgs.append(avg_perplexity)
        loss_avgs.append(avg_loss)
        model_summaries.append(model_summary)
    #Metrics
    captions = ["A woman makes a flower display.", "A person flips through a book.",
        "A dog walks down path with red flowers and a man follows.", 
        "A horse runs in dirt corral.", "People walk in indian fish market.",
            "A person buys produce in a market.", 
        "A person plays piano.",
            "People in work in an office.", 
            "People ice skate in front of building.",
            "A man in a red shirt kicks soccar ball in a feild.", 
        "A woman scans items in a lab. "]
    avg_scores = []
    avg_accuracy = []
    threshold = 0.3
    for j, model in enumerate(model_summaries):
        scores = []
        for i, pred in enumerate(model):
            print(f"For model {model_names[j]}: Actual:> {captions[i]}\n Predicitons: {pred}")
            score = calculate_similarity_transformer(pred, captions[i])
            scores.append(score)
        print(f"{model} Metrics:")
        mean_similarity = np.mean(scores)
        avg_scores.append(mean_similarity)
        print(f"Average Similarity Score: {mean_similarity:.4f}")
        precision, accuracy, recall,f1 = sklearn_metrics(scores, threshold) 
        avg_accuracy.append(accuracy)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, avg_accuracy, color='skyblue', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', fontsize=12)

    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, avg_scores, color='skyblue', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', fontsize=12)

    plt.title('Model Cosine Similarity Comparison', fontsize=16)
    plt.ylabel('Cosine Similarity', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, perplexity_avgs, color='skyblue', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', ha='center', fontsize=12)

    plt.title('Model Fluency Comparison', fontsize=16)
    plt.ylabel('Fluency Difficulty', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()
if __name__ == "__main__":
    main()
