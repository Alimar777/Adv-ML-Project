import os
import time
import statistics
from config import *
from utils import extract_frames, generate_caption, summarize_captions, rephrase_summary, save_results, print_summary

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

    frames, total_frames, video_duration = extract_frames(video_path, FRAME_INTERVAL)
    captions, csv_data, frame_times = [], [], []
    total_start_time = time.time()

    for frame_number, frame in frames:
        start_time = time.time()
        caption = generate_caption(frame, processor, model, device)
        end_time = time.time()

        captions.append(caption)
        frame_times.append(end_time - start_time)
        csv_data.append([frame_number, f"frame_{frame_number}.jpg", caption, end_time - start_time])

    mean_time = statistics.mean(frame_times)
    median_time = statistics.median(frame_times)
    mode_time = statistics.mode(frame_times) if len(frame_times) > 1 else frame_times[0]

    # Summarize and rephrase captions
    summary = summarize_captions(captions, summarizer)
    final_summary = rephrase_summary(summary, rephraser)
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    total_stats = {
        "Total Frames": total_frames,
        "Video Duration (s)": video_duration,
        "Mean Processing Time (s)": mean_time,
        "Median Processing Time (s)": median_time,
        "Mode Processing Time (s)": mode_time,
        "Total Processing Time (s)": total_processing_time,
        "Summary": final_summary
    }

    save_results(video_filename, video_output_dir, csv_data, total_stats)
    print_summary(final_summary, total_stats)
