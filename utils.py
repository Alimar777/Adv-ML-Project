import os
import time
import csv
import statistics
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2
import torch
import nltk
from config import *

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull


import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.patches as mpatches

from scipy.spatial import ConvexHull, QhullError
from matplotlib.patches import Ellipse
import numpy as np

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# === ANSI Color Codes ===
DARK_GREEN = "\033[38;2;0;128;0m"
CYAN = "\033[38;2;0;255;255m"
YELLOW = "\033[38;2;255;255;0m"
RED = "\033[38;2;255;0;0m"
MAGENTA = "\033[38;2;255;0;255m"
RESET = "\033[0m"

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

def extract_frames(video_path, interval=1):
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

def generate_caption(image, processor, blip_model, device):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    generated_ids = blip_model.generate(pixel_values=inputs["pixel_values"], max_new_tokens=60)
    return processor.decode(generated_ids[0], skip_special_tokens=True)


def condense_captions(captions, summarizer, threshold=0.85):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(captions, convert_to_tensor=True)
    clusters = []
    cluster_map = {}
    cluster_captions = defaultdict(list)

    for i, caption in enumerate(captions):
        assigned = False
        for cid, cluster_embedding in enumerate([embeddings[j[0]] for j in clusters]):
            sim = util.pytorch_cos_sim(embeddings[i], cluster_embedding).item()
            if sim > threshold:
                clusters[cid].append(i)
                cluster_map[i] = cid
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            cluster_map[i] = len(clusters) - 1

    condensed_output = []
    for cid, indices in enumerate(clusters):
        grouped_captions = [captions[i] for i in indices]
        duration = len(indices)
        summary = summarizer("\n".join(grouped_captions))
        condensed_output.append(f"{summary.strip()} ({duration}s)")

    return condensed_output, cluster_map

def save_condensed_txt(summary_output_dir, video_filename, condensed):
    output_path = os.path.join(summary_output_dir, f"{video_filename[:-4]}_condensed.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(condensed))


def visualize_clusters(captions, cluster_map, output_dir, video_filename, method="both", cluster_labels=None, threshold = 0.85):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(captions)
    cluster_ids = [cluster_map[i] for i in range(len(captions))]
    cluster_ids = np.array(cluster_ids)

    # Auto-generate cluster labels if none provided
    if cluster_labels is None:
        cluster_caption_map = defaultdict(list)
        for idx, cid in enumerate(cluster_ids):
            cluster_caption_map[cid].append(captions[idx])
        
        cluster_labels = {}
        for cid, cluster_caps in cluster_caption_map.items():
            most_common_caption, _ = Counter(cluster_caps).most_common(1)[0]
            cluster_labels[cid] = most_common_caption

    def _plot(reduced, title_prefix, suffix_prefix, is_pca=True):
        threshold_str = f"th{int(threshold * 100)}"
        title_prefix = f"{title_prefix} (Threshold: {threshold:.2f})"
        suffix_prefix = f"{suffix_prefix}_{threshold_str}"

        unique_clusters = sorted(set(cluster_ids))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        color_map = {cid: colors[i] for i, cid in enumerate(unique_clusters)}
        legend_handles = [mpatches.Patch(color=color_map[cid], label=cluster_labels[cid]) for cid in unique_clusters]

        # 1. Simple Scatter (no enhancements)
        fig_scatter, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        for i, cid in enumerate(unique_clusters):
            points = reduced[cluster_ids == cid]
            ax.scatter(points[:, 0], points[:, 1], s=100, color=color_map[cid], edgecolors='k', alpha=0.9)
        ax.set_title(f"{title_prefix} - Raw Scatter", fontsize=16)
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        fig_scatter.savefig(os.path.join(output_dir, f"{video_filename[:-4]}_{suffix_prefix}_raw.png"))
        plt.close(fig_scatter)

        # 2. With Convex Hulls & Lines
        fig_hull, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        for i, cid in enumerate(unique_clusters):
            points = reduced[cluster_ids == cid]
            color = color_map[cid]
            ax.scatter(points[:, 0], points[:, 1], s=100, color=color, edgecolors='k', alpha=0.9)

            num_unique = len(np.unique(points, axis=0))
            if len(points) >= 3 and num_unique >= 3:
                try:
                    hull = ConvexHull(points, qhull_options='QJ')
                    hull_points = points[hull.vertices]
                    polygon = plt.Polygon(hull_points, closed=True, fill=True, alpha=0.2, color=color)
                    ax.add_patch(polygon)
                except QhullError as e:
                    print(f"Warning: Could not compute hull for cluster {cid}: {e}")
            elif len(points) == 2:
                p1, p2 = points
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, alpha=0.4)
            elif len(points) == 1:
                ax.scatter(points[0][0], points[0][1], s=150, color=color, edgecolors='k', alpha=0.8)
        ax.set_title(f"{title_prefix} - Hulls + Lines", fontsize=16)
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        fig_hull.savefig(os.path.join(output_dir, f"{video_filename[:-4]}_{suffix_prefix}_hull_lines.png"))
        plt.close(fig_hull)

        # 3. Full Transition View (includes arrows between sequential captions)
        fig_full, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        for i, cid in enumerate(unique_clusters):
            points = reduced[cluster_ids == cid]
            color = color_map[cid]
            ax.scatter(points[:, 0], points[:, 1], s=100, color=color, edgecolors='k', alpha=0.9)

            num_unique = len(np.unique(points, axis=0))
            if len(points) >= 3 and num_unique >= 3:
                try:
                    hull = ConvexHull(points, qhull_options='QJ')
                    hull_points = points[hull.vertices]
                    polygon = plt.Polygon(hull_points, closed=True, fill=True, alpha=0.2, color=color)
                    ax.add_patch(polygon)
                except QhullError as e:
                    print(f"Warning: Could not compute hull for cluster {cid}: {e}")
            elif len(points) == 2:
                p1, p2 = points
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, alpha=0.4)
            elif len(points) == 1:
                ax.scatter(points[0][0], points[0][1], s=150, color=color, edgecolors='k', alpha=0.8)

        # Draw caption transition arrows (colored by cluster they start from)
        for i in range(1, len(reduced)):
            start = reduced[i - 1]
            end = reduced[i]
            cid = cluster_ids[i - 1]
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color_map[cid], lw=1.5, alpha=0.6)
            )
            mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, str(i), fontsize=8, color='black', alpha=0.6)

        ax.set_title(f"{title_prefix} - With Arrows", fontsize=16)
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        fig_full.savefig(os.path.join(output_dir, f"{video_filename[:-4]}_{suffix_prefix}_arrows.png"))
        plt.close(fig_full)

        # Save legend once only
        fig_legend = plt.figure(figsize=(6, len(legend_handles) * 0.5))
        plt.legend(handles=legend_handles, loc='center', fontsize=12, framealpha=0.95)
        plt.axis('off')
        fig_legend.tight_layout()
        fig_legend.savefig(os.path.join(output_dir, f"{video_filename[:-4]}_{suffix_prefix}_legend.png"))
        plt.close(fig_legend)

    # Run for PCA
    if method in ["pca", "both"]:
        reduced_pca = PCA(n_components=2).fit_transform(embeddings)
        _plot(reduced_pca, "Caption Clusters (PCA)", "cluster_plot_pca", is_pca=True)

    # Run for t-SNE
    if method in ["tsne", "both"]:
        reduced_tsne = TSNE(n_components=2, perplexity=5, learning_rate=100, init='random', random_state=42).fit_transform(embeddings)
        _plot(reduced_tsne, "Caption Clusters (t-SNE)", "cluster_plot_tsne", is_pca=False)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def animate_caption_clusters(captions, cluster_map, output_path, video_filename, method='pca'):
    assert method in ['pca', 'tsne'], "method must be 'pca' or 'tsne'"

    from matplotlib.animation import FuncAnimation, PillowWriter
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(captions)
    cluster_ids = np.array([cluster_map[i] for i in range(len(captions))])

    # Choose dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "Caption Timeline Animation (PCA)"
        suffix = "_timeline_pca.gif"
    else:
        reducer = TSNE(n_components=2, perplexity=5, learning_rate=100, init='random', random_state=42)
        title = "Caption Timeline Animation (t-SNE)"
        suffix = "_timeline_tsne.gif"

    reduced = reducer.fit_transform(embeddings)

    # Expanded axis limits
    x_range = reduced[:, 0].max() - reduced[:, 0].min()
    y_range = reduced[:, 1].max() - reduced[:, 1].min()
    padding_x = x_range * 0.15
    padding_y = y_range * 0.15
    x_min, x_max = reduced[:, 0].min() - padding_x, reduced[:, 0].max() + padding_x
    y_min, y_max = reduced[:, 1].min() - padding_y, reduced[:, 1].max() + padding_y

    unique_clusters = sorted(set(cluster_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cid: colors[i] for i, cid in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    drawn_artists = []

    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        return []

    def update(frame):
        while drawn_artists:
            artist = drawn_artists.pop()
            artist.remove()

        # Trail: draw all previous transitions with faded arrows
        for i in range(frame):
            start = reduced[i]
            end = reduced[i + 1]
            cid = cluster_ids[i]
            arrow = ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle='->', color=color_map[cid], lw=1, alpha=0.3))
            drawn_artists.append(arrow)

        # Current points
        current_points = reduced[:frame + 1]
        current_clusters = cluster_ids[:frame + 1]
        current_colors = [color_map[cid] for cid in current_clusters]
        scatter = ax.scatter(current_points[:, 0], current_points[:, 1], s=100, c=current_colors, edgecolors='k', alpha=0.9)
        drawn_artists.append(scatter)

        # Draw current arrow
        if frame > 0:
            start = reduced[frame - 1]
            end = reduced[frame]
            cid = cluster_ids[frame - 1]
            arrow = ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle='->', color=color_map[cid], lw=2, alpha=0.8))
            drawn_artists.append(arrow)

        x, y = reduced[frame]
        cid = cluster_ids[frame]

        # Cluster label: offset from arrow tip
        cluster_label = ax.text(x + 0.8, y + 0.8, f"Cluster {cid}",
                                fontsize=10, color='black', fontweight='bold', alpha=0.30)
        drawn_artists.append(cluster_label)

        # Frame number: centered on arrow
        if frame > 0:
            start = reduced[frame - 1]
            end = reduced[frame]
            mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            label = ax.text(mid_x, mid_y, str(frame),
                            fontsize=11, color='black', alpha=0.60, fontweight='bold')
            drawn_artists.append(label)

        # Progress indicator
        progress = ax.text(0.02, 0.95, f"Frame {frame+1}/{len(captions)}",
                           transform=ax.transAxes, fontsize=10, color='gray')
        drawn_artists.append(progress)

        return drawn_artists

    ani = FuncAnimation(fig, update, frames=len(captions), init_func=init, blit=False, repeat=False)

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f"{video_filename[:-4]}{suffix}")
    ani.save(save_path, dpi=100, writer=PillowWriter(fps=1))
    print(f"✅ {method.upper()} animation saved to: {save_path}")




def save_results(video_filename, video_output_dir, csv_data, total_stats):
    csv_filename = os.path.join(video_output_dir, f"{video_filename[:-4]}_captions.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Image File", "Caption", "Processing Time (s)"])
        writer.writerows(csv_data)
        writer.writerow([])
        writer.writerow(["Video Stats"])
        for key, value in total_stats.items():
            writer.writerow([key, value])

def save_summary_csv(summary_output_dir, video_filename, summary_model, final_summary, summary_time, total_processing_time, num_captions):
    summary_filename = os.path.join(summary_output_dir, f"{video_filename[:-4]}_summary.csv")
    with open(summary_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Summary Model", summary_model])
        writer.writerow(["Total Captions", num_captions])
        writer.writerow(["Summary Text", final_summary])
        writer.writerow(["Summary Processing Time (s)", summary_time])
        writer.writerow(["Total Video Processing Time (s)", total_processing_time])

def save_captions_txt(summary_output_dir, video_filename, captions):
    captions_txt_path = os.path.join(summary_output_dir, f"{video_filename[:-4]}_captions.txt")
    with open(captions_txt_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(captions))

def save_transitions_csv(summary_output_dir, video_filename, transitions):
    transitions_csv_path = os.path.join(summary_output_dir, f"{video_filename[:-4]}_transitions.csv")
    with open(transitions_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Previous Caption", "Current Caption", "Transition Summary"])
        writer.writerows(transitions)


def print_summary(final_summary, total_stats):
    print("\n=== Final Scene Summary ===")
    print(final_summary)
    print("\n=== Processing Time Stats ===")
    for key, value in total_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def print_transition(transition):
    print(f"{YELLOW}Transition:{RESET} {transition}")

def print_unique_transitions(transitions):
    print(f"\n{MAGENTA}=== All Unique Transitions ==={RESET}")
    for unique in sorted(set(t[2] for t in transitions)):
        print(f"- {unique}")

def print_video_summary(final_summary, total_stats):
    print(f"\n{MAGENTA}=== Final Video Summary ==={RESET}")
    print(f"{final_summary}")

    print(f"\n{MAGENTA}=== Processing Time Stats ==={RESET}")
    for key, value in total_stats.items():
        if isinstance(value, float):
            print(f"{CYAN}{key}:{RESET} {value:.4f} seconds")
        else:
            print(f"{CYAN}{key}:{RESET} {value}")
