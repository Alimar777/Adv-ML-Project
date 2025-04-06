import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

captions = [...]  # <- Your full caption list here

# Encode and cluster
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(captions, convert_to_tensor=True)

threshold = 0.85
clusters, cluster_map = [], {}
for i, emb in enumerate(embeddings):
    found = False
    for cid, indices in enumerate(clusters):
        sim = util.pytorch_cos_sim(emb, embeddings[indices[0]]).item()
        if sim > threshold:
            clusters[cid].append(i)
            cluster_map[i] = cid
            found = True
            break
    if not found:
        clusters.append([i])
        cluster_map[i] = len(clusters) - 1

# Reduce with PCA/TSNE
pca_coords = PCA(n_components=2).fit_transform(embeddings)
tsne_coords = TSNE(n_components=2, perplexity=5, learning_rate=100, init='random', random_state=42).fit_transform(embeddings)

# Node CSV
nodes = []
for i, caption in enumerate(captions):
    nodes.append({
        "Id": i,
        "Label": i,
        "Timestamp": i,
        "caption": caption,
        "cluster": cluster_map[i]
    })
node_df = pd.DataFrame(nodes)

# Edge CSV (timeline + TSNE cluster cliques)
edges = []
# Timeline edges
for i in range(len(captions) - 1):
    edges.append({"Source": i, "Target": i+1, "Type": "Directed", "Id": f"{i}->{i+1}"})
# TSNE intra-cluster edges
for group in clusters:
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            edges.append({"Source": group[i], "Target": group[j], "Type": "Directed", "Id": f"{group[i]}->{group[j]}"})
edge_df = pd.DataFrame(edges)

# Save CSVs
output_dir = "gephi_export"
os.makedirs(output_dir, exist_ok=True)
node_df.to_csv(os.path.join(output_dir, "nodes.csv"), index=False)
edge_df.to_csv(os.path.join(output_dir, "edges.csv"), index=False)
print(f"✅ CSVs saved to: {output_dir}/nodes.csv and edges.csv")
