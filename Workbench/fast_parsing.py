import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Sample captions (semantically similar)
captions = [
    "a man walking down a dirt path with red flowers",
    "a man walking down a dirt path with a backpack",
    "a man walking down a dirt path with a backpack",
    "a person walking down a dirt path with red flowers",
    "a person walking down a dirt path with red flowers",
    "a man walking down a dirt path with a backpack",
    "a man walking down a dirt path with a backpack",
    "a person walking down a dirt path with red flowers",
    "a man walking down a dirt path with red flowers",
    "a person walking down a dirt road with red flowers",
    "a person walking down a dirt road with red flowers",
    "a person walking down a dirt path with red flowers"
]

print("\n=== Sample Captions ===")
for cap in captions:
    print(f"- {cap}")

# === Method 1: Fast Merge with spaCy ===
nlp = spacy.load("en_core_web_sm")

def fast_merge_summary(captions):
    docs = [nlp(c) for c in captions]
    noun_chunks = set()
    verbs = set()
    adjectives = set()

    for doc in docs:
        for chunk in doc.noun_chunks:
            noun_chunks.add(chunk.text.lower())
        for token in doc:
            if token.pos_ == "VERB":
                verbs.add(token.lemma_)
            elif token.pos_ == "ADJ":
                adjectives.add(token.text.lower())

    noun_phrase = " ".join(sorted(noun_chunks))
    verb_phrase = " and then ".join(sorted(verbs))

    return f"{verb_phrase.capitalize()} {noun_phrase}".strip().replace("  ", " ")

# === Method 2: Centroid + Closest Caption ===
st_model = SentenceTransformer("all-MiniLM-L6-v2")

def centroid_caption_summary(captions):
    embeddings = st_model.encode(captions, convert_to_tensor=True)
    centroid = torch.mean(embeddings, dim=0)
    scores = util.pytorch_cos_sim(centroid, embeddings)[0]
    best_idx = torch.argmax(scores).item()
    return captions[best_idx]

# === Method 3: BART Summarization ===
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_with_bart(captions):
    joined = "\n".join(captions)
    result = bart_summarizer(joined, max_length=100, min_length=15, do_sample=False)[0]["summary_text"]
    return result

# === Optional: TinyLLAMA (if desired) ===
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

def summarize_with_llama(captions):
    prompt = (
        "In one sentence, summarize the following scene descriptions from a video.\n"
        "Only include what is explicitly described. Avoid repetition.\n\n"
        "Captions:\n" + "\n".join(captions) + "\n\nSummary:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Run & Print All Summaries ===
print("\n\n=== 🔍 Summarizer Comparison ===")

print(f"\n🧠 spaCy Merge Summary:\n{fast_merge_summary(captions)}")

print(f"\n🎯 Centroid + Nearest Summary:\n{centroid_caption_summary(captions)}")

print(f"\n📰 BART Summary:\n{summarize_with_bart(captions)}")

print(f"\n🐑 TinyLlama Summary:\n{summarize_with_llama(captions)}")
