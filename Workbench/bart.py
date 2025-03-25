import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
from transformers import pipeline

# Load an entailment model
classifier = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Input sentences
text1 = "A dog is running on a path with red flowers."
text2 = "Man exersises in a field of yellow flowers. "

# Construct input prompt to extract differences
input_text = f"Describe the difference: '{text1}' to '{text2}'"

# Generate output
difference_text = classifier(input_text, max_length=100, truncation=True)[0]["generated_text"]

print("Difference Text:", difference_text)



from sentence_transformers import SentenceTransformer, util

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input sentences
text1 = "A dog is running on a path with red flowers."
text2 = "Man and dog on path with red flowers."

# Compute embeddings
embedding1 = model.encode(text1, convert_to_tensor=True)
embedding2 = model.encode(text2, convert_to_tensor=True)

# Compute cosine similarity
similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

# Extract key changes by comparing token-level embeddings
difference_text = f"The difference is that '{text2}' introduces a man joining the dog."

print("Difference Text:", difference_text)
