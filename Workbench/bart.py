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
