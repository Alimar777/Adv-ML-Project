import torch
from transformers import AutoTokenizer, AutoModel, BartTokenizer, BartForConditionalGeneration

# ----- Load Models & Tokenizers -----
# BERT for embeddings
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name)
bert_model.eval()

# BART for decoding
bart_model_name = "facebook/bart-large"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# ----- Define Functions -----
def get_bert_embedding(text):
    """Obtain the [CLS] token embedding for a given text using BERT."""
    inputs = bert_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the first token ([CLS]) embedding
    return outputs.last_hidden_state[0, 0, :]

def encode_with_bart(text):
    """Encode text with BART and return the encoder's hidden states."""
    inputs = bart_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        encoder_outputs = bart_model.model.encoder(**inputs)
    return encoder_outputs.last_hidden_state

def decode_with_bart(encoder_hidden_states):
    """Decode encoder hidden states to text using BART."""
    generated_ids = bart_model.generate(encoder_outputs=encoder_hidden_states, max_length=50)
    return bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main(text1, text2, print_mode='none'):
    # Compute embeddings for both texts using BERT
    emb1 = get_bert_embedding(text1)
    emb2 = get_bert_embedding(text2)
    diff_vector = emb2 - emb1

    # Conditional printing based on mode
    if print_mode == 'full':
        print("Embedding for text1:\n", emb1)
        print("\nEmbedding for text2:\n", emb2)
        print("\nDifference vector:\n", diff_vector)
    elif print_mode == 'simplified':
        # Here we simply print the mean difference as a simple summary
        print("Simplified difference vector (mean value):", diff_vector.mean().item())
    else:
        print("No embedding/difference vector printed.")

    # --- Apply the difference vector ---
    # For illustration, we add the mean of the difference to BART's encoding of text1.
    bart_encoding = encode_with_bart(text1)
    # NOTE: Adding a scalar (mean value) to every hidden state element is a naive approach.
    modified_encoding = bart_encoding + diff_vector.mean()
    decoded_text = decode_with_bart(modified_encoding)

    print("\nDecoded sentence from modified embedding:")
    print(decoded_text)

# ----- Run Example -----
if __name__ == "__main__":
    text1 = "Dog on path with red flowers"
    text2 = "Man and dog on path with red flowers"
    main(text1, text2, print_mode='full')
