import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# Load BART tokenizer & model
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
bart_model = BartForConditionalGeneration.from_pretrained(model_name)

def encode_with_bart(text):
    """Encodes a sentence using BART and returns the full encoder output."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    with torch.no_grad():
        encoder_outputs = bart_model.model.encoder(**inputs)  # Returns an object, not a tensor
    return encoder_outputs  # Keep it as an object

def decode_with_bart(encoder_outputs):
    """Decodes the encoder hidden states back into natural language using BART."""
    # Ensure encoder outputs are passed correctly
    generated_ids = bart_model.generate(encoder_outputs=encoder_outputs, max_length=50)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Example usage
text1 = "A dog is walking on a path with red flowers."
text2 = "Man and dog on path with red flowers."

# Encode inputs
encoder_output1 = encode_with_bart(text1) 
encoder_output2 = encode_with_bart(text2) 

# Extract the last hidden states (tensor representations)
hidden_state1 = encoder_output1.last_hidden_state  # Tensor of shape (batch_size, seq_len, hidden_dim)
hidden_state2 = encoder_output2.last_hidden_state  # Tensor of shape (batch_size, seq_len, hidden_dim)

# Ensure they have the same shape
min_len = min(hidden_state1.shape[1], hidden_state2.shape[1])
hidden_state1 = hidden_state1[:, :min_len, :]
hidden_state2 = hidden_state2[:, :min_len, :]

# Compute difference
diff_vec = hidden_state1 - hidden_state2  # This is now a tensor

# Wrap the diff_vec tensor into a BaseModelOutput
diff_output = BaseModelOutput(last_hidden_state=diff_vec)

# Decode using BART
reconstructed_text = decode_with_bart(diff_output)

print("Original Text 1 Encoding Shape:", hidden_state1.shape)
print("Original Text 2 Encoding Shape:", hidden_state2.shape)
print("Difference Text:", reconstructed_text)
