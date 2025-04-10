import torch
from transformers import pipeline

def main():
    # Choose your Flan-T5 variant
    model_name = "google/flan-t5-large"
    
    # Create a pipeline for text2text-generation
    flan_t5 = pipeline(
        "text2text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # More direct and clearer prompt formatting
    PROMPT = """
    Compare these two scenes and identify only the differences:
    Scene 1: a dog walking down a dirt road with red flowers
    Scene 2: a black cat walking down a dirt path with red flowers
    
    List only what changed between the scenes in one short sentence:
    """
    
    # Generate the response with adjusted parameters
    response = flan_t5(
        PROMPT,
        max_length=100,      # Increased to give model more space
        min_length=10,       # Set minimum to ensure substantive response
        do_sample=True,      # Enable sampling for more creative responses
        temperature=0.7,     # Add temperature for more diverse outputs
        top_p=0.9,           # Add top_p sampling
        num_return_sequences=10,  # Generate multiple options to choose from
    )
    
    # Print the model's outputs
    print("\n=== Prompt ===")
    print(PROMPT)
    print("\n=== Flan-T5 Outputs ===")
    for i, res in enumerate(response):
        print(f"Option {i+1}: {res['generated_text'].strip()}")

if __name__ == "__main__":
    main()