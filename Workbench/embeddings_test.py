import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch.nn.functional as F

class EmbeddingTransitionDescriber:
    def __init__(self, 
                 embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                 language_model_name='gpt2-medium'):
        """
        Initialize the transition describer with embedding and language models.
        
        :param embedding_model_name: Model for creating sentence embeddings
        :param language_model_name: Model for generating natural language descriptions
        """
        # Embedding model
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        
        # Language generation model
        self.lang_tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.lang_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        
    def get_embedding(self, sentence):
        """
        Generate embedding for a given sentence.
        
        :param sentence: Input sentence to embed
        :return: Embedding vector
        """
        # Tokenize the sentence
        tokens = self.embedding_tokenizer(sentence, return_tensors='pt', 
                                          padding=True, truncation=True)
        
        # Generate embedding
        with torch.no_grad():
            model_output = self.embedding_model(**tokens)
        
        # Mean pooling
        embeddings = model_output.last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = torch.sum(embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        
        return embeddings.squeeze()
    
    def calculate_transition_vector(self, embedding1, embedding2):
        """
        Calculate the transition vector between two embeddings.
        
        :param embedding1: First embedding
        :param embedding2: Second embedding
        :return: Transition vector
        """
        return embedding2 - embedding1
    
    def generate_transition_description(self, sentence1, sentence2, 
        generation_prompt="Describe the change from the first scene to the second scene: "):
        """
        Generate a natural language description of the embedding transition.
        
        :param sentence1: First input sentence
        :param sentence2: Second input sentence
        :param generation_prompt: Prompt for language model
        :return: Generated description of the transition
        """
        # Get embeddings
        embedding1 = self.get_embedding(sentence1)
        embedding2 = self.get_embedding(sentence2)
        
        # Calculate transition vector
        transition_vector = self.calculate_transition_vector(embedding1, embedding2)
        
        # Prepare input for language model
        input_text = (f"{generation_prompt}\n"
                      f"Scene 1: {sentence1}\n"
                      f"Scene 2: {sentence2}\n"
                      "Description of change: ")
        
        # Tokenize input
        input_ids = self.lang_tokenizer.encode(input_text, return_tensors='pt')
        
        # Generate description
        output = self.lang_model.generate(
            input_ids, 
            max_length=input_ids.shape[1] + 50,  # Add some extra tokens for generation
            num_return_sequences=1,
            temperature=0.7
        )
        
        # Decode the generated text
        generated_text = self.lang_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract the transition description
        transition_description = generated_text.split("Description of change: ")[-1].strip()
        
        return {
            'embedding1': embedding1,
            'embedding2': embedding2,
            'transition_vector': transition_vector,
            'transition_description': transition_description
        }

# Example usage
def main():
    # Create the transition describer
    describer = EmbeddingTransitionDescriber()
    
    # Example sentences
    sentence1 = "Dog on path with red flowers"
    sentence2 = "Man and dog on path with red flowers"
    
    # Analyze the transition
    result = describer.generate_transition_description(sentence1, sentence2)
    
    # Print results
    print("Sentence 1:", sentence1)
    print("Sentence 2:", sentence2)
    print("\nTransition Description:", result['transition_description'])
    print("\nTransition Vector Shape:", result['transition_vector'].shape)

if __name__ == "__main__":
    main()