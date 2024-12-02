from transformers import AutoTokenizer

# Load tokenizer for the embedding model
tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")

def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

# Example usage
text = "Your input text here."
print(f"Number of tokens: {count_tokens(text)}")