from typing import List

class SimpleTokenizer:
    """A minimal tokenizer that maps whitespace-separated tokens to IDs and back."""

    def __init__(self):
        """Initialise the tokenizer using whitespace tokenisation."""
        self.vocab = {}  # Token to ID mapping
        self.id_to_token = {}  # Reverse lookup from ID to token
        self.next_id = 0  # Next available token ID

    def add_token(self, token: str) -> int:
        """Register a new token and return its ID."""
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]

    def tokenize(self, text: str) -> List[int]:
        """Split text into tokens and return their IDs."""
        tokens = text.split()  # Split on whitespace
        token_ids = []
        for token in tokens:
            token_id = self.add_token(token)  # Add token and retrieve its ID
            token_ids.append(token_id)
        return token_ids

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back into a whitespace-separated string."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "")  # Look up the token for each ID
            tokens.append(token)
        return " ".join(tokens)  # Join tokens back into text

# Example usage
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # Tokenise text
    text = "Hello world! This is a sample sentence."
    token_ids = tokenizer.tokenize(text)
    print(f"Token IDs: {token_ids}")

    # Convert IDs back into text
    decoded_text = tokenizer.detokenize(token_ids)
    print(f"Decoded text: {decoded_text}")
