import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from typing import List


# Sample texts
texts = [
    "Hello, how are you?",
    "The weather is great today.",
    "The Python programming language is very popular.",
    "Deep learning is a branch of artificial intelligence.",
    "Python is an important tool for text processing."
]



class TurkishTokenizer:
    def __init__(self):
        self.vocab = {}  # Token -> ID
        self.id_to_token = {}  # ID -> Token
        self.next_id = 0  # Next token ID
        self.unk_token = "<UNK>"  # Unknown token placeholder
        self.pad_token = "<PAD>"  # Padding token
        self.special_tokens = [self.unk_token, self.pad_token]

        # Add special tokens to the vocabulary
        for token in self.special_tokens:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """
        Add a new token to the vocabulary and assign an ID.
        """
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]

    def tokenize(self, text: str) -> List[int]:
        """
        Split text into tokens and return their IDs.
        """
        # Lowercase text and isolate punctuation
        text = text.lower()
        tokens = re.findall(r"\w+|\S", text)  # Words and punctuation marks
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.unk_token])  # Unknown token fallback
        return token_ids

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.unk_token)
        return " ".join(tokens)

    def build_vocab(self, texts: List[str]):
        """
        Build a vocabulary from the provided texts.
        """
        counter = Counter()
        for text in texts:
            text = text.lower()
            tokens = re.findall(r"\w+|\S", text)
            counter.update(tokens)

        # Add the most frequent tokens
        for token, _ in counter.most_common():
            self.add_token(token)
            
            
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.tokenize(text)
        return torch.tensor(token_ids, dtype=torch.long)

# Build the tokenizer and dataset
tokenizer = TurkishTokenizer()
tokenizer.build_vocab(texts)  # Populate the vocabulary

dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Test the tokenizer
test_text = "The weather is great today."
token_ids = tokenizer.tokenize(test_text)
print(f"Token IDs: {token_ids}")

# Convert token IDs back to text
decoded_text = tokenizer.detokenize(token_ids)
print(f"Decoded text: {decoded_text}")

print(dataset)
# Iterate through DataLoader samples
for batch in dataloader:
    print("Batch:", batch)