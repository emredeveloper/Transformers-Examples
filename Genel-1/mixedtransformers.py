import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TextProcessor:
    def __init__(self, max_vocab=20000, max_len=128):
        self.vocab = {'<pad>': 0, '<unk>': 1}
        self.max_len = max_len

    def build_vocab(self, texts):
        counter = counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)
        vocab_list = ['<pad>', '<unk>'] + [word for word, _ in counter.most_common(20000)]
        self.vocab = {word: idx for idx, word in enumerate(vocab_list)}

    def text_to_indices(self, text):
        tokens = text.lower().split()[:self.max_len]
        return [self.vocab.get(token, 1) for token in tokens]

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_size=256, heads=8, window_size=3):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.window_size = window_size

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, S, _ = x.shape
        x = x.view(B, S, self.heads, self.head_dim)
        
        # Padding ekle
        pad_size = self.window_size
        padded_x = F.pad(x, (0,0,0,0, pad_size, pad_size))
        
        # Pencereleri oluştur
        windows = []
        for i in range(S):
            start = i
            end = start + 2*self.window_size + 1
            windows.append(padded_x[:, start:end])
        windows = torch.stack(windows, dim=1)
        
        # Attention hesapla
        Q = self.query(windows)
        K = self.key(windows)
        V = self.value(windows)
        
        energy = torch.einsum("bswhd,bswhd->bswh", Q, K) / (self.head_dim**0.5)
        attention = F.softmax(energy, dim=2)
        
        out = torch.einsum("bswh,bswhd->bshd", attention, V)
        out = out.reshape(B, S, self.embed_size)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size=256, heads=8):
        super().__init__()
        self.attention = SlidingWindowAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size, embed_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        attn = self.attention(x)
        x = self.norm1(x + attn)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x.mean(dim=1))

# Test
if __name__ == "__main__":
    texts = ["positive text", "negative text"]
    labels = [1, 0]
    
    processor = TextProcessor()
    processor.build_vocab(texts)
    
    dataset = TextDataset(texts, labels, processor)
    loader = DataLoader(dataset, batch_size=2)
    
    model = TextTransformer(len(processor.vocab))
    test_input = torch.randint(0, 1000, (2, 128))
    output = model(test_input)
    print("Output shape:", output.shape)  # Should be (2, 2)