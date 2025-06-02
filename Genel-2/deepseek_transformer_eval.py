<<<<<<< HEAD
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

# Part 1: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# Part 2: Multi-head Latent Attention (MLA)
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, latent_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K1 = nn.Linear(d_model, latent_dim)
        self.linear_V1 = nn.Linear(d_model, latent_dim)
        self.linear_K2 = nn.Linear(latent_dim, d_model)
        self.linear_V2 = nn.Linear(latent_dim, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.size()
        Q = self.linear_Q(x)
        K_latent = self.linear_K1(x)
        V_latent = self.linear_V1(x)
        K_full = self.linear_K2(K_latent)
        V_full = self.linear_V2(V_latent)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K_full.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V_full.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.to(scores.dtype)
            mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, mask_value)
        attn = self.softmax(scores)
        attn_output = torch.matmul(attn, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)
        output = self.out_linear(attn_output)
        return output

# Part 3: Causal Mask
def create_causal_mask(seq_len: int):
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

# Part 4: MoE Feed-Forward Block
class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        ) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)
        topk_vals, topk_indices = torch.topk(gate_probs, self.k, dim=-1)
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (topk_indices == i).float()
            gating_weight = (topk_vals * mask).sum(dim=-1, keepdim=True)
            expert_out = self.experts[i](x)
            output += expert_out * gating_weight
        return output

# Part 5: Transformer Block: DeepSeek-Inspired Block
class DeepSeekBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, latent_dim: int, d_ff: int,
                 moe_experts: int = 4, moe_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, num_heads, latent_dim)
        self.ffn = MoEFeedForward(d_model, d_ff, num_experts=moe_experts, k=moe_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

# Part 6: Transformer Model: DeepSeek-Inspired Transformer
class DeepSeekTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, latent_dim: int, d_ff: int,
                 num_layers: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([DeepSeekBlock(d_model, num_heads, latent_dim, d_ff, dropout=dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.fc_out(x)

# Part 7: WikiText-2 Dataset: Vocabulary Limitation
class WikiTextDataset(Dataset):
    def __init__(self, split: str = "train", seq_len: int = 50, vocab_size: int = 30000):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.seq_len = seq_len
        text = "\n".join(dataset["text"])
        tokens = text.split()
        counter = Counter(tokens)
        most_common = counter.most_common(vocab_size - 1)
        self.vocab = {word: idx+1 for idx, (word, _) in enumerate(most_common)}
        self.vocab["<mask>"] = 0
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        self.token_ids = [self.vocab.get(word, 0) for word in tokens]

    def __len__(self):
        return len(self.token_ids) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for shifted targets
        tokens = self.token_ids[start:end]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)  # Shifted targets
        return x, y

# Part 8: Training Loop
VOCAB_SIZE = 100000
D_MODEL = 512
NUM_HEADS = 8
LATENT_DIM = 64
D_FF = 512
NUM_LAYERS = 4
MAX_SEQ_LEN = 500
BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 0.001

train_dataset = WikiTextDataset(split="train", seq_len=MAX_SEQ_LEN, vocab_size=VOCAB_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = DeepSeekTransformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, LATENT_DIM, D_FF,
                            NUM_LAYERS, MAX_SEQ_LEN, dropout=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*NUM_EPOCHS)

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
    for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        # Create causal mask
        mask = create_causal_mask(batch_x.size(1)).to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(batch_x, mask)
            loss = criterion(logits.view(-1, VOCAB_SIZE), batch_y.view(-1))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {epoch_loss/len(train_loader):.4f}")

# Part 9: Text Generation
def generate_text(model, prompt, vocab, inv_vocab, max_len=50, temperature=0.7, top_k=40):
    model.eval()
    tokens = [vocab.get(word, 0) for word in prompt.split()]
    input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    generated = input_tensor

    with torch.no_grad():
        for _ in range(max_len - len(tokens)):  # Generate only up to max_len - input_len
            mask = create_causal_mask(generated.size(1)).to(device)
            logits = model(generated, mask)[:, -1, :] / temperature

            # Use top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1))

            generated = torch.cat((generated, next_token), dim=1)

    return ' '.join([inv_vocab.get(token.item(), "") for token in generated.squeeze() if token != 0])

# Adjusted parameters
prompt = "DeepSeek is an"
generated_text = generate_text(model, prompt, train_dataset.vocab, train_dataset.inv_vocab, max_len=100, temperature=1.0, top_k=40)
=======
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

# Part 1: Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# Part 2: Multi-head Latent Attention (MLA)
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, latent_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K1 = nn.Linear(d_model, latent_dim)
        self.linear_V1 = nn.Linear(d_model, latent_dim)
        self.linear_K2 = nn.Linear(latent_dim, d_model)
        self.linear_V2 = nn.Linear(latent_dim, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.size()
        Q = self.linear_Q(x)
        K_latent = self.linear_K1(x)
        V_latent = self.linear_V1(x)
        K_full = self.linear_K2(K_latent)
        V_full = self.linear_V2(V_latent)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K_full.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V_full.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.to(scores.dtype)
            mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, mask_value)
        attn = self.softmax(scores)
        attn_output = torch.matmul(attn, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)
        output = self.out_linear(attn_output)
        return output

# Part 3: Causal Mask
def create_causal_mask(seq_len: int):
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

# Part 4: MoE Feed-Forward Block
class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        ) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)
        topk_vals, topk_indices = torch.topk(gate_probs, self.k, dim=-1)
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (topk_indices == i).float()
            gating_weight = (topk_vals * mask).sum(dim=-1, keepdim=True)
            expert_out = self.experts[i](x)
            output += expert_out * gating_weight
        return output

# Part 5: Transformer Block: DeepSeek-Inspired Block
class DeepSeekBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, latent_dim: int, d_ff: int,
                 moe_experts: int = 4, moe_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, num_heads, latent_dim)
        self.ffn = MoEFeedForward(d_model, d_ff, num_experts=moe_experts, k=moe_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

# Part 6: Transformer Model: DeepSeek-Inspired Transformer
class DeepSeekTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, latent_dim: int, d_ff: int,
                 num_layers: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([DeepSeekBlock(d_model, num_heads, latent_dim, d_ff, dropout=dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.fc_out(x)

# Part 7: WikiText-2 Dataset: Vocabulary Limitation
class WikiTextDataset(Dataset):
    def __init__(self, split: str = "train", seq_len: int = 50, vocab_size: int = 30000):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.seq_len = seq_len
        text = "\n".join(dataset["text"])
        tokens = text.split()
        counter = Counter(tokens)
        most_common = counter.most_common(vocab_size - 1)
        self.vocab = {word: idx+1 for idx, (word, _) in enumerate(most_common)}
        self.vocab["<mask>"] = 0
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        self.token_ids = [self.vocab.get(word, 0) for word in tokens]

    def __len__(self):
        return len(self.token_ids) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for shifted targets
        tokens = self.token_ids[start:end]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)  # Shifted targets
        return x, y

# Part 8: Training Loop
VOCAB_SIZE = 100000
D_MODEL = 512
NUM_HEADS = 8
LATENT_DIM = 64
D_FF = 512
NUM_LAYERS = 4
MAX_SEQ_LEN = 500
BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 0.001

train_dataset = WikiTextDataset(split="train", seq_len=MAX_SEQ_LEN, vocab_size=VOCAB_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = DeepSeekTransformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, LATENT_DIM, D_FF,
                            NUM_LAYERS, MAX_SEQ_LEN, dropout=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*NUM_EPOCHS)

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
    for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        # Create causal mask
        mask = create_causal_mask(batch_x.size(1)).to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(batch_x, mask)
            loss = criterion(logits.view(-1, VOCAB_SIZE), batch_y.view(-1))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {epoch_loss/len(train_loader):.4f}")

# Part 9: Text Generation
def generate_text(model, prompt, vocab, inv_vocab, max_len=50, temperature=0.7, top_k=40):
    model.eval()
    tokens = [vocab.get(word, 0) for word in prompt.split()]
    input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    generated = input_tensor

    with torch.no_grad():
        for _ in range(max_len - len(tokens)):  # Generate only up to max_len - input_len
            mask = create_causal_mask(generated.size(1)).to(device)
            logits = model(generated, mask)[:, -1, :] / temperature

            # Use top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1))

            generated = torch.cat((generated, next_token), dim=1)

    return ' '.join([inv_vocab.get(token.item(), "") for token in generated.squeeze() if token != 0])

# Adjusted parameters
prompt = "DeepSeek is an"
generated_text = generate_text(model, prompt, train_dataset.vocab, train_dataset.inv_vocab, max_len=100, temperature=1.0, top_k=40)
>>>>>>> 6e953b09772621b8cc37bb192b05fdf7daad2d9a
print(generated_text)