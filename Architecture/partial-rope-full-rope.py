import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

class PartialRoPE(nn.Module):
    """Partial RoPE implementation that rotates only a fraction of dimensions."""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, partial_rotary_factor=0.5):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor
        
        # Use only the fraction defined by the partial factor
        self.rotary_dim = int(self.dim * self.partial_rotary_factor)
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[2]
        
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        # Apply RoPE only to the rotary portion
        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]
        
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # Embed only the rotary portion with RoPE
        q_rot_embed = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (self._rotate_half(k_rot) * sin)
        
        # Concatenate rotary and passthrough sections
        q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
        
        return q_embed, k_embed


class FullRoPE(nn.Module):
    """Full RoPE implementation."""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[2]
        
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


class AttentionWithRoPE(nn.Module):
    """Attention layer that uses RoPE."""
    def __init__(self, dim, num_heads, rope_module):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.rope = rope_module
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # Merge the heads back together
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_output)
        
        return output


class SimpleTransformerBlock(nn.Module):
    """Simple Transformer block."""
    def __init__(self, dim, num_heads, rope_module, mlp_ratio=4):
        super().__init__()
        self.attention = AttentionWithRoPE(dim, num_heads, rope_module)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    
    def forward(self, x, mask=None):
        # Attention + residual
        x = x + self.attention(self.norm1(x), mask)
        # MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


class LanguageModel(nn.Module):
    """Simple language model."""
    def __init__(self, vocab_size, dim, num_heads, num_layers, rope_module):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(dim, num_heads, rope_module)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
    
    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


def create_causal_mask(seq_len, device):
    """Create a causal mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0


def train_step(model, data, labels, optimizer, device):
    """Single training step."""
    model.train()
    data, labels = data.to(device), labels.to(device)
    
    # Build the causal mask
    seq_len = data.shape[1]
    mask = create_causal_mask(seq_len, device)
    
    # Forward pass
    logits = model(data, mask)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_perplexity(model, data, labels, device):
    """Compute perplexity."""
    model.eval()
    data, labels = data.to(device), labels.to(device)
    
    with torch.no_grad():
        seq_len = data.shape[1]
        mask = create_causal_mask(seq_len, device)
        logits = model(data, mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss)
    
    return perplexity.item()


def benchmark_rope_performance():
    """Compare the performance of partial and full RoPE."""

    # Sample Turkish texts used for training data
    turkish_texts = [
        "Hello world! The weather is wonderful today.",
        "Istanbul's historic and cultural richness is famous worldwide.",
        "Turkish cuisine is known for its rich flavours and variety.",
        "Artificial intelligence technologies are rapidly advancing.",
        "Reading books is a fantastic activity that boosts imagination.",
        "Exercising is important for a healthy life.",
        "Music is considered a universal language.",
        "Nature gives people peace and inspiration.",
        "Education is the cornerstone of societal development.",
        "Technology makes our lives easier but should be used in balance."
    ]

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    
    # Configure the padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Tokenise the texts
    print("Tokenising texts...")
    encoded = tokenizer(
        turkish_texts * 10,  # Repeat to create additional data
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    
    # Model parameters
    vocab_size = tokenizer.vocab_size
    dim = 128
    num_heads = 8
    num_layers = 4
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Prepare mini-batches
    num_samples = input_ids.shape[0]
    num_batches = num_samples // batch_size
    
    results = {
        "partial_rope": {"losses": [], "perplexities": [], "times": []},
        "full_rope": {"losses": [], "perplexities": [], "times": []}
    }
    
    # Partial RoPE model
    print("\n=== Training Partial RoPE ===")
    partial_rope = PartialRoPE(dim // num_heads, partial_rotary_factor=0.5)
    model_partial = LanguageModel(vocab_size, dim, num_heads, num_layers, partial_rope).to(device)
    optimizer_partial = torch.optim.Adam(model_partial.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_time = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_data = input_ids[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            start_time = time.time()
            loss = train_step(model_partial, batch_data, batch_labels, optimizer_partial, device)
            epoch_time += time.time() - start_time
            epoch_loss += loss
        
        avg_loss = epoch_loss / num_batches
        perplexity = evaluate_perplexity(model_partial, input_ids[:batch_size], labels[:batch_size], device)
        
        results["partial_rope"]["losses"].append(avg_loss)
        results["partial_rope"]["perplexities"].append(perplexity)
        results["partial_rope"]["times"].append(epoch_time)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}, Time={epoch_time:.3f}s")
    
    # Full RoPE model
    print("\n=== Training Full RoPE ===")
    full_rope = FullRoPE(dim // num_heads)
    model_full = LanguageModel(vocab_size, dim, num_heads, num_layers, full_rope).to(device)
    optimizer_full = torch.optim.Adam(model_full.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_time = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_data = input_ids[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            start_time = time.time()
            loss = train_step(model_full, batch_data, batch_labels, optimizer_full, device)
            epoch_time += time.time() - start_time
            epoch_loss += loss
        
        avg_loss = epoch_loss / num_batches
        perplexity = evaluate_perplexity(model_full, input_ids[:batch_size], labels[:batch_size], device)
        
        results["full_rope"]["losses"].append(avg_loss)
        results["full_rope"]["perplexities"].append(perplexity)
        results["full_rope"]["times"].append(epoch_time)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}, Time={epoch_time:.3f}s")
    
    # Visualise the results
    visualize_results(results)
    
    # Summary statistics
    print("\n=== Performance Summary ===")
    print(f"Partial RoPE - Final Loss: {results['partial_rope']['losses'][-1]:.4f}")
    print(f"Full RoPE - Final Loss: {results['full_rope']['losses'][-1]:.4f}")
    print(f"Partial RoPE - Final Perplexity: {results['partial_rope']['perplexities'][-1]:.2f}")
    print(f"Full RoPE - Final Perplexity: {results['full_rope']['perplexities'][-1]:.2f}")
    print(f"Partial RoPE - Avg Time/Epoch: {np.mean(results['partial_rope']['times']):.3f}s")
    print(f"Full RoPE - Avg Time/Epoch: {np.mean(results['full_rope']['times']):.3f}s")

    # Speed-up percentage
    speed_gain = (np.mean(results['full_rope']['times']) - np.mean(results['partial_rope']['times'])) / np.mean(results['full_rope']['times']) * 100
    print(f"\nPartial RoPE speed-up: {speed_gain:.1f}%")


def visualize_results(results):
    """Visualise benchmarking metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, len(results["partial_rope"]["losses"]) + 1)
    
    # Loss chart
    axes[0].plot(epochs, results["partial_rope"]["losses"], 'b-', label='Partial RoPE', linewidth=2)
    axes[0].plot(epochs, results["full_rope"]["losses"], 'r-', label='Full RoPE', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity chart
    axes[1].plot(epochs, results["partial_rope"]["perplexities"], 'b-', label='Partial RoPE', linewidth=2)
    axes[1].plot(epochs, results["full_rope"]["perplexities"], 'r-', label='Full RoPE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Perplexity Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Training time chart
    axes[2].plot(epochs, results["partial_rope"]["times"], 'b-', label='Partial RoPE', linewidth=2)
    axes[2].plot(epochs, results["full_rope"]["times"], 'r-', label='Full RoPE', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Training Time per Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rope_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def inference_comparison(model_partial, model_full, tokenizer, device):
    """Compare inference behaviour between the models."""
    print("\n=== Inference Performance Comparison ===")
    
    test_texts = [
        "Today the weather",
        "The capital of Turkey",
        "Artificial intelligence",
        "My favourite meal"
    ]
    
    model_partial.eval()
    model_full.eval()
    
    for text in test_texts:
        print(f"\nInput: '{text}'")
        
        # Tokenise the prompt
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        # Prediction with Partial RoPE
        with torch.no_grad():
            start_time = time.time()
            mask = create_causal_mask(input_ids.shape[1], device)
            logits_partial = model_partial(input_ids, mask)
            partial_time = time.time() - start_time
            
            # Select the most probable next token
            next_token_partial = torch.argmax(logits_partial[0, -1, :])
            next_word_partial = tokenizer.decode(next_token_partial)
        
        # Prediction with Full RoPE
        with torch.no_grad():
            start_time = time.time()
            logits_full = model_full(input_ids, mask)
            full_time = time.time() - start_time
            
            # Select the most probable next token
            next_token_full = torch.argmax(logits_full[0, -1, :])
            next_word_full = tokenizer.decode(next_token_full)
        
        print(f"  Partial RoPE prediction: '{next_word_partial}' (Time: {partial_time*1000:.2f}ms)")
        print(f"  Full RoPE prediction: '{next_word_full}' (Time: {full_time*1000:.2f}ms)")


def memory_comparison():
    """Contrast memory usage between the two approaches."""
    print("\n=== Memory Usage Comparison ===")
    
    dim = 512
    seq_len = 1024
    batch_size = 32
    num_heads = 8
    head_dim = dim // num_heads
    
    # Memory usage for Partial RoPE
    partial_rope = PartialRoPE(head_dim, max_position_embeddings=seq_len, partial_rotary_factor=0.5)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        partial_rope = partial_rope.cuda()
        q, k = q.cuda(), k.cuda()
        
        # Forward pass
        _ = partial_rope(q, k)
        partial_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Memory usage for Full RoPE
        torch.cuda.reset_peak_memory_stats()
        full_rope = FullRoPE(head_dim, max_position_embeddings=seq_len).cuda()
        
        # Forward pass
        _ = full_rope(q, k)
        full_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        print(f"Partial RoPE memory usage: {partial_memory:.2f} MB")
        print(f"Full RoPE memory usage: {full_memory:.2f} MB")
        print(f"Memory savings: {(full_memory - partial_memory) / full_memory * 100:.1f}%")
    else:
        print("CUDA is not available, memory comparison skipped.")


def ablation_study():
    """Ablation study for varying partial_rotary_factor values."""
    print("\n=== Ablation Study: Partial Rotary Factor Variants ===")
    
    factors = [0.25, 0.5, 0.75, 1.0]
    dim = 64
    seq_len = 128
    batch_size = 16
    num_iterations = 100
    
    results = {}
    
    for factor in factors:
        if factor == 1.0:
            rope = FullRoPE(dim)
        else:
            rope = PartialRoPE(dim, partial_rotary_factor=factor)
        
        q = torch.randn(batch_size, 1, seq_len, dim)
        k = torch.randn(batch_size, 1, seq_len, dim)
        
        # Measure execution time
        start_time = time.time()
        for _ in range(num_iterations):
            _ = rope(q, k)
        total_time = time.time() - start_time
        
        avg_time = total_time / num_iterations * 1000  # ms
        results[factor] = avg_time
        
        print(f"Factor {factor}: {avg_time:.3f} ms/iteration")
    
    # Visualise the ablation results
    plt.figure(figsize=(8, 6))
    factors_list = list(results.keys())
    times_list = list(results.values())
    
    plt.bar(factors_list, times_list, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Partial Rotary Factor')
    plt.ylabel('Average Time (ms)')
    plt.title('Performance for Different Partial Rotary Factors')
    plt.grid(True, alpha=0.3)
    
    # Annotate the bars with values
    for i, (factor, exec_time) in enumerate(zip(factors_list, times_list)):
        plt.text(i, exec_time + 0.01, f'{exec_time:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main entry point
if __name__ == "__main__":
    print("Partial RoPE vs Full RoPE Performance Benchmark")
    print("=" * 60)

    # Main benchmark
    benchmark_rope_performance()

    # Memory comparison
    memory_comparison()

    # Ablation study
    ablation_study()

    print("\n✅ All benchmarks completed!")
    print("📊 Charts saved as 'rope_comparison.png' and 'ablation_study.png'.")