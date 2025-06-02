import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np

# 1. RoPE (Rotary Position Embedding) - Llama'da kullanılan
class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE, pozisyonel bilgiyi doğrudan attention hesaplamasına entegre eder.
    Avantajları:
    - Extrapolation capability (training'den uzun sequence'larda çalışır)
    - Relative position bilgisi
    - Efficiency
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Frequency hesaplama
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache için sin/cos değerleri
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rotary_pos_emb(q, k, cos, sin):
    """RoPE uygulama fonksiyonu"""
    def rotate_half(x):
        # x'in yarısını rotate et
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

# 2. RMSNorm - LayerNorm'dan daha verimli
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization
    - LayerNorm'dan daha hızlı (mean hesaplama yok)
    - Llama'da kullanılır
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS hesaplama
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm

# 3. SwiGLU Activation - Llama'nın kullandığı
class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    - GLU (Gated Linear Unit) + Swish activation
    - Standard FFN'den daha iyi performance
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Output projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = Swish(xW1) ⊙ (xW3) W2
        gate = F.silu(self.w1(x))  # Swish activation
        up = self.w3(x)
        return self.w2(gate * up)

# 4. Grouped Query Attention (GQA) - Memory efficient
class GroupedQueryAttention(nn.Module):
    """
    GQA - Query/Key/Value head'leri farklı sayıda
    - Multi-Head Attention ve Multi-Query Attention arası compromise
    - Memory efficiency + quality balance
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.group_size = n_heads // n_kv_heads
        
        # Query için tüm head'ler
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        # Key/Value için daha az head
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        
        # Q, K, V projections
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        
        # RoPE uygula
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # K, V'yi group_size kadar repeat et
        k = k.repeat_interleave(self.group_size, dim=2)
        v = v.repeat_interleave(self.group_size, dim=2)
        
        # Attention hesaplama
        q = q.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape ve output projection
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)

# 5. Modern Transformer Block with Pre-normalization
class TransformerBlock(nn.Module):
    """
    Modern transformer block:
    - Pre-normalization (norm önce gelir)
    - Residual connections
    - SwiGLU FFN
    - GQA attention
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, norm_eps: float = 1e-6):
        super().__init__()
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        self.feed_forward = SwiGLU(dim, int(2.67 * dim))  # Llama scaling
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-norm attention
        attn_out = self.attention(self.attention_norm(x), mask)
        x = x + attn_out  # Residual connection
        
        # Pre-norm FFN
        ffn_out = self.feed_forward(self.ffn_norm(x))
        x = x + ffn_out  # Residual connection
        
        return x

# 6. Weight Tying ve Advanced Initialization
def scaled_init_(tensor: torch.Tensor, scale: float = 1.0):
    """Modern weight initialization"""
    std = scale / math.sqrt(tensor.shape[-1])
    torch.nn.init.normal_(tensor, mean=0.0, std=std)

class ModernLLM(nn.Module):
    """
    Modern LLM with all techniques:
    - RoPE, RMSNorm, SwiGLU, GQA
    - Pre-normalization
    - Weight tying
    - Scaled initialization
    """
    def __init__(self, vocab_size: int, dim: int, n_layers: int, 
                 n_heads: int, n_kv_heads: int, norm_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, norm_eps)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(dim, norm_eps)
        
        # Output projection (weight tying ile)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying: input ve output embedding'leri paylaş
        self.output.weight = self.tok_embeddings.weight
        
        # Modern initialization
        self.init_weights()
    
    def init_weights(self):
        """Scaled initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                scaled_init_(module.weight)
            elif isinstance(module, nn.Embedding):
                scaled_init_(module.weight)
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        bsz, seq_len = tokens.shape
        
        # Token embeddings
        x = self.tok_embeddings(tokens)
        
        # Causal mask oluştur
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tokens.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Output projection
        logits = self.output(x)
        
        if targets is not None:
            # Training loss hesaplama
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
            return logits, loss
        
        return logits

# Usage example
if __name__ == "__main__":
    # Model parametreleri (Llama-style)
    model = ModernLLM(
        vocab_size=32000,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,  # GQA için daha az KV head
        norm_eps=1e-6
    )
    
    # Örnek input
    batch_size, seq_len = 2, 512
    tokens = torch.randint(0, 32000, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(tokens)
        print(f"Output shape: {logits.shape}")  # (2, 512, 32000)
        
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
