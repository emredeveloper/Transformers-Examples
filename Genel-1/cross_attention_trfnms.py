import torch
import torch.nn as nn
import math
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps  # Small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scaling parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable bias parameter

    def forward(self, x):
        # Compute the mean and standard deviation of the input
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Normalization formula: (x - mean) / (std + eps) * alpha + bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # Projection layers
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply the gate and up projections
        gate = torch.sigmoid(self.gate_proj(x))  # Gate mechanism
        up = self.up_proj(x)  # Up projection
        # Combine the gate and up paths
        x = gate * up
        # Apply dropout and the down projection
        return self.down_proj(self.dropout(x))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding dimension
        self.vocab_size = vocab_size  # Vocabulary size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer

    def forward(self, x):
        # Convert token indices to embeddings and scale them
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding dimension
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        # Build the positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Position vector
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Divisor term
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
        pe = pe.unsqueeze(0)  # Add the batch dimension
        self.register_buffer('pe', pe)  # Register the positional encoding as a buffer

    def forward(self, x):
        # Add positional encoding to the input
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.norm = LayerNormalization(features)  # Layer normalization

    def forward(self, x, sublayer):
        # Residual connection: x + dropout(sublayer(norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)  # Output projection

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # This line fills masked positions with -1e9 to suppress attention on those tokens.
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Query, key, value projections
        query = self.q_proj(q)
        key = self.k_proj(k)
        value = self.v_proj(v)

        # Reshape for multi-head attention
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply the attention mechanism
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Merge the heads and apply the output projection
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.o_proj(x)


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention layer
        self.feed_forward_block = feed_forward_block  # Feed-forward network
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])  # Residual connections

    def forward(self, x, src_mask):
        # Self-attention with a residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Feed-forward network with a residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Encoder blocks
        self.norm = LayerNormalization(features)  # Final layer normalization

    def forward(self, x, mask):
        # Apply all encoder blocks
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # Final layer normalization


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention layer
        self.cross_attention_block = cross_attention_block  # Cross-attention layer
        self.feed_forward_block = feed_forward_block  # Feed-forward network
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])  # Residual connections

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Self-attention: the decoder attends to its own outputs
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Cross-attention: the decoder attends to the encoder outputs
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Decoder blocks
        self.norm = LayerNormalization(features)  # Final layer normalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Apply all decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  # Final layer normalization


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # Linear projection layer

    def forward(self, x) -> None:
        # Project the input to the vocabulary dimension
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder  # Encoder module
        self.decoder = decoder  # Decoder module
        self.src_embed = src_embed  # Source embedding layer
        self.tgt_embed = tgt_embed  # Target embedding layer
        self.src_pos = src_pos  # Source positional encoding
        self.tgt_pos = tgt_pos  # Target positional encoding
        self.projection_layer = projection_layer  # Projection layer

    def encode(self, src, src_mask):
        # Encode the source sequence
        src = self.src_embed(src)  # Embedding layer
        src = self.src_pos(src)  # Positional encoding
        return self.encoder(src, src_mask)  # Encoder module

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # Decode the target sequence
        tgt = self.tgt_embed(tgt)  # Embedding layer
        tgt = self.tgt_pos(tgt)  # Positional encoding
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # Decoder module

    def project(self, x):
        # Project the output to the vocabulary dimension
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Build the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Build the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Build the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Build the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Build the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Build the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Build the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize parameters with Xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps  # Small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scaling parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable bias parameter

    def forward(self, x):
        # Compute the mean and standard deviation of the input
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Normalization formula: (x - mean) / (std + eps) * alpha + bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # Projection layers
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply the gate and up projections
        gate = torch.sigmoid(self.gate_proj(x))  # Gate mechanism
        up = self.up_proj(x)  # Up projection
        # Combine the gate and up paths
        x = gate * up
        # Apply dropout and the down projection
        return self.down_proj(self.dropout(x))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding dimension
        self.vocab_size = vocab_size  # Vocabulary size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer

    def forward(self, x):
        # Convert token indices to embeddings and scale them
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding dimension
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        # Build the positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Position vector
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Divisor term
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
        pe = pe.unsqueeze(0)  # Add the batch dimension
        self.register_buffer('pe', pe)  # Register the positional encoding as a buffer

    def forward(self, x):
        # Add positional encoding to the input
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.norm = LayerNormalization(features)  # Layer normalization

    def forward(self, x, sublayer):
        # Residual connection: x + dropout(sublayer(norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)  # Output projection

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # This line fills masked positions with -1e9 to suppress attention on those tokens.
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Query, key, value projections
        query = self.q_proj(q)
        key = self.k_proj(k)
        value = self.v_proj(v)

        # Reshape for multi-head attention
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply the attention mechanism
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Merge the heads and apply the output projection
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.o_proj(x)


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention layer
        self.feed_forward_block = feed_forward_block  # Feed-forward network
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])  # Residual connections

    def forward(self, x, src_mask):
        # Self-attention with a residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Feed-forward network with a residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Encoder blocks
        self.norm = LayerNormalization(features)  # Final layer normalization

    def forward(self, x, mask):
        # Apply all encoder blocks
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # Final layer normalization


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention layer
        self.cross_attention_block = cross_attention_block  # Cross-attention layer
        self.feed_forward_block = feed_forward_block  # Feed-forward network
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])  # Residual connections

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Self-attention: the decoder attends to its own outputs
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Cross-attention: the decoder attends to the encoder outputs
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Decoder blocks
        self.norm = LayerNormalization(features)  # Final layer normalization

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Apply all decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  # Final layer normalization


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # Linear projection layer

    def forward(self, x) -> None:
        # Project the input to the vocabulary dimension
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder  # Encoder module
        self.decoder = decoder  # Decoder module
        self.src_embed = src_embed  # Source embedding layer
        self.tgt_embed = tgt_embed  # Target embedding layer
        self.src_pos = src_pos  # Source positional encoding
        self.tgt_pos = tgt_pos  # Target positional encoding
        self.projection_layer = projection_layer  # Projection layer

    def encode(self, src, src_mask):
        # Encode the source sequence
        src = self.src_embed(src)  # Embedding layer
        src = self.src_pos(src)  # Positional encoding
        return self.encoder(src, src_mask)  # Encoder module

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # Decode the target sequence
        tgt = self.tgt_embed(tgt)  # Embedding layer
        tgt = self.tgt_pos(tgt)  # Positional encoding
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # Decoder module

    def project(self, x):
        # Project the output to the vocabulary dimension
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Build the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Build the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Build the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Build the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Build the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Build the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Build the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize parameters with Xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
