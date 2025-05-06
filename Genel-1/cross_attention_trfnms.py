import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps  # Küçük bir değer, sıfıra bölünmeyi önlemek için
        self.alpha = nn.Parameter(torch.ones(features))  # Ölçeklendirme parametresi (öğrenilebilir)
        self.bias = nn.Parameter(torch.zeros(features))  # Kaydırma parametresi (öğrenilebilir)

    def forward(self, x):
        # Girdinin ortalamasını ve standart sapmasını hesapla
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Normalizasyon formülü: (x - mean) / (std + eps) * alpha + bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # Projeksiyon katmanları
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)  # Gate projeksiyonu
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)  # Up projeksiyonu
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)  # Down projeksiyonu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Gate ve Up projeksiyonlarını uygula
        gate = torch.sigmoid(self.gate_proj(x))  # Gate mekanizması
        up = self.up_proj(x)  # Up projeksiyonu
        # Gate ve Up'ı birleştir
        x = gate * up
        # Dropout ve Down projeksiyonu uygula
        return self.down_proj(self.dropout(x))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # Gömme vektörlerinin boyutu
        self.vocab_size = vocab_size  # Kelime dağarcığı boyutu
        self.embedding = nn.Embedding(vocab_size, d_model)  # Gömme katmanı

    def forward(self, x):
        # Token indekslerini gömme vektörlerine dönüştür ve ölçeklendir
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Gömme vektörlerinin boyutu
        self.seq_len = seq_len  # Maksimum dizi uzunluğu
        self.dropout = nn.Dropout(dropout)  # Dropout katmanı
        # Konumsal kodlama matrisini oluştur
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Pozisyon vektörü
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Bölme terimi
        pe[:, 0::2] = torch.sin(position * div_term)  # Çift indeksler için sinüs
        pe[:, 1::2] = torch.cos(position * div_term)  # Tek indeksler için kosinüs
        pe = pe.unsqueeze(0)  # Batch boyutu ekle
        self.register_buffer('pe', pe)  # Konumsal kodlamayı sabit olarak kaydet

    def forward(self, x):
        # Girdiye konumsal kodlamayı ekle
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout katmanı
        self.norm = LayerNormalization(features)  # Katman normalizasyonu

    def forward(self, x, sublayer):
        # Artık bağlantı: x + dropout(sublayer(norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        # Projeksiyon katmanları
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # Query projeksiyonu
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # Key projeksiyonu
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # Value projeksiyonu
        self.o_proj = nn.Linear(d_model, d_model, bias=False)  # Çıktı projeksiyonu

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # Bu satır, mask değeri sıfır olan konumları −1e9 ile doldurur. Böylece o konumlardaki dikkat skorları etkisiz hâle getirilir (maskeleme).
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Query, Key, Value projeksiyonları
        query = self.q_proj(q)
        key = self.k_proj(k)
        value = self.v_proj(v)

        # Çok kafalı dikkat için şekil değiştir
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Dikkat mekanizmasını uygula
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Kafaları birleştir ve çıktı projeksiyonu uygula
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.o_proj(x)


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention katmanı
        self.feed_forward_block = feed_forward_block  # İleri beslemeli sinir ağı
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])  # Artık bağlantılar

    def forward(self, x, src_mask):
        # Self-attention ve artık bağlantı
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # İleri beslemeli sinir ağı ve artık bağlantı
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Encoder blokları
        self.norm = LayerNormalization(features)  # Son katman normalizasyonu

    def forward(self, x, mask):
        # Tüm encoder bloklarını uygula
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # Son katman normalizasyonu


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # Self-attention katmanı
        self.cross_attention_block = cross_attention_block  # Cross-attention katmanı
        self.feed_forward_block = feed_forward_block  # İleri beslemeli sinir ağı
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])  # Artık bağlantılar

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Self-attention: Decoder'ın kendi çıktısına dikkat eder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Cross-attention: Decoder, encoder'ın çıktısına dikkat eder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # İleri beslemeli sinir ağı
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # Decoder blokları
        self.norm = LayerNormalization(features)  # Son katman normalizasyonu

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Tüm decoder bloklarını uygula
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  # Son katman normalizasyonu


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # Lineer projeksiyon katmanı

    def forward(self, x) -> None:
        # Girdiyi kelime dağarcığı boyutuna projelendir
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder  # Encoder katmanı
        self.decoder = decoder  # Decoder katmanı
        self.src_embed = src_embed  # Kaynak gömme katmanı
        self.tgt_embed = tgt_embed  # Hedef gömme katmanı
        self.src_pos = src_pos  # Kaynak konumsal kodlama
        self.tgt_pos = tgt_pos  # Hedef konumsal kodlama
        self.projection_layer = projection_layer  # Projeksiyon katmanı

    def encode(self, src, src_mask):
        # Kaynak diziyi kodla
        src = self.src_embed(src)  # Gömme katmanı
        src = self.src_pos(src)  # Konumsal kodlama
        return self.encoder(src, src_mask)  # Encoder katmanı

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # Hedef diziyi çöz
        tgt = self.tgt_embed(tgt)  # Gömme katmanı
        tgt = self.tgt_pos(tgt)  # Konumsal kodlama
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # Decoder katmanı

    def project(self, x):
        # Çıktıyı kelime dağarcığı boyutuna projelendir
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Gömme katmanlarını oluştur
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Konumsal kodlama katmanlarını oluştur
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Encoder bloklarını oluştur
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder bloklarını oluştur
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Encoder ve Decoder'ı oluştur
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Projeksiyon katmanını oluştur
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Transformer modelini oluştur
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Parametreleri Xavier uniform ile başlat
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer