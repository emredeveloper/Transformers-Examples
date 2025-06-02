import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import pandas as pd
from collections import Counter
import re
import os
from typing import List, Dict, Tuple, Optional, Union

# =============================================================================
# 1. TOKENIZER - Metni sayısal verilere dönüştürür
# =============================================================================

class SimpleTokenizer:
    """Geliştirilmiş tokenizer"""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.pad_token_id = 0  # PAD token ID'sini 0 olarak ayarla
        self.unk_token_id = 1  # UNK token ID'si
        self.bos_token_id = 2  # BOS token ID'si
        self.eos_token_id = 3  # EOS token ID'si
        
    def fit(self, texts: List[str]):
        """Metinlerden vocab oluştur"""
        # Tüm karakterleri topla ve frekanslarını hesapla
        char_freq = {}
        for text in texts:
            for char in text:
                if char not in ['\n', ' ']:  # Boşluk ve yeni satırı özel karakterlerden ayır
                    char_freq[char] = char_freq.get(char, 0) + 1
        
        # Özel tokenlar ve sık kullanılan karakterler
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token, '\n', ' ']
        
        # En sık kullanılan 200 karakteri al (özel tokenlar hariç)
        common_chars = [char for char, _ in sorted(char_freq.items(), key=lambda x: -x[1])[:200]]
        
        # Özel token'ları ve yaygın karakterleri birleştir
        all_chars = special_tokens + common_chars
        
        # Benzersiz karakterlerin listesini oluştur
        unique_chars = []
        for char in all_chars:
            if char not in unique_chars:
                unique_chars.append(char)
        
        # Sözlükleri oluştur
        self.char_to_id = {char: i for i, char in enumerate(unique_chars)}
        self.id_to_char = {i: char for i, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        
        # Özel token ID'lerini güncelle
        self.pad_token_id = self.char_to_id[self.pad_token]
        self.unk_token_id = self.char_to_id[self.unk_token]
        self.bos_token_id = self.char_to_id[self.bos_token]
        self.eos_token_id = self.char_to_id[self.eos_token]
        
        # Vocab'ı oluştur (özel tokenlar + en sık kullanılan karakterler)
        vocab = special_tokens + common_chars
        
        # ID mapping oluştur
        self.char_to_id = {char: i for i, char in enumerate(vocab)}
        self.id_to_char = {i: char for i, char in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        # Özel token ID'lerini sakla
        self.pad_token_id = self.char_to_id.get('<PAD>', 0)
        self.unk_token_id = self.char_to_id.get('<UNK>', 1)
        self.bos_token_id = self.char_to_id.get('<BOS>', 2)
        self.eos_token_id = self.char_to_id.get('<EOS>', 3)
        
        print(f"Vocab boyutu: {self.vocab_size}")
        print(f"İlk 20 token: {vocab[:20]}")
        
    def encode(self, text: str, max_length: int = 512, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Metni token ID'lerine çevir"""
        # Özel tokenları ekle
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)
            
        # Metni tokenlara çevir
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_token_id))
            
        if add_eos:
            tokens.append(self.eos_token_id)
            
        # Uzunluğu max_length'e göre ayarla
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [tokens[-1]]  # Son token'ı koru
        elif len(tokens) < max_length:
            # Padding ekle
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
            
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Token ID'leri metne dönüştür"""
        chars = []
        for token_id in token_ids:
            if token_id == self.char_to_id['<EOS>']:
                break
            if token_id != self.char_to_id['<PAD>']:
                chars.append(self.id_to_char.get(token_id, '<UNK>'))
        return ''.join(chars)

# =============================================================================
# 2. TRANSFORMER COMPONENTS - Attention ve FFN katmanları
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention katmanı"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Q, K, V hesapla
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention hesapla
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal mask uygula (gelecekteki tokenlara bakmasın)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Attention uygula
        context = torch.matmul(attention_weights, V)
        
        # Reshape ve output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return output

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer decoder bloğu"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention + residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward + residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# =============================================================================
# 3. LLM MODEL - Ana transformer modeli
# =============================================================================

class SimpleLLM(nn.Module):
    """Basit Large Language Model"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding katmanları
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer katmanları
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Output katmanı
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Parametreleri initialize et
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Causal mask oluştur (gelecekteki tokenlara bakmasın)"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Position IDs oluştur
        position_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        
        # Attention mask'i oluştur
        # Shape: (batch_size, 1, 1, seq_len) olmalı
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        
        # Mask değerlerini float'a çevir ve çok küçük bir sayı yap
        # 1 olan yerler gözükür, 0 olanlar maskelenir
        mask = (1.0 - mask.float()) * -1e9
        
        # Transformer katmanları
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
            
        # Language model head
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9, pad_token_id=None):
        """
        Metin üretme metodu
        
        Args:
            input_ids: Giriş token ID'leri (batch_size, seq_len)
            max_length: Maksimum üretilecek token sayısı
            temperature: Düşük değerler daha tahmin edilebilir çıktılar üretir
            top_k: Top-k sampling için k değeri
            top_p: Nucleus sampling için p değeri
            pad_token_id: Padding token ID'si
            
        Returns:
            Üretilen token ID'leri (batch_size, seq_len + max_length)
        """
        device = next(self.parameters()).device
        batch_size = input_ids.size(0)
        
        # Girişi cihaza taşı
        input_ids = input_ids.to(device)
        
        # Çıktıyı girişle başlat
        generated = input_ids
        
        # Eğitim modunu kapat
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Mevcut çıktı için maske oluştur
                seq_len = generated.size(1)
                attn_mask = (generated != pad_token_id).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
                
                # Modelden çıktı al
                outputs = self(generated, mask=attn_mask)
                
                # Sadece son token için logitleri al
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Top-k sampling uygula
                if top_k > 0:
                    # En yüksek olasılıklı k token dışındakileri -inf yap
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Kümülatif olasılığı p'den büyük olan en küçük indeksleri bul
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # İlk indeksi koru (en yüksek olasılıklı token)
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Sıralanmış logitlerden kaldırılacak olanları -inf yap
                    sorted_logits[sorted_indices_to_remove] = -float('Inf')
                    
                    # Orijinal sıraya geri dön
                    next_token_logits = torch.zeros_like(next_token_logits).scatter_(
                        dim=1, index=sorted_indices, src=sorted_logits
                    )
                
                # Sonraki tokenı seç
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Eğer pad_token_id verildiyse ve tüm olasılıklar -inf ise pad_token_id kullan
                if pad_token_id is not None and torch.all(torch.isinf(probs)):
                    next_tokens = torch.full_like(next_tokens, pad_token_id)
                
                # Üretilen token'ı çıktıya ekle
                generated = torch.cat((generated, next_tokens), dim=1)
                
                # Eğer tüm örnekler sonlandırıldıysa döngüden çık
                if pad_token_id is not None and torch.all(next_tokens == pad_token_id):
                    break
        
        return generated

# =============================================================================
# 4. DATASET - Eğitim verisi hazırlama
# =============================================================================

def build_vocab_from_csv(csv_path: str) -> List[str]:
    """CSV dosyasından karakter bazında vocabulary oluştur"""
    df = pd.read_csv(csv_path)
    all_text = ""
    
    # Tüm metinleri birleştir
    for _, row in df.iterrows():
        all_text += row['Question'] + " "
        all_text += row['A'] + " " + row['B'] + " " + row['C'] + " " + row['D'] + " "
    
    # Benzersiz karakterleri al ve sırala
    unique_chars = sorted(list(set(all_text)))
    return unique_chars

class MMLUDataset(Dataset):
    """MMLU dataset sınıfı"""
    
    def __init__(self, csv_path: str, tokenizer: SimpleTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # CSV'yi yükle
        df = pd.read_csv(csv_path)
        
        # Tüm metinleri topla
        all_texts = []
        for _, row in df.iterrows():
            question = row['Question']
            options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
            prompt = f"Question: {question}\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\nAnswer:"
            all_texts.append(prompt)
        
        # Tokenizer'ı eğit
        self.tokenizer.fit(all_texts)
        
        # Veri setini oluştur
        for i, row in df.iterrows():
            question = row['Question']
            options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
            answer = row['Answer']
            
            # Prompt formatı: "Question: [soru]\nA) [A]\nB) [B]\nC) [C]\nD) [D]\nAnswer:"
            prompt = f"Question: {question}\n"
            for i, opt in enumerate(['A', 'B', 'C', 'D']):
                prompt += f"{opt}) {options[i]}\n"
            prompt += "Answer:"
            
            # Cevabı token olarak kodla (A->0, B->1, C->2, D->3)
            target = ord(str(answer).strip().upper()[0]) - ord('A')
            
            # Tokenize et ve kaydet
            tokens = self.tokenizer.encode(prompt, max_length)
            if len(tokens) > 0 and 0 <= target <= 3:  # Sadece geçerli hedefleri kabul et
                self.data.append((tokens, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens, target = self.data[idx]
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)
        return x, y

# =============================================================================
# 5. TRAINING - Model eğitimi
# =============================================================================

class LLMTrainer:
    """LLM eğitim sınıfı"""
    
    def __init__(self, model: SimpleLLM, tokenizer: SimpleTokenizer, device: str = 'cpu',
                 learning_rate: float = 3e-4, weight_decay: float = 0.01,
                 warmup_steps: int = 1000, total_steps: int = 10000):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1  # Minimum learning rate
        )
        
        # Warmup için
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Loss fonksiyonu
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Bir epoch eğitim"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Sadece son token'ın çıktısını al (cevap pozisyonu)
            last_token_logits = outputs[:, -1, :]  # [batch_size, vocab_size]
            
            # Loss hesapla
            loss = self.criterion(last_token_logits, targets)
            
            # Doğru tahminleri say
            _, predicted = torch.max(last_token_logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Backward pass ve optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer adımı
            self.optimizer.step()
            
            # Learning rate warmup ve schedule
            self.current_step += 1
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr_scale = min(1.0, float(self.current_step) / self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.optimizer.defaults['lr'] * lr_scale
            else:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Her 10 batch'te bir log göster
            if (batch_idx + 1) % 10 == 0:
                batch_acc = (predicted == targets).float().mean().item() * 100
                print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%")
        
        accuracy = 100 * correct / total if total > 0 else 0
        return total_loss / len(dataloader), accuracy
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Metin üretimi"""
        self.model.eval()
        
        # Prompt'u tokenize et
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Model prediction
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sampling
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).item()
                
                # EOS token kontrolü
                if next_token == self.tokenizer.char_to_id['<EOS>']:
                    break
                
                # Yeni token ekle
                generated_tokens.append(next_token)
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                
                # Maksimum sequence length kontrolü
                if input_ids.size(1) >= self.model.max_seq_len:
                    input_ids = input_ids[:, 1:]  # İlk tokenı çıkar
        
        return self.tokenizer.decode(generated_tokens)

# =============================================================================
# 6. MAIN - Ana çalıştırma kodu
# =============================================================================

def main():
    # Parametreler
    batch_size = 16
    max_length = 512  # Daha uzun sequence'ler için
    d_model = 512  # Daha büyük model boyutu
    n_heads = 8  # Daha fazla head
    n_layers = 6  # Daha fazla layer
    d_ff = 2048  # Daha büyük feed forward
    dropout = 0.1  
    num_epochs = 1 # 5 epoch için ayarlandı  # Daha fazla epoch
    max_examples = 1000
    learning_rate = 3e-4
    weight_decay = 0.01
    warmup_steps = 1000  # Learning rate warmup için adım sayısı
    
    # Cihaz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CSV dosya yolu
    csv_path = r"c:\Users\emreq\Downloads\archive\mmlu.csv"
    
    # Tokenizer'ı oluştur
    tokenizer = SimpleTokenizer()
    
    # MMLU veri setini yükle
    print("Veri seti yükleniyor...")
    dataset = MMLUDataset(csv_path, tokenizer, max_length)
    
    # Eğer veri kümesi boşsa hata ver
    if len(dataset) == 0:
        raise ValueError("Veri kümesi boş. CSV dosyasını ve veri yapısını kontrol edin.")
    
    # Veri setini 1000 örnekle sınırla
    if len(dataset) > max_examples:
        print(f"Veri seti {len(dataset)} örnekten {max_examples} örneğe indiriliyor...")
        indices = torch.randperm(len(dataset))[:max_examples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Eğitim ve test setlerine ayır (%80 eğitim, %20 test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # DataLoader'ları oluştur
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Modeli oluştur
    print("Model oluşturuluyor...")
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_length,
        dropout=dropout
    ).to(device)
    
    # Learning rate scheduler için toplam adım sayısı
    total_steps = len(train_dataloader) * num_epochs
    
    # Trainer oluştur
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # Eğitim döngüsü
    print(f"Eğitim başlıyor... Toplam {len(train_dataset)} eğitim, {len(test_dataset)} test örneği")
    
    best_test_acc = 0
    for epoch in range(num_epochs):
        # Eğitim
        train_loss, train_acc = trainer.train_epoch(train_dataloader)
        
        # Test
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                last_token_logits = outputs[:, -1, :]
                
                # Loss hesapla
                loss = F.cross_entropy(last_token_logits, targets)
                test_loss += loss.item()
                
                # Doğruluk hesapla
                _, predicted = torch.max(last_token_logits, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        test_loss = test_loss / len(test_dataloader)
        test_acc = 100 * correct / total if total > 0 else 0
        
        # En iyi modeli kaydet
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Sadece gerekli bilgileri kaydet
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': test_acc,
                'tokenizer_chars': tokenizer.char_to_id,  # Sadece karakter-ID eşlemesini kaydet
                'vocab_size': tokenizer.vocab_size
            }, 'best_model.pt')
            print(f"Yeni en iyi model kaydedildi! Test Doğruluğu: {test_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
    
    # Test etme
    print("\n=== Final Test ===\n")
    
    # En iyi modeli yükle
    if os.path.exists('best_model.pt'):
        checkpoint = torch.load('best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nEn iyi model yüklendi (Doğruluk: {checkpoint['accuracy']:.2f}%)\n")
    
    model.eval()
    
    # Örnek test soruları ve doğru cevapları
    test_questions = [
        {
            "question": "Soru: İstanbul'un fethi hangi yılda olmuştur?\nA) 1451\nB) 1453\nC) 1455\nD) 1457\nCevap:",
            "correct": "B"
        },
        {
            "question": "Soru: Python programlama dili kim tarafından geliştirilmiştir?\nA) Guido van Rossum\nB) James Gosling\nC) Bjarne Stroustrup\nD) Dennis Ritchie\nCevap:",
            "correct": "A"
        },
        {
            "question": "Soru: Dünya'nın en büyük okyanusu hangisidir?\nA) Atlas Okyanusu\nB) Hint Okyanusu\nC) Arktik Okyanusu\nD) Büyük Okyanus\nCevap:",
            "correct": "D"
        },
        {
            "question": "Soru: Aşağıdakilerden hangisi bir yapay zeka kütüphanesidir?\nA) React\nB) TensorFlow\nC) Django\nD) Flask\nCevap:",
            "correct": "B"
        }
    ]
    
    correct_answers = 0
    
    for i, item in enumerate(test_questions, 1):
        question = item["question"]
        correct = item["correct"]
        
        print(f"\n--- Test {i} ---")
        print("Soru:")
        print(question)
        
        # Modelden cevap al
        with torch.no_grad():
            # Sadece soru kısmını tokenize et
            question_tokens = tokenizer.encode(question, max_length=512, add_bos=True, add_eos=False)
            input_tensor = torch.tensor([question_tokens], device=device)
            
            # Cevap oluştur
            output = model.generate(
                input_tensor,
                max_length=len(question_tokens) + 5,  # Cevap için 5 token yeterli
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Tüm çıktıyı al
            full_output = tokenizer.decode(output[0].tolist())
            
            # Sadece son 5 tokeni al (cevap genellikle sonlarda olur)
            last_tokens = output[0][-5:].tolist()
            last_chars = tokenizer.decode(last_tokens)
            
            # Cevap olarak A, B, C veya D harfini ara
            answer = None
            for c in last_chars.upper():
                if c in ['A', 'B', 'C', 'D']:
                    answer = c
                    break
            
            if answer is None:
                answer = "(Cevap bulunamadı)"
            
            # Doğru cevabı kontrol et
            is_correct = (answer == correct)
            if is_correct:
                correct_answers += 1
            
            print("\nModelin Cevabı:", answer)
            print("Doğru Cevap:", correct)
            print("Sonuç:", "✅ Doğru" if is_correct else "❌ Yanlış")
            print("\nTam Çıktı:", full_output)
        
        print("\n" + "="*80)
    
    # Genel başarı oranını göster
    accuracy = (correct_answers / len(test_questions)) * 100
    print(f"\n=== Test Sonuçları ===")
    print(f"Doğru Cevaplar: {correct_answers}/{len(test_questions)}")
    print(f"Başarı Oranı: {accuracy:.1f}%")
    print("\n=== Eğitim ve Test Tamamlandı! ===")
    
    return model, tokenizer, trainer

if __name__ == "__main__":
    model, tokenizer, trainer = main()