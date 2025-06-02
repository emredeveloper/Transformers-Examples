import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from typing import Optional
import time
from tqdm import tqdm
import os


# --- Konumsal Kodlama ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 32768):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# --- RMSNorm Implementation ---
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# --- GQA Mekanizması ---
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, num_q_heads: int = 8, num_kv_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.kv_rep_factor = num_q_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(hidden_size, self.head_dim * num_kv_heads)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int, num_heads: int):
        return tensor.view(batch_size, seq_len, num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self._shape(q, seq_len, batch_size, self.num_q_heads)
        k = self._shape(k, seq_len, batch_size, self.num_kv_heads)
        v = self._shape(v, seq_len, batch_size, self.num_kv_heads)

        if self.kv_rep_factor > 1:
            k = k.repeat_interleave(self.kv_rep_factor, dim=1)
            v = v.repeat_interleave(self.kv_rep_factor, dim=1)

        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale_factor
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)
        output = self.o_proj(context)
        return output


# --- FeedForward Network ---
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# --- Transformer Katmanı ---
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_q_heads: int, num_kv_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GroupedQueryAttention(hidden_size, num_q_heads, num_kv_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_size, intermediate_size, dropout)
        self.ln1 = RMSNorm(hidden_size)
        self.ln2 = RMSNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# --- Ana Model ---
class Qwen3SmallModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_q_heads: int = 8,
        num_kv_heads: int = 4,
        intermediate_size: int = 1024,
        max_seq_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.eos_token_id = 102  # BERT'de [SEP]
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_q_heads, num_kv_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size, seq_length = input_ids.size()
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.pos_encoding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1 - extended_attention_mask) * -1e4

        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}


# --- Dataset ve Collate Fonksiyonu ---
class HFDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['soru'].strip()
        answer = item['cevap'].strip()
        if len(answer.split()) < 5:
            answer += " Belirtilmemiş."
        text = f"Soru: {question}\nCevap: {answer}"
        encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': encoded["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


# --- Generate Fonksiyonu (Top-k ve Top-p Sampling ile) ---
def generate_text(model, prompt, tokenizer, device="cuda", max_new_tokens=100, temperature=0.4, top_k=50, top_p=0.9):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    generated = input_ids
    
    cur_len = generated.shape[1]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if attention_mask is not None:
                attention_mask_extended = torch.ones((1, generated.shape[1]), dtype=torch.long, device=device)
                attention_mask_extended[:, :cur_len] = attention_mask
            else:
                attention_mask_extended = None
                
            outputs = model(
                input_ids=generated, 
                attention_mask=attention_mask_extended
            )
            
            next_token_logits = outputs['logits'][:, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            special_tokens_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            for token_id in [tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id]:
                if token_id is not None:
                    special_tokens_mask[0, token_id] = True
            next_token_logits.masked_fill_(special_tokens_mask, -float('inf'))
            
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            zero_mask = probs > 0
            if not zero_mask.any():
                probs = torch.ones_like(probs) / probs.size(-1)
            elif not zero_mask.all():
                probs[~zero_mask] = 0
                probs = probs / probs.sum()
            
            try:
                next_token = torch.multinomial(probs, num_samples=1)
            except RuntimeError:
                next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)

            if next_token.item() == tokenizer.sep_token_id:
                break

            generated = torch.cat([generated, next_token], dim=1)

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = output[len(prompt):].strip()
    
    return answer


# --- Eğitim Fonksiyonu ---
def train_model(
    model, 
    dataloader, 
    optimizer, 
    scheduler=None, 
    epochs=20, 
    device="cuda", 
    gradient_accumulation_steps=4,
    save_path="checkpoints",
    save_steps=100
):
    model.train()
    os.makedirs(save_path, exist_ok=True)
    global_step = 0
    
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % save_steps == 0:
                    checkpoint_path = os.path.join(save_path, f"checkpoint-{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss.item() * gradient_accumulation_steps,
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
            
            total_loss += loss.item() * gradient_accumulation_steps
            bar.set_postfix(loss=loss.item() * gradient_accumulation_steps, lr=optimizer.param_groups[0]['lr'])
            
        print(f"Epoch {epoch+1} Ortalama Kayıp: {total_loss / len(dataloader):.4f}, Süre: {time.time()-start_time:.2f}s")


# --- Main Fonksiyonu ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": 256,
        "num_layers": 6,
        "num_q_heads": 8,
        "num_kv_heads": 4,
        "intermediate_size": 1024,
        "max_seq_length": 512,
        "dropout": 0.1,
    }

    model = Qwen3SmallModel(**model_config).to(device)
    
    batch_size = 8
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    num_epochs = 15
    warmup_steps = 100
    weight_decay = 0.01
    learning_rate = 5e-5
    
    no_decay = ["bias", "LayerNorm.weight", "RMSNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    hf_dataset = load_dataset("umarigan/turkiye_finance_qa")['train']
    train_data = HFDataset(hf_dataset, tokenizer, max_length=512)
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    print(f"Model Eğitiliyor... (Etkin batch boyutu: {effective_batch_size})")
    train_model(
        model, 
        train_loader, 
        optimizer, 
        scheduler=scheduler, 
        epochs=num_epochs, 
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_path="c:/Users/emreq/Desktop/Transformers/Qwen3/checkpoints",
        save_steps=200
    )

    print("\nTest Başlatılıyor...")
    sample_questions = [
        "Yatırım fonlarına yatırım yapmanın dezavantajları nelerdir?",
        "Kamu harcamaları neleri içerir?"
    ]
    for question in sample_questions:
        prompt = f"Soru: {question}\nCevap:"
        answer = generate_text(model, prompt, tokenizer, device=device)
        print(f"\nSoru: {question}")
        print(f"Model Cevabı: {answer}")

    final_model_path = "c:/Users/emreq/Desktop/Transformers/Qwen3/turkiye_finance_qa_model_improved.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model kaydedildi: {final_model_path}")


if __name__ == "__main__":
    main()