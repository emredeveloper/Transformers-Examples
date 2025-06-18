from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import re
from tqdm import tqdm
from collections import Counter

# HuggingFace veri setini yükle
dataset = load_dataset('salihturkoglu/se_data_set', split='train')
instructions = [ex['instruction'] for ex in dataset]
responses = [ex['response'] for ex in dataset]

# Gelişmiş Türkçe tokenizer
def turkish_tokenize(text):
    # Noktalama, sayılar, Türkçe karakterler ve kelime kökleri için daha iyi ayrıştırma
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    text = re.sub(r"([0-9]+)", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip().split()

# Vocab oluştur (daha büyük ve çeşitli)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
all_texts = instructions + responses
counter = Counter()
for text in all_texts:
    counter.update(turkish_tokenize(text))
vocab_list = [PAD_TOKEN, UNK_TOKEN] + [tok for tok, count in counter.items() if count >= 1][:50000]
vocab = {tok: idx for idx, tok in enumerate(vocab_list)}
reverse_vocab = {idx: tok for tok, idx in vocab.items()}

def encode(text):
    return [vocab.get(tok, vocab[UNK_TOKEN]) for tok in turkish_tokenize(text)]

def decode(token_ids):
    # <UNK> oranını azaltmak için tekrarları ve padleri temizle
    words = []
    for idx in token_ids:
        if idx == vocab[PAD_TOKEN]:
            continue
        word = reverse_vocab.get(idx, UNK_TOKEN)
        if not words or word != words[-1]:
            words.append(word)
    return " ".join(words)

def build_prompt(instruction, response=None):
    # Prompt formatı
    if response is not None:
        return f"Instruction: {instruction} Response: {response}"
    else:
        return f"Instruction: {instruction} Response:"

class InstructionResponseDataset(Dataset):
    def __init__(self, instructions, responses, vocab, max_len=128, prompt_len=64):
        self.inputs = []
        self.targets = []
        self.max_len = max_len
        self.prompt_len = prompt_len
        self.vocab = vocab
        for instr, resp in zip(instructions, responses):
            prompt = build_prompt(instr)
            prompt_ids = encode(prompt)[:prompt_len]
            prompt_ids += [vocab[PAD_TOKEN]] * (prompt_len - len(prompt_ids))
            resp_ids = encode(resp)[:(max_len - prompt_len)]
            resp_ids += [vocab[PAD_TOKEN]] * ((max_len - prompt_len) - len(resp_ids))
            self.inputs.append(torch.tensor(prompt_ids, dtype=torch.long))
            self.targets.append(torch.tensor(resp_ids, dtype=torch.long))  # Sadece response target!

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # input: prompt, target: response
        inp = self.inputs[idx]
        tgt = self.targets[idx]
        return inp, tgt

max_len = 256
prompt_len = 64
response_len = max_len - prompt_len
dataset = InstructionResponseDataset(instructions, responses, vocab, max_len=max_len, prompt_len=prompt_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

def add_noise(batch, noise_level=0.5):
    noisy = batch.clone()
    mask = (torch.rand(noisy.shape) < noise_level)
    random_tokens = torch.randint(2, len(vocab), noisy.shape, device=batch.device)
    noisy[mask] = random_tokens[mask]
    return noisy

# Cosine noise schedule (daha iyi diffusion için)
def cosine_noise_schedule(step, total_steps):
    import math
    return math.cos((step / total_steps) * math.pi / 2)

class DiffusionTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=384, hidden_dim=3072, num_layers=12, nhead=12, max_steps=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.timestep_embed = nn.Embedding(max_steps, embedding_dim)
        self.prompt_proj = nn.Linear(embedding_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, prompt, x, timestep, prompt_emb, src_key_padding_mask=None):
        # prompt: (batch, prompt_len), x: (batch, response_len)
        prompt_embs = self.embedding(prompt)
        x_embs = self.embedding(x)
        t_emb = self.timestep_embed(timestep).unsqueeze(1)
        prompt_cond = self.prompt_proj(prompt_emb).unsqueeze(1)
        emb = torch.cat([prompt_embs, x_embs], dim=1) + t_emb + prompt_cond
        # src_key_padding_mask shape düzeltme
        if src_key_padding_mask is not None:
            # src_key_padding_mask: (batch, response_len) -> (batch, prompt_len + response_len)
            pad = torch.zeros((src_key_padding_mask.shape[0], prompt_embs.shape[1]), dtype=torch.bool, device=src_key_padding_mask.device)
            src_key_padding_mask = torch.cat([pad, src_key_padding_mask], dim=1)
        out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
        out = self.fc(out)
        return out[:, prompt_embs.shape[1]:, :]

def get_prompt_embedding(prompt_texts, vocab, model, prompt_len=64):
    batch = []
    for text in prompt_texts:
        ids = encode(text)[:prompt_len]
        ids += [vocab[PAD_TOKEN]] * (prompt_len - len(ids))
        batch.append(ids)
    ids = torch.tensor(batch, dtype=torch.long, device=model.embedding.weight.device)
    with torch.no_grad():
        emb = model.embedding(ids)
        prompt_emb = emb.mean(dim=1)
    return prompt_emb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiffusionTextModel(len(vocab)).to(device)

def train_diffusion_model(model, dataloader, epochs=10, steps=16):
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD_TOKEN])
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_prompts, batch_targets in loop:
            batch_prompts = batch_prompts.to(device)
            batch_targets = batch_targets.to(device)
            step = np.random.randint(0, steps)
            timestep = torch.full((batch_prompts.size(0),), step, dtype=torch.long, device=device)
            prompt_texts = [decode(p.tolist()) for p in batch_prompts]
            prompt_emb = get_prompt_embedding(prompt_texts, vocab, model, prompt_len=prompt_len)
            noisy_targets = add_noise(batch_targets, cosine_noise_schedule(step, steps))
            mask = (batch_targets == vocab[PAD_TOKEN])
            optimizer.zero_grad()
            outputs = model(batch_prompts, noisy_targets, timestep, prompt_emb, src_key_padding_mask=mask)
            # .view yerine .reshape kullan
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), batch_targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

train_diffusion_model(model, dataloader, epochs=8, steps=16)

def generate_response(model, instruction, steps=16, max_len=256, prompt_len=64):
    model.eval()
    prompt = build_prompt(instruction)
    prompt_ids = encode(prompt)[:prompt_len]
    prompt_ids += [vocab[PAD_TOKEN]] * (prompt_len - len(prompt_ids))
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_emb = get_prompt_embedding([prompt], vocab, model, prompt_len=prompt_len)
    # Response kısmı random başlatılır
    response_len = max_len - prompt_len
    response_part = torch.randint(2, len(vocab), (1, response_len), device=device)
    generated = response_part.clone()
    for step in tqdm(range(steps), desc="Diffusion Steps", leave=False):
        mask = (generated == vocab[PAD_TOKEN])
        timestep = torch.full((1,), step, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(prompt_tensor, generated, timestep, prompt_emb, src_key_padding_mask=mask)
            predicted = outputs.argmax(dim=-1)
        prob = cosine_noise_schedule(step, steps)
        random_mask = (torch.rand_like(generated.float()) < prob)
        generated[random_mask] = predicted[random_mask]
    tokens = generated[0].tolist()
    decoded = decode(tokens)
    return decoded

test_instruction = instructions[0]
print('Instruction:', test_instruction)
print('Gerçek Response:', responses[0])
print('Model Response:', generate_response(model, test_instruction, steps=16, max_len=max_len, prompt_len=prompt_len))

test_instruction = "Çift anadal veya yandal yapmak istiyorum. Hangi bölümlerle yapabilirim?"
print('Instruction:', test_instruction)
print('Gerçek Response:', responses[instructions.index(test_instruction)] if test_instruction in instructions else "Yok")
print('Model Response:', generate_response(model, test_instruction, steps=16, max_len=max_len, prompt_len=prompt_len))

def evaluate_diffusion_model(model, dataset, n_samples=100, steps=16, max_len=256, prompt_len=64):
    model.eval()
    total = 0
    correct = 0
    loop = tqdm(range(min(n_samples, len(dataset))), desc="Evaluating", leave=False)
    for i in loop:
        prompt, tgt = dataset[i]
        prompt = prompt.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)
        response_len = max_len - prompt_len
        generated = torch.randint(2, len(vocab), (1, response_len), device=device)
        prompt_texts = [decode(prompt.squeeze(0).tolist())]
        prompt_emb = get_prompt_embedding(prompt_texts, vocab, model, prompt_len=prompt_len)
        for step in range(steps):
            mask = (generated == vocab[PAD_TOKEN])
            timestep = torch.full((1,), step, dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(prompt, generated, timestep, prompt_emb, src_key_padding_mask=mask)
                predicted = outputs.argmax(dim=-1)
            prob = cosine_noise_schedule(step, steps)
            random_mask = (torch.rand_like(generated.float()) < prob)
            generated[random_mask] = predicted[random_mask]
        mask = (tgt != vocab[PAD_TOKEN])
        total += mask.sum().item()
        correct += ((generated == tgt) & mask).sum().item()
        loop.set_postfix(acc=(correct/total if total > 0 else 0.0))
    accuracy = correct / total if total > 0 else 0.0
    print(f"Test doğruluğu: {accuracy:.2%} ({correct}/{total})")

evaluate_diffusion_model(model, dataset, n_samples=100, steps=16, max_len=max_len, prompt_len=prompt_len)