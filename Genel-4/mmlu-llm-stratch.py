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
# 1. TOKENIZER - Convert text into numerical features
# =============================================================================

class SimpleTokenizer:
    """Improved tokenizer implementation"""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.pad_token_id = 0  # Assign 0 to the PAD token ID
        self.unk_token_id = 1  # ID value reserved for UNK
        self.bos_token_id = 2  # ID for the BOS token
        self.eos_token_id = 3  # ID for the EOS token

    def fit(self, texts: List[str]):
        """Build the vocabulary from raw text"""
        # Collect every character and compute frequencies
        char_freq = {}
        for text in texts:
            for char in text:
                if char not in ['\n', ' ']:  # Separate spaces and newlines from the other tokens
                    char_freq[char] = char_freq.get(char, 0) + 1

        # Special tokens and the most common characters
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token, '\n', ' ']

        # Select the 200 most frequent characters (excluding the special tokens)
        common_chars = [char for char, _ in sorted(char_freq.items(), key=lambda x: -x[1])[:200]]

        # Combine the special tokens and the frequent characters
        all_chars = special_tokens + common_chars

        # Build a list of unique characters
        unique_chars = []
        for char in all_chars:
            if char not in unique_chars:
                unique_chars.append(char)

        # Create lookup tables
        self.char_to_id = {char: i for i, char in enumerate(unique_chars)}
        self.id_to_char = {i: char for i, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

        # Update the special token IDs
        self.pad_token_id = self.char_to_id[self.pad_token]
        self.unk_token_id = self.char_to_id[self.unk_token]
        self.bos_token_id = self.char_to_id[self.bos_token]
        self.eos_token_id = self.char_to_id[self.eos_token]

        # Build the final vocabulary list (special tokens + frequent characters)
        vocab = special_tokens + common_chars

        # Re-create the mappings so indices align with the final list
        self.char_to_id = {char: i for i, char in enumerate(vocab)}
        self.id_to_char = {i: char for i, char in enumerate(vocab)}
        self.vocab_size = len(vocab)

        # Persist the special token IDs
        self.pad_token_id = self.char_to_id.get('<PAD>', 0)
        self.unk_token_id = self.char_to_id.get('<UNK>', 1)
        self.bos_token_id = self.char_to_id.get('<BOS>', 2)
        self.eos_token_id = self.char_to_id.get('<EOS>', 3)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"First 20 tokens: {vocab[:20]}")

    def encode(self, text: str, max_length: int = 512, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert raw text into token IDs"""
        # Add the optional special tokens
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)

        # Map characters to token IDs
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_token_id))

        if add_eos:
            tokens.append(self.eos_token_id)

        # Constrain the sequence length
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [tokens[-1]]  # Preserve the last token
        elif len(tokens) < max_length:
            # Apply padding
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back into readable text"""
        chars = []
        for token_id in token_ids:
            if token_id == self.char_to_id['<EOS>']:
                break
            if token_id != self.char_to_id['<PAD>']:
                chars.append(self.id_to_char.get(token_id, '<UNK>'))
        return ''.join(chars)

# =============================================================================
# 2. TRANSFORMER COMPONENTS - Attention and FFN layers
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer"""
    
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
        
        # Compute Q, K, V projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply a causal mask so the model cannot peek ahead
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply the attention weights
        context = torch.matmul(attention_weights, V)
        
        # Reshape back and project to the model dimension
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
    """Transformer decoder block"""
    
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
# 3. LLM MODEL - Core transformer architecture
# =============================================================================

class SimpleLLM(nn.Module):
    """Lightweight large language model"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialise parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create a causal mask so the model cannot look ahead"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Build position IDs
        position_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        
        # Build the attention mask
        # Shape should be (batch_size, 1, 1, seq_len)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        
        # Convert mask values to float and suppress masked positions
        # Ones stay visible while zeros are masked out
        mask = (1.0 - mask.float()) * -1e9
        
        # Transformer layers
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
            
        # Language model head
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9, pad_token_id=None):
        """
        Text generation helper.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_length: Maximum number of tokens to extend the sequence
            temperature: Lower values produce more predictable outputs
            top_k: Top-k sampling threshold
            top_p: Nucleus sampling probability threshold
            pad_token_id: Padding token ID

        Returns:
            Generated token IDs with the continuation appended (batch_size, seq_len + max_length)
        """
        device = next(self.parameters()).device
        batch_size = input_ids.size(0)
        
        # Move the inputs to the target device
        input_ids = input_ids.to(device)
        
        # Start the output with the provided prompt
        generated = input_ids
        
        # Disable training mode
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Build a mask for the current sequence
                seq_len = generated.size(1)
                attn_mask = (generated != pad_token_id).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
                
                # Run the model
                outputs = self(generated, mask=attn_mask)
                
                # Focus on the logits for the last token
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    # Remove tokens outside the top-k candidates
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Identify the smallest indices whose cumulative probability exceeds p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep the first index (highest probability token)
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Suppress logits that fall outside the nucleus
                    sorted_logits[sorted_indices_to_remove] = -float('Inf')
                    
                    # Restore the original order
                    next_token_logits = torch.zeros_like(next_token_logits).scatter_(
                        dim=1, index=sorted_indices, src=sorted_logits
                    )
                
                # Sample the next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Fall back to the pad_token_id if every probability is -inf
                if pad_token_id is not None and torch.all(torch.isinf(probs)):
                    next_tokens = torch.full_like(next_tokens, pad_token_id)
                
                # Append the generated token to the output
                generated = torch.cat((generated, next_tokens), dim=1)
                
                # Exit early if every sequence has finished
                if pad_token_id is not None and torch.all(next_tokens == pad_token_id):
                    break
        
        return generated

# =============================================================================
# 4. DATASET - Prepare training data
# =============================================================================

def build_vocab_from_csv(csv_path: str) -> List[str]:
    """Create a character-level vocabulary from the CSV file"""
    df = pd.read_csv(csv_path)
    all_text = ""
    
    # Concatenate all text
    for _, row in df.iterrows():
        all_text += row['Question'] + " "
        all_text += row['A'] + " " + row['B'] + " " + row['C'] + " " + row['D'] + " "
    
    # Collect and sort unique characters
    unique_chars = sorted(list(set(all_text)))
    return unique_chars

class MMLUDataset(Dataset):
    """MMLU dataset wrapper"""
    
    def __init__(self, csv_path: str, tokenizer: SimpleTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Gather every prompt text
        all_texts = []
        for _, row in df.iterrows():
            question = row['Question']
            options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
            prompt = f"Question: {question}\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\nAnswer:"
            all_texts.append(prompt)
        
        # Train the tokenizer
        self.tokenizer.fit(all_texts)
        
        # Build the dataset entries
        for i, row in df.iterrows():
            question = row['Question']
            options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
            answer = row['Answer']
            
            # Prompt format: "Question: [question]\nA) [A]\nB) [B]\nC) [C]\nD) [D]\nAnswer:"
            prompt = f"Question: {question}\n"
            for i, opt in enumerate(['A', 'B', 'C', 'D']):
                prompt += f"{opt}) {options[i]}\n"
            prompt += "Answer:"
            
            # Encode the answer choice as a token (A->0, B->1, C->2, D->3)
            target = ord(str(answer).strip().upper()[0]) - ord('A')
            
            # Tokenize and store
            tokens = self.tokenizer.encode(prompt, max_length)
            if len(tokens) > 0 and 0 <= target <= 3:  # Only accept valid targets
                self.data.append((tokens, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens, target = self.data[idx]
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)
        return x, y

# =============================================================================
# 5. TRAINING - Model training
# =============================================================================

class LLMTrainer:
    """LLM training helper"""
    
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
        
        # Warmup configuration
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for a single epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Consider only the final token (answer position)
            last_token_logits = outputs[:, -1, :]  # [batch_size, vocab_size]
            
            # Compute the loss
            loss = self.criterion(last_token_logits, targets)
            
            # Count correct predictions
            _, predicted = torch.max(last_token_logits, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Backpropagation and optimisation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimiser step
            self.optimizer.step()
            
            # Learning-rate warmup and scheduling
            self.current_step += 1
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr_scale = min(1.0, float(self.current_step) / self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.optimizer.defaults['lr'] * lr_scale
            else:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                batch_acc = (predicted == targets).float().mean().item() * 100
                print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%")
        
        accuracy = 100 * correct / total if total > 0 else 0
        return total_loss / len(dataloader), accuracy
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text"""
        self.model.eval()
        
        # Tokenise the prompt
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
                
                # Stop if the EOS token is produced
                if next_token == self.tokenizer.char_to_id['<EOS>']:
                    break
                
                # Append the new token
                generated_tokens.append(next_token)
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                
                # Enforce the maximum sequence length
                if input_ids.size(1) >= self.model.max_seq_len:
                    input_ids = input_ids[:, 1:]  # Drop the first token to stay within limits
        
        return self.tokenizer.decode(generated_tokens)

# =============================================================================
# 6. MAIN - Entry point
# =============================================================================

def main():
    # Hyper-parameters
    batch_size = 16
    max_length = 512  # Allow longer sequences
    d_model = 512  # Larger model width
    n_heads = 8  # Increased number of heads
    n_layers = 6  # Deeper stack
    d_ff = 2048  # Wider feed-forward network
    dropout = 0.1  
    num_epochs = 1  # Set to 1 for quick experimentation
    max_examples = 1000
    learning_rate = 3e-4
    weight_decay = 0.01
    warmup_steps = 1000  # Warmup steps for the learning rate
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CSV file path
    csv_path = r"c:\Users\emreq\Downloads\archive\mmlu.csv"
    
    # Build the tokenizer
    tokenizer = SimpleTokenizer()
    
    # Load the MMLU dataset
    print("Loading dataset...")
    dataset = MMLUDataset(csv_path, tokenizer, max_length)
    
    # Guard against an empty dataset
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the CSV file and structure.")
    
    # Limit the dataset to 1000 examples
    if len(dataset) > max_examples:
        print(f"Truncating dataset from {len(dataset)} to {max_examples} examples...")
        indices = torch.randperm(len(dataset))[:max_examples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Split into train and test sets (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Build the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the model
    print("Building model...")
    model = SimpleLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_length,
        dropout=dropout
    ).to(device)
    
    # Total number of steps for the learning-rate scheduler
    total_steps = len(train_dataloader) * num_epochs
    
    # Instantiate the trainer
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # Training loop
    print(f"Training starting... {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    
    best_test_acc = 0
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = trainer.train_epoch(train_dataloader)
        
        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                last_token_logits = outputs[:, -1, :]
                
                # Compute the loss
                loss = F.cross_entropy(last_token_logits, targets)
                test_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(last_token_logits, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        test_loss = test_loss / len(test_dataloader)
        test_acc = 100 * correct / total if total > 0 else 0
        
        # Save the best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Persist only the required information
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': test_acc,
                'tokenizer_chars': tokenizer.char_to_id,  # Persist only the character-to-id mapping
                'vocab_size': tokenizer.vocab_size
            }, 'best_model.pt')
            print(f"New best model saved! Test accuracy: {test_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
    
    # Evaluation etme
    print("\n=== Final Test ===\n")
    
    # Load the best checkpoint if it exists
    if os.path.exists('best_model.pt'):
        checkpoint = torch.load('best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model (Accuracy: {checkpoint['accuracy']:.2f}%)\n")
    
    model.eval()
    
    # Example test questions with expected answers
    test_questions = [
        {
            "question": "Question: In which year was the conquest of Istanbul?\nA) 1451\nB) 1453\nC) 1455\nD) 1457\nAnswer:",
            "correct": "B"
        },
        {
            "question": "Question: Who created the Python programming language?\nA) Guido van Rossum\nB) James Gosling\nC) Bjarne Stroustrup\nD) Dennis Ritchie\nAnswer:",
            "correct": "A"
        },
        {
            "question": "Question: Which is the largest ocean in the world?\nA) Atlantic Ocean\nB) Indian Ocean\nC) Arctic Ocean\nD) Pacific Ocean\nAnswer:",
            "correct": "D"
        },
        {
            "question": "Question: Which of the following is an artificial intelligence library?\nA) React\nB) TensorFlow\nC) Django\nD) Flask\nAnswer:",
            "correct": "B"
        }
    ]
    
    correct_answers = 0
    
    for i, item in enumerate(test_questions, 1):
        question = item["question"]
        correct = item["correct"]
        
        print(f"\n--- Test {i} ---")
        print("Question:")
        print(question)
        
        # Modelden cevap al
        with torch.no_grad():
            # Tokenise only the question text
            question_tokens = tokenizer.encode(question, max_length=512, add_bos=True, add_eos=False)
            input_tensor = torch.tensor([question_tokens], device=device)
            
            # Generate an answer
            output = model.generate(
                input_tensor,
                max_length=len(question_tokens) + 5,  # Five extra tokens are usually enough for the answer
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode the full output
            full_output = tokenizer.decode(output[0].tolist())
            
            # Inspect the final 5 tokens (the answer is usually near the end)
            last_tokens = output[0][-5:].tolist()
            last_chars = tokenizer.decode(last_tokens)
            
            # Search for an answer choice (A, B, C, or D)
            answer = None
            for c in last_chars.upper():
                if c in ['A', 'B', 'C', 'D']:
                    answer = c
                    break
            
            if answer is None:
                answer = "(Answer not found)"
            
            # Compare against the expected answer
            is_correct = (answer == correct)
            if is_correct:
                correct_answers += 1
            
            print("\nModel Answer:", answer)
            print("Correct Answer:", correct)
            print("Result:", "✅ Correct" if is_correct else "❌ Incorrect")
            print("\nFull Output:", full_output)
        
        print("\n" + "="*80)
    
    # Report the overall success rate
    accuracy = (correct_answers / len(test_questions)) * 100
    print(f"\n=== Test Results ===")
    print(f"Correct Answers: {correct_answers}/{len(test_questions)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print("\n=== Training and evaluation complete! ===")
    
    return model, tokenizer, trainer

if __name__ == "__main__":
    model, tokenizer, trainer = main()