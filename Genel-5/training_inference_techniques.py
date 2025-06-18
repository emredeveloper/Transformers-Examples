import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple
import time
from dataclasses import dataclass

# 1. ADVANCED TRAINING TECHNIQUES

class CosineScheduler:
    """
    Cosine Learning Rate Scheduling - Modern optimization
    - Warmup + Cosine decay
    - Llama ve GPT-4'te kullanılır
    """
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, 
                 min_lr: float = 0.0, max_lr: float = 1e-4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class GradientClipper:
    """
    Gradient Clipping - Training stability
    - Global norm clipping
    - Exploding gradient problemini çözer
    """
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip_gradients(self, model: nn.Module) -> float:
        # Global gradient norm hesapla
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return total_norm

class ModernTrainer:
    """
    Modern LLM Training Pipeline
    - Cosine scheduling
    - Gradient clipping
    - Mixed precision
    - Gradient accumulation
    """
    def __init__(self, model: nn.Module, train_loader, 
                 max_lr: float = 1e-4, weight_decay: float = 0.1,
                 warmup_steps: int = 2000, max_steps: int = 100000,
                 grad_accum_steps: int = 4):
        
        self.model = model
        self.train_loader = train_loader
        self.grad_accum_steps = grad_accum_steps
        
        # Optimizer (AdamW with weight decay)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=max_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),  # Llama betas
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = CosineScheduler(
            self.optimizer, warmup_steps, max_steps, 
            min_lr=max_lr * 0.1, max_lr=max_lr
        )
        
        # Gradient clipper
        self.clipper = GradientClipper(max_norm=1.0)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.step_count = 0
    
    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Single training step with all modern techniques"""
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            logits, loss = self.model(tokens, targets)
            loss = loss / self.grad_accum_steps  # Scale for accumulation
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        metrics = {'loss': loss.item() * self.grad_accum_steps}
        
        # Gradient accumulation
        if (self.step_count + 1) % self.grad_accum_steps == 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = self.clipper.clip_gradients(self.model)
            metrics['grad_norm'] = grad_norm
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Learning rate scheduling
            lr = self.scheduler.step()
            metrics['lr'] = lr
        
        self.step_count += 1
        return metrics

# 2. KV CACHING for Fast Inference

@dataclass
class KVCache:
    """Key-Value cache for efficient generation"""
    k: torch.Tensor
    v: torch.Tensor
    
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor, 
               start_pos: int) -> 'KVCache':
        """Update cache with new key-value pairs"""
        seq_len = new_k.size(2)
        self.k[:, :, start_pos:start_pos + seq_len] = new_k
        self.v[:, :, start_pos:start_pos + seq_len] = new_v
        return self

class CachedAttention(nn.Module):
    """Attention with KV caching for fast generation"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor, start_pos: int, 
                kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        bsz, seq_len, _ = x.shape
        
        # Q, K, V projections
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (bsz, n_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)
        
        # Update KV cache
        if kv_cache is None:
            # Initialize cache
            cache_k = torch.zeros(bsz, self.n_kv_heads, self.max_seq_len, self.head_dim, 
                                device=x.device, dtype=x.dtype)
            cache_v = torch.zeros(bsz, self.n_kv_heads, self.max_seq_len, self.head_dim,
                                device=x.device, dtype=x.dtype)
            kv_cache = KVCache(cache_k, cache_v)
        
        kv_cache.update(k, v, start_pos)
        
        # Use cached K, V for attention
        keys = kv_cache.k[:, :, :start_pos + seq_len]
        values = kv_cache.v[:, :, :start_pos + seq_len]
        
        # Repeat K, V for grouped attention
        keys = keys.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        values = values.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        # Compute attention
        scores = torch.matmul(q, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask (only for new positions)
        if seq_len > 1:
            mask = torch.tril(torch.ones(seq_len, start_pos + seq_len, device=x.device))
            scores = scores.masked_fill(mask[-seq_len:] == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, values)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out), kv_cache

# 3. ADVANCED SAMPLING TECHNIQUES

class NucleusSampler:
    """
    Nucleus (Top-p) Sampling
    - Dinamik vocabulary filtering
    - Temperature scaling
    - Repetition penalty
    """
    def __init__(self, temperature: float = 1.0, top_p: float = 0.9, 
                 top_k: int = 50, repetition_penalty: float = 1.0):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
    
    def sample(self, logits: torch.Tensor, 
               previous_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advanced sampling with multiple techniques"""
        
        # Repetition penalty
        if previous_tokens is not None and self.repetition_penalty != 1.0:
            for token in set(previous_tokens.tolist()):
                logits[token] /= self.repetition_penalty
        
        # Temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Top-k filtering
        if self.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(0, top_k_indices, top_k_logits)
            logits = logits_filtered
        
        # Nucleus (top-p) sampling
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Find cutoff
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            # Remove tokens
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # Sample from filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

class FastGenerator:
    """
    Fast text generation with KV caching
    - Efficient inference
    - Advanced sampling
    - Generation metrics
    """
    def __init__(self, model: nn.Module, tokenizer, max_seq_len: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sampler = NucleusSampler()
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100, 
                 **sampling_kwargs) -> Dict:
        """Generate text with performance metrics"""
        
        # Update sampler parameters
        for key, value in sampling_kwargs.items():
            if hasattr(self.sampler, key):
                setattr(self.sampler, key, value)
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        start_time = time.time()
        generated_tokens = []
        kv_caches = [None] * len(self.model.layers)
        
        # Generate tokens one by one
        for step in range(max_new_tokens):
            start_pos = tokens.size(1) - 1 if step > 0 else 0
            current_token = tokens[:, -1:] if step > 0 else tokens
            
            # Forward pass with caching
            x = self.model.tok_embeddings(current_token)
            
            for i, layer in enumerate(self.model.layers):
                # Use cached attention if available
                if hasattr(layer.attention, 'forward_cached'):
                    x, kv_caches[i] = layer.attention.forward_cached(x, start_pos, kv_caches[i])
                else:
                    x = layer(x)
            
            x = self.model.norm(x)
            logits = self.model.output(x[:, -1, :])  # Only last token
            
            # Sample next token
            next_token = self.sampler.sample(logits[0], tokens[0])
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())
            
            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens)
        total_time = time.time() - start_time
        
        return {
            'text': generated_text,
            'tokens_generated': len(generated_tokens),
            'generation_time': total_time,
            'tokens_per_second': len(generated_tokens) / total_time if total_time > 0 else 0,
            'total_tokens': tokens.size(1)
        }

# 4. EVALUATION METRICS

class ModelEvaluator:
    """
    Comprehensive model evaluation
    - Perplexity calculation
    - Generation quality metrics
    - Performance benchmarking
    """
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def calculate_perplexity(self, text_samples: List[str]) -> float:
        """Calculate perplexity on text samples"""
        total_loss = 0.0
        total_tokens = 0
        
        self.model.eval()
        
        for text in text_samples:
            tokens = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
            
            if tokens.size(1) < 2:
                continue
            
            # Forward pass
            logits, loss = self.model(tokens[:, :-1], tokens[:, 1:])
            
            total_loss += loss.item() * (tokens.size(1) - 1)
            total_tokens += tokens.size(1) - 1
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def benchmark_generation_speed(self, prompts: List[str], max_new_tokens: int = 50) -> Dict:
        """Benchmark generation performance"""
        generator = FastGenerator(self.model, self.tokenizer)
        
        times = []
        tokens_per_sec = []
        
        for prompt in prompts:
            result = generator.generate(prompt, max_new_tokens=max_new_tokens)
            times.append(result['generation_time'])
            tokens_per_sec.append(result['tokens_per_second'])
        
        return {
            'avg_generation_time': sum(times) / len(times),
            'avg_tokens_per_second': sum(tokens_per_sec) / len(tokens_per_sec),
            'min_tokens_per_second': min(tokens_per_sec),
            'max_tokens_per_second': max(tokens_per_sec),
            'total_prompts': len(prompts)
        }

# 5. MEMORY OPTIMIZATION TECHNIQUES

class MemoryTracker:
    """GPU memory usage tracking"""
    def __init__(self):
        self.baseline_memory = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.baseline_memory = torch.cuda.memory_allocated()
    
    def get_memory_usage(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {'gpu_memory_mb': 0}
        
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        
        return {
            'current_memory_mb': (current_memory - self.baseline_memory) / 1024 / 1024,
            'max_memory_mb': (max_memory - self.baseline_memory) / 1024 / 1024,
            'memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024
        }

class GradientCheckpointing:
    """
    Gradient checkpointing for memory efficiency
    - Trade computation for memory
    - Enables training larger models
    """
    @staticmethod
    def checkpoint_sequential(functions, segments, *inputs):
        """Checkpoint a sequential model"""
        def run_function(start, end, functions):
            def forward(*inputs):
                for j in range(start, end + 1):
                    inputs = functions[j](*inputs)
                return inputs
            return forward
        
        if len(functions) == 1:
            return functions[0](*inputs)
        
        # Divide functions into segments
        segment_size = len(functions) // segments
        
        # Run first segment normally
        outputs = inputs
        for i in range(0, segment_size):
            outputs = functions[i](*outputs if isinstance(outputs, tuple) else (outputs,))
        
        # Checkpoint remaining segments
        for i in range(segments - 1):
            start_idx = (i + 1) * segment_size
            end_idx = min(start_idx + segment_size - 1, len(functions) - 1)
            segment_func = run_function(start_idx, end_idx, functions)
            outputs = torch.utils.checkpoint.checkpoint(
                segment_func, 
                *outputs if isinstance(outputs, tuple) else (outputs,)
            )
        
        return outputs

# 6. ADVANCED TOKENIZATION

class BPETokenizer:
    """
    Modern BPE tokenizer implementation
    - Subword tokenization
    - Special token handling
    - Instruction tuning ready
    """
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<s>': 2,      # BOS (Beginning of Sequence)
            '</s>': 3,     # EOS (End of Sequence)
            '<inst>': 4,   # Instruction start
            '</inst>': 5,  # Instruction end
            '<sys>': 6,    # System message
            '</sys>': 7,   # System message end
        }
        self.bos_token_id = self.special_tokens['<s>']
        self.eos_token_id = self.special_tokens['</s>']
        self.pad_token_id = self.special_tokens['<pad>']
        
        # Mock vocabulary for demonstration
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token ids"""
        # Simplified encoding (in practice, use real BPE)
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        # Mock tokenization - split by spaces
        words = text.split()
        for word in words:
            # In real BPE, this would be subword tokenization
            token_id = self.token_to_id.get(word, self.special_tokens['<unk>'])
            tokens.append(token_id)
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['<pad>', '<s>', '</s>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def encode_instruction(self, instruction: str, input_text: str = "", 
                          system_message: str = "") -> List[int]:
        """Encode instruction-following format"""
        tokens = [self.bos_token_id]
        
        if system_message:
            tokens.extend([self.special_tokens['<sys>']])
            tokens.extend(self.encode(system_message, add_bos=False))
            tokens.extend([self.special_tokens['</sys>']])
        
        tokens.extend([self.special_tokens['<inst>']])
        tokens.extend(self.encode(instruction, add_bos=False))
        
        if input_text:
            tokens.extend(self.encode(input_text, add_bos=False))
        
        tokens.extend([self.special_tokens['</inst>']])
        
        return tokens

# 7. INSTRUCTION TUNING DATASET

class InstructionDataset(torch.utils.data.Dataset):
    """
    Dataset for instruction tuning
    - Supports various instruction formats
    - Proper loss masking
    - ChatGPT-style conversations
    """
    def __init__(self, data: List[Dict], tokenizer: BPETokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if 'conversations' in item:
            # Multi-turn conversation format
            return self._process_conversation(item['conversations'])
        else:
            # Single instruction format
            return self._process_instruction(item)
    
    def _process_instruction(self, item: Dict):
        """Process single instruction-response pair"""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        response = item.get('output', '')
        system = item.get('system', '')
        
        # Encode instruction part
        input_ids = self.tokenizer.encode_instruction(instruction, input_text, system)
        
        # Encode response
        response_ids = self.tokenizer.encode(response, add_bos=False, add_eos=True)
        
        # Combine
        full_ids = input_ids + response_ids
        
        # Create labels (mask instruction part)
        labels = [-100] * len(input_ids) + response_ids
        
        # Truncate if necessary
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # Pad to max length
        while len(full_ids) < self.max_length:
            full_ids.append(self.tokenizer.pad_token_id)
            labels.append(-100)
        
        return torch.tensor(full_ids), torch.tensor(labels)
    
    def _process_conversation(self, conversations: List[Dict]):
        """Process multi-turn conversation"""
        full_ids = [self.tokenizer.bos_token_id]
        labels = [-100]
        
        for turn in conversations:
            role = turn['role']  # 'user' or 'assistant'
            content = turn['content']
            
            if role == 'user':
                # User input - mask in loss
                tokens = self.tokenizer.encode(f"User: {content}", add_bos=False)
                full_ids.extend(tokens)
                labels.extend([-100] * len(tokens))
            else:
                # Assistant response - include in loss
                tokens = self.tokenizer.encode(f"Assistant: {content}", add_bos=False)
                full_ids.extend(tokens)
                labels.extend(tokens)
        
        # Add EOS
        full_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        
        # Truncate and pad
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        while len(full_ids) < self.max_length:
            full_ids.append(self.tokenizer.pad_token_id)
            labels.append(-100)
        
        return torch.tensor(full_ids), torch.tensor(labels)

# 8. COMPLETE TRAINING PIPELINE

class InstructionTuner:
    """
    Complete instruction tuning pipeline
    - Modern training techniques
    - Proper evaluation
    - Memory optimization
    """
    def __init__(self, model: nn.Module, tokenizer: BPETokenizer,
                 train_data: List[Dict], eval_data: List[Dict],
                 batch_size: int = 4, max_length: int = 2048):
        
        self.model = model
        self.tokenizer = tokenizer
        self.memory_tracker = MemoryTracker()
        
        # Datasets
        self.train_dataset = InstructionDataset(train_data, tokenizer, max_length)
        self.eval_dataset = InstructionDataset(eval_data, tokenizer, max_length)
        
        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        self.eval_loader = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        # Trainer
        self.trainer = ModernTrainer(
            model, self.train_loader,
            max_lr=2e-5,  # Lower LR for instruction tuning
            warmup_steps=100,
            max_steps=len(self.train_loader) * 3,  # 3 epochs
            grad_accum_steps=4
        )
        
        # Evaluator
        self.evaluator = ModelEvaluator(model, tokenizer)
    
    def train(self, num_epochs: int = 3):
        """Full training loop with evaluation"""
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            for batch_idx, (input_ids, labels) in enumerate(self.train_loader):
                # Move to device
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    labels = labels.cuda()
                
                # Training step
                metrics = self.trainer.train_step(input_ids, labels)
                epoch_losses.append(metrics['loss'])
                
                # Logging
                if batch_idx % 10 == 0:
                    memory_info = self.memory_tracker.get_memory_usage()
                    print(f"Batch {batch_idx:4d} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"LR: {metrics.get('lr', 0):.2e} | "
                          f"Memory: {memory_info['current_memory_mb']:.1f}MB")
            
            # Evaluation
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            eval_metrics = self.evaluate()
            
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"Train Loss: {avg_loss:.4f}")
            print(f"Eval Perplexity: {eval_metrics['perplexity']:.2f}")
            print(f"Generation Speed: {eval_metrics['tokens_per_second']:.1f} tok/s")
    
    def evaluate(self) -> Dict:
        """Comprehensive evaluation"""
        self.model.eval()
        
        # Perplexity evaluation
        eval_texts = []
        with torch.no_grad():
            for input_ids, labels in self.eval_loader:
                # Convert back to text for perplexity calculation
                for i in range(input_ids.size(0)):
                    text = self.tokenizer.decode(input_ids[i].tolist())
                    eval_texts.append(text)
        
        perplexity = self.evaluator.calculate_perplexity(eval_texts[:100])  # Sample
        
        # Generation speed benchmark
        test_prompts = [
            "Explain the concept of machine learning",
            "Write a short story about a robot",
            "What are the benefits of renewable energy?"
        ]
        speed_metrics = self.evaluator.benchmark_generation_speed(test_prompts)
        
        return {
            'perplexity': perplexity,
            'tokens_per_second': speed_metrics['avg_tokens_per_second'],
            'generation_time': speed_metrics['avg_generation_time']
        }

# USAGE EXAMPLE
if __name__ == "__main__":
    # Initialize components
    tokenizer = BPETokenizer(vocab_size=32000)
    
    # Sample instruction data
    train_data = [
        {
            "instruction": "Explain what artificial intelligence is",
            "input": "",
            "output": "Artificial intelligence (AI) is a branch of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."
        },
        {
            "conversations": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."}
            ]
        }
    ]
    
    eval_data = train_data[:1]  # Small eval set for demo
    
    # Initialize model (using our modern architecture)
    from modern_llm_components import ModernLLM
    model = ModernLLM(
        vocab_size=32000,
        dim=1024,  # Smaller for demo
        n_layers=12,
        n_heads=16,
        n_kv_heads=4
    )
    
    # Initialize tuner
    tuner = InstructionTuner(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        batch_size=1,  # Small batch for demo
        max_length=512
    )
    
    print("Modern LLM Training Pipeline Initialized!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")
    
    # Uncomment to run training
    # tuner.train(num_epochs=1)