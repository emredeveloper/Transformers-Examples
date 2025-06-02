import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

@dataclass
class ModelConfig:
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    window_size: int = 512  # For sliding window attention
    max_seq_length: int = 2048

class KeyValueCache:
    """Implementation of KV Cache for efficient inference"""
    
    def __init__(self, batch_size, max_seq_length, num_layers, hidden_size, num_heads):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Initialize empty cache - shape: [batch, layers, seq_len, heads, head_dim]
        self.keys = torch.zeros(batch_size, num_layers, max_seq_length, num_heads, self.head_dim)
        self.values = torch.zeros(batch_size, num_layers, max_seq_length, num_heads, self.head_dim)
        self.current_length = 0
    
    def update(self, layer_idx, key, value, position):
        """Update the cache with new key-value pairs at a specific position"""
        batch_size, num_heads, seq_len, head_dim = key.size()
        
        if position + seq_len > self.max_seq_length:
            # Shift the cache if we're exceeding max length
            shift = position + seq_len - self.max_seq_length
            self.keys[:, :, :-shift] = self.keys[:, :, shift:]
            self.values[:, :, :-shift] = self.values[:, :, shift:]
            position = self.max_seq_length - seq_len
        
        # Store keys and values with the right dimensions
        # Permute from [batch, heads, seq, dim] to [batch, seq, heads, dim]
        keys_for_cache = key.permute(0, 2, 1, 3)
        values_for_cache = value.permute(0, 2, 1, 3)
        
        self.keys[:, layer_idx, position:position+seq_len] = keys_for_cache
        self.values[:, layer_idx, position:position+seq_len] = values_for_cache
        self.current_length = max(self.current_length, position + seq_len)
    
    def get(self, layer_idx, position):
        """Get cached keys and values for a specific position"""
        if position > self.current_length:
            return None, None
            
        # Get the cached keys and values up to the current position
        cached_keys = self.keys[:, layer_idx, :position]
        cached_values = self.values[:, layer_idx, :position]
        
        # Permute back from [batch, seq, heads, dim] to [batch, heads, seq, dim]
        # which is the format expected by the attention module
        return (
            cached_keys.permute(0, 2, 1, 3),
            cached_values.permute(0, 2, 1, 3)
        )

class HybridAttention(nn.Module):
    """ 
    Implements a hybrid attention mechanism combining sliding window 
    and full attention approaches
    """ 
    def __init__(self, hidden_size, num_heads, window_size, is_sliding_window=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.is_sliding_window = is_sliding_window  # Toggle between sliding window and full attention
        
        # Projection matrices
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Scaling factor for dot product attention
        self.scale = self.head_dim ** -0.5

    def forward(self, x, kv_cache=None, layer_idx=None, use_cache=False):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Rearrange for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Update KV cache if used
        if use_cache and kv_cache is not None:
            # Store the current keys and values
            kv_cache.update(layer_idx, k, v, kv_cache.current_length)
            # Get all cached keys and values
            cached_k, cached_v = kv_cache.get(layer_idx, kv_cache.current_length)
            if cached_k is not None and cached_v is not None:
                k, v = cached_k, cached_v
        
        # Compute attention scores
        if self.is_sliding_window:
            # Sliding window attention - only attends to nearby tokens
            attn_output = self._sliding_window_attention(q, k, v)
        else:
            # Full attention - attends to all tokens
            attn_output = self._full_attention(q, k, v)
            
        # Project back to hidden size
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output

    def _sliding_window_attention(self, q, k, v):
        batch_size, num_heads, seq_len, head_dim = q.shape
        total_length = k.size(2)
        
        # For each position, only attend to a window of tokens
        outputs = []
        for i in range(seq_len):
            # Define window boundaries
            window_start = max(0, i - self.window_size // 2)
            window_end = min(total_length, i + self.window_size // 2 + 1)
            
            # Extract window keys and values
            window_k = k[:, :, window_start:window_end]
            window_v = v[:, :, window_start:window_end]
            
            # Compute attention scores for current position
            q_i = q[:, :, i:i+1]  # (batch_size, num_heads, 1, head_dim)
            attn_scores = torch.matmul(q_i, window_k.transpose(-1, -2)) * self.scale
            
            # Apply softmax to get attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Apply attention weights to values
            pos_output = torch.matmul(attn_weights, window_v)
            outputs.append(pos_output)
        
        # Concatenate outputs for all positions
        return torch.cat(outputs, dim=2)

    def _full_attention(self, q, k, v):
        # Standard scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v)

class TransformerLayer(nn.Module):
    """A single transformer layer with hybrid attention mechanism"""
        
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Determine if this layer uses sliding window or full attention
        # Interleave sliding window and full attention layers
        self.attention = HybridAttention(
            config.hidden_size, 
            config.num_heads, 
            config.window_size,
            is_sliding_window=(layer_idx % 2 == 0)  # Even layers use sliding window
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, kv_cache=None, use_cache=False):
        # Self-attention block
        residual = x
        x = self.ln1(x)
        x = self.attention(x, kv_cache, self.layer_idx, use_cache)
        x = x + residual
        
        # Feed-forward block
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x

class HybridTransformerModel(nn.Module):
    """Complete transformer model with hybrid attention mechanism"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config, i) for i in range(config.num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, use_cache=False, past_kv_cache=None):
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        x = self.embeddings(input_ids)
        
        # Add position embeddings - calculate correct positions
        if past_kv_cache is not None:
            # If we're continuing from a past state, offset positions
            positions = torch.arange(past_kv_cache.current_length, 
                                    past_kv_cache.current_length + seq_len,
                                    device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        else:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        x = x + self.position_embeddings(positions)
        
        # Use provided KV cache or create a new one if needed
        kv_cache = past_kv_cache
        if use_cache and kv_cache is None:
            kv_cache = KeyValueCache(
                batch_size, 
                self.config.max_seq_length,
                self.config.num_layers,
                self.config.hidden_size,
                self.config.num_heads
            )
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, kv_cache, use_cache)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits

class CoPGTrainer:
    """
    Implements Cooperative Preference Optimization for training language models
    based on human feedback
    """
    
    def __init__(self, model, optimizer, human_preferences_dataset):
        self.model = model
        self.optimizer = optimizer
        self.human_preferences_dataset = human_preferences_dataset
        
        # Loss scaling factors
        self.preference_weight = 1.0
        self.kl_weight = 0.1  # KL divergence from original model
        
        # Reference model (frozen copy of the initial model)
        self.reference_model = self._create_reference_model(model)

    def _create_reference_model(self, model):
        """Create a frozen copy of the model to serve as reference"""
        reference = type(model)(model.config)
        reference.load_state_dict(model.state_dict())
        for param in reference.parameters():
            param.requires_grad = False
        return reference

    def generate_responses(self, prompt, num_responses=2):
        """Generate multiple responses for a given prompt"""
        responses = []
        for _ in range(num_responses):
            # Simple greedy decoding for demonstration
            input_ids = self._tokenize(prompt)
            output_ids = self._generate(input_ids)
            response = self._detokenize(output_ids)
            responses.append(response)
        return responses

    def _tokenize(self, text):
        """Mock tokenization function"""
        # In a real implementation, this would use a tokenizer
        return torch.randint(0, self.model.config.vocab_size, (1, 50))
    
    def _detokenize(self, ids):
        """Simulate detokenization function with more realistic output"""
        # In a real implementation, this would convert ids back to text
        # For demonstration, create a simple mapping of common tokens to words
        token_map = {
            0: "<EOS>", 
            2565: "artificial", 
            2410: "intelligence", 
            844: "is", 
            4170: "the", 
            3561: "simulation", 
            4695: "of", 
            913: "human", 
            3507: "thinking", 
            1918: "by", 
            2059: "machines", 
            937: "and", 
            4140: "computer", 
            1718: "systems", 
            450: "learning", 
            494: "from", 
            2917: "data", 
            3740: "to", 
            3197: "perform", 
            2563: "tasks", 
            4830: "without", 
            4872: "explicit", 
            3854: "programming", 
            4721: "instructions", 
            2619: "using", 
            3249: "algorithms", 
            3525: "neural", 
            1376: "networks", 
            1015: "are", 
            4430: "inspired", 
            410: "by", 
            4017: "the", 
            4556: "structure", 
            3533: "functioning", 
            2427: "brain", 
            2592: "they", 
            780: "learn", 
            2627: "patterns", 
            4771: "through", 
            4286: "training", 
            3332: "with", 
            1889: "examples", 
            931: "deep", 
            1622: "learning", 
        }
        
        # Convert token IDs to text
        words = []
        for token_id in ids[0].tolist():  # Assuming batch size of 1
            if token_id in token_map:
                words.append(token_map[token_id])
            else:
                words.append(f"<token_{token_id}>")
        
        # Join words with spaces
        return " ".join(words)
    
    def _generate(self, input_ids, max_length=100):
        """Simple autoregressive generation"""
        self.model.eval()
        current_ids = input_ids.clone()
        
        # Create a KV cache for generation
        kv_cache = KeyValueCache(
            batch_size=current_ids.size(0),
            max_seq_length=self.model.config.max_seq_length,
            num_layers=self.model.config.num_layers,
            hidden_size=self.model.config.hidden_size,
            num_heads=self.model.config.num_heads
        )
        
        with torch.no_grad():
            # First forward pass to fill the cache with initial sequence
            _ = self.model(current_ids, use_cache=True)
            
            # Generate new tokens one by one
            for _ in range(max_length):
                # Forward pass with KV cache - only process the last token
                logits = self.model(current_ids[:, -1:], use_cache=True)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Check for end of sequence token
                if next_token.item() == 0:  # Assuming 0 is EOS token
                    break
        
        return current_ids
    
    def train_step(self, batch):
        """Train one batch with CoPG"""
        self.model.train()
        
        prompts, chosen_responses, rejected_responses = batch
        
        # Tokenize inputs
        prompt_ids = self._tokenize_batch(prompts)
        chosen_ids = self._tokenize_batch(chosen_responses)
        rejected_ids = self._tokenize_batch(rejected_responses)
        
        # Compute log probabilities for chosen and rejected responses
        chosen_log_probs = self._compute_log_probs(prompt_ids, chosen_ids)
        rejected_log_probs = self._compute_log_probs(prompt_ids, rejected_ids)
        
        # Compute preference loss: chosen should have higher probability than rejected
        preference_loss = -torch.mean(torch.log(torch.sigmoid(chosen_log_probs - rejected_log_probs)))
        
        # Compute KL divergence from reference model (original policy)
        kl_div = self._compute_kl_divergence(prompt_ids, chosen_ids)
        
        # Combined loss
        loss = preference_loss + self.kl_weight * kl_div
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "preference_loss": preference_loss.item(),
            "kl_div": kl_div.item()
        }
    
    def _tokenize_batch(self, texts):
        """Mock batch tokenization"""
        # In a real implementation, this would batch tokenize texts
        return torch.randint(0, self.model.config.vocab_size, (len(texts), 50))
    
    def _compute_log_probs(self, prompts, responses):
        """Compute log probabilities of responses given prompts"""
        # In a real implementation, this would compute actual log probs
        # Here we create dummy values that require gradients
        batch_size = prompts.size(0)
        log_probs = torch.randn(batch_size, requires_grad=True)
        # Connect to model parameters to ensure gradient flow
        dummy_param = next(self.model.parameters())
        return log_probs * dummy_param.sum() * 0.0001 + log_probs.detach()
    
    def _compute_kl_divergence(self, prompt_ids, response_ids):
        """Compute KL divergence between current and reference model"""
        # In a real implementation, this would compute actual KL divergence
        # Here we create a dummy value that requires gradients
        dummy_param = next(self.model.parameters())
        kl_div = torch.tensor(0.1, device=dummy_param.device)
        # Connect to model parameters to ensure gradient flow
        return kl_div * dummy_param.sum() * 0.0001 + kl_div.detach()

def log_to_file(filepath, content, mode='a'):
    """Write content to a file"""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

# Example usage
def main():
    # Configuration - slightly larger for deeper training
    config = ModelConfig(
        vocab_size=5000,
        hidden_size=384,  # Increased from 256
        num_layers=6,     # Increased from 4
        num_heads=12,     # Increased from 8
        window_size=64,
        max_seq_length=512
    )
    
    log_filepath = "c:\\Users\\emreq\\Desktop\\Transformers\\training_log.txt"
    
    # Clear previous log
    with open(log_filepath, 'w') as f:
        f.write("=== Training and Generation Log ===\n\n")
    
    # Log configuration
    log_to_file(log_filepath, f"Model Configuration:\n{config}")
    
    # Create model
    model = HybridTransformerModel(config)
    log_to_file(log_filepath, f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)  # Increased learning rate
    
    # Expanded examples for more robust training
    human_preferences_dataset = [
        (
            "What is artificial intelligence?", 
            "Artificial intelligence (AI) is the simulation of human intelligence by machines, especially computer systems.",
            "AI is when computers can think like humans."
        ),
        (
            "How do neural networks work?",
            "Neural networks are computational models inspired by the human brain that learn from data through pattern recognition.",
            "Neural networks are just complex math."
        ),
        (
            "What is machine learning?",
            "Machine learning is a field of AI where algorithms improve through experience without being explicitly programmed.",
            "Machine learning means computers can learn stuff."
        )
    ]
    
    log_to_file(log_filepath, f"Training with {len(human_preferences_dataset)} examples")
    for i, example in enumerate(human_preferences_dataset):
        log_to_file(log_filepath, f"\nExample {i+1}:")
        log_to_file(log_filepath, f"Question: {example[0]}")
        log_to_file(log_filepath, f"Preferred: {example[1]}")
        log_to_file(log_filepath, f"Less preferred: {example[2]}")
    
    # Create CoPG trainer
    trainer = CoPGTrainer(model, optimizer, human_preferences_dataset)
    
    # Deeper training with more epochs
    log_to_file(log_filepath, "\n=== Starting CoPG Training ===")
    print("Starting CoPG training with Hybrid Attention and KV Cache...")
    
    num_epochs = 15  # Increased from 5
    batches_per_epoch = 20  # Increased from 10
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(batches_per_epoch):
            # Create a batch with random example
            example_idx = np.random.randint(0, len(human_preferences_dataset))
            batch = [
                [human_preferences_dataset[example_idx][0]],
                [human_preferences_dataset[example_idx][1]],
                [human_preferences_dataset[example_idx][2]]
            ]
            
            # Train step
            metrics = trainer.train_step(batch)
            epoch_loss += metrics["loss"]
            
            if i % 5 == 0:  # Log only every 5 batches to keep log file concise
                log_message = f"Epoch {epoch}, Batch {i}: Loss = {metrics['loss']:.4f}, Preference Loss = {metrics['preference_loss']:.4f}, KL Div = {metrics['kl_div']:.4f}"
                print(log_message)
                log_to_file(log_filepath, log_message)
                
        avg_loss = epoch_loss/batches_per_epoch
        log_message = f"Epoch {epoch} completed: Average Loss = {avg_loss:.4f}"
        print(log_message)
        log_to_file(log_filepath, log_message)
    
    # Test generation with the trained model
    log_to_file(log_filepath, "\n=== Generation Testing ===")
    print("\nGenerating response with KV Cache...")
    
    # Expanded list of test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain machine learning in simple terms",
        "What are the applications of deep learning?",
        "How is AI changing the world?"
    ]
    
    for test_prompt in test_prompts:
        log_to_file(log_filepath, f"\nPrompt: {test_prompt}")
        print(f"\nPrompt: {test_prompt}")
        
        input_ids = trainer._tokenize(test_prompt)
        
        # Generate with KV Cache
        model.eval()
        with torch.no_grad():
            # Check if CUDA is available
            use_cuda_timing = torch.cuda.is_available()
            
            if use_cuda_timing:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                import time
                start_time = time.time()
            
            # Modified _generate method with logging
            def generate_with_logging(input_ids, max_length=50):  # Increased from 30
                current_ids = input_ids.clone()
                generation_steps = []
                generated_tokens = []
                
                # Create a KV cache for generation
                kv_cache = KeyValueCache(
                    batch_size=current_ids.size(0),
                    max_seq_length=model.config.max_seq_length,
                    num_layers=model.config.num_layers,
                    hidden_size=model.config.hidden_size,
                    num_heads=model.config.num_heads
                )
                
                # First forward pass to fill the cache with initial sequence
                _ = model(current_ids, use_cache=True, past_kv_cache=kv_cache)
                
                # Generate new tokens one by one
                for step in range(max_length):
                    # Forward pass with KV cache - only process the last token
                    logits = model(current_ids[:, -1:], use_cache=True, past_kv_cache=kv_cache)
                    next_token_logits = logits[:, -1, :]
                    
                    # Get top 3 tokens for logging
                    topk_values, topk_indices = torch.topk(next_token_logits, 3, dim=-1)
                    topk_probs = F.softmax(topk_values, dim=-1)
                    
                    # Select the next token
                    next_token = topk_indices[:, 0:1]
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    generated_tokens.append(next_token.item())
                    
                    # Log generation step
                    generation_steps.append({
                        'step': step,
                        'token_id': next_token.item(),
                        'top_tokens': topk_indices[0].tolist(),
                        'top_probs': topk_probs[0].tolist()
                    })
                    
                    # Check for end of sequence token
                    if next_token.item() == 0:  # Assuming 0 is EOS token
                        break
                
                return current_ids, generation_steps, generated_tokens
            
            output_with_cache, generation_steps, generated_tokens = generate_with_logging(input_ids)
            
            if use_cuda_timing:
                end_event.record()
                torch.cuda.synchronize()
                time_with_cache = start_event.elapsed_time(end_event)
            else:
                time_with_cache = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get the full generated text
            generated_text = trainer._detokenize(output_with_cache)
            
            # Log generation steps with token text
            log_to_file(log_filepath, "\nGeneration with KV Cache:")
            for step in generation_steps:
                token_id = step['token_id']
                token_text = "unknown"
                token_map = trainer._detokenize.func_globals['token_map'] if hasattr(trainer._detokenize, 'func_globals') else {}
                if token_id in token_map:
                    token_text = token_map[token_id]
                else:
                    token_text = f"<token_{token_id}>"
                
                log_to_file(log_filepath, 
                    f"Step {step['step']}: Token ID {token_id} ('{token_text}'), " + 
                    f"Top tokens: {step['top_tokens']}, Probabilities: {[f'{p:.4f}' for p in step['top_probs']]}"
                )
            
            # Log the complete generated text
            log_to_file(log_filepath, f"\nComplete generated response: {generated_text}")
            log_to_file(log_filepath, f"Generation time with KV Cache: {time_with_cache:.2f}ms")
            
            print(f"Generated response (with KV Cache): {generated_text}")
            print(f"Generation time: {time_with_cache:.2f}ms")
            
            # Generate without KV Cache for comparison (simplified)
            if use_cuda_timing:
                start_event.record()
            else:
                start_time = time.time()
                
            # Modified generation without cache for comparison
            def generate_without_cache_logging(input_ids, max_length=50):
                current_ids = input_ids.clone()
                tokens_generated = []
                
                for step in range(max_length):
                    # Process the full sequence each time
                    logits = model(current_ids, use_cache=False)
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    tokens_generated.append(next_token.item())
                    
                    if step == 0:
                        first_token_text = "unknown"
                        token_id = next_token.item()
                        token_map = trainer._detokenize.func_globals['token_map'] if hasattr(trainer._detokenize, 'func_globals') else {}
                        if token_id in token_map:
                            first_token_text = token_map[token_id]
                        else:
                            first_token_text = f"<token_{token_id}>"
                            
                        log_to_file(log_filepath, f"\nFirst token generated without cache: {token_id} ('{first_token_text}')")
                    
                    if next_token.item() == 0:  # Assuming 0 is EOS token
                        break
                
                return current_ids, tokens_generated
                
            output_without_cache, tokens_without_cache = generate_without_cache_logging(input_ids)
            
            if use_cuda_timing:
                end_event.record()
                torch.cuda.synchronize()
                time_without_cache = start_event.elapsed_time(end_event)
            else:
                time_without_cache = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get text without KV cache
            text_without_cache = trainer._detokenize(output_without_cache)
        
        # Compare results
        log_to_file(log_filepath, f"\nComplete response without KV Cache: {text_without_cache}")
        log_to_file(log_filepath, f"\nComparison of Generation Times:")
        log_to_file(log_filepath, f"With KV Cache: {time_with_cache:.2f}ms")
        log_to_file(log_filepath, f"Without KV Cache: {time_without_cache:.2f}ms")
        log_to_file(log_filepath, f"Speedup factor: {time_without_cache / time_with_cache:.2f}x")
        
        print(f"Response without KV Cache: {text_without_cache}")
        print(f"Generation time with KV Cache: {time_with_cache:.2f}ms")
        print(f"Generation time without KV Cache: {time_without_cache:.2f}ms")
        print(f"Speedup factor: {time_without_cache / time_with_cache:.2f}x")

    print(f"\nTraining and generation log has been saved to: {log_filepath}")

if __name__ == "__main__":
    main()