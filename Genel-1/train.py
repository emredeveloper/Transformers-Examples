import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from datasets import load_dataset
import numpy as np


JAX_TRACEBACK_FILTERING="off"
# 1. Flash Attention + Sparse MoE
class FlashMoeAttention(nn.Module):
    num_heads: int
    num_experts: int = 8
    top_k: int = 2
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        batch, seq_len, dim = x.shape
        head_dim = dim // self.num_heads
        
        # qkv projeksiyonu
        qkv = nn.Dense(dim * 3, dtype=self.dtype)(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        
        # Attention hesaplaması
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(head_dim)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attn_output = attn_output.reshape(batch, seq_len, dim)

        # MoE (Mixture of Experts)
        gate = nn.Dense(self.num_experts, dtype=self.dtype)(x)
        gate = jax.nn.softmax(gate, axis=-1)
        top_k_gates, top_k_indices = jax.lax.top_k(gate, self.top_k)
        
        expert_outputs = []
        for i in range(self.num_experts):
            expert = nn.Dense(dim, dtype=self.dtype)(x)
            mask = (top_k_indices == i).astype(jnp.float32)
            expert_outputs.append(expert * mask[..., None] * top_k_gates[..., None])
        
        return attn_output + sum(expert_outputs)

# 2. Ultra Derin Dil Modeli
class DeepSeekClone(nn.Module):
    vocab_size: int
    num_layers: int = 32
    num_heads: int = 16
    dim: int = 2048
    expert_count: int = 8

    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(self.vocab_size, self.dim)(inputs)
        for _ in range(self.num_layers):
            # Pre-LayerNorm
            x = nn.LayerNorm()(x)
            
            # Flash+Moe Attention
            residual = x
            x = FlashMoeAttention(num_heads=self.num_heads)(x)
            x = residual + x
            
            # Gated FFN
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.dim * 4)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.dim)(x)
        
        return nn.Dense(self.vocab_size)(x)

# 3. Optimizasyon ve Eğitim State
def create_train_state(rng, config):
    model = DeepSeekClone(**config)
    params = model.init(rng, jnp.ones((1, 512), dtype=jnp.int32))['params']
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-5, b1=0.9, b2=0.98),
        optax.add_decayed_weights(0.1)
    )
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

# 4. Veri Hazırlama
def build_data_pipeline(batch_size=256, seq_len=512):
    # C4 veri kümesini yükle
    ds = load_dataset("c4", "en", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000).take(1_000_000)  # 1M örnek al
    
    # Tokenizer oluştur
    vectorizer = TextVectorization(
        max_tokens=50000,
        output_sequence_length=seq_len,
        standardize="lower_and_strip_punctuation"
    )
    
    # Tokenizer'ı adapte et
    def adapt_tokenizer(dataset):
        text_data = dataset.map(lambda x: x["text"])
        vectorizer.adapt(text_data)
    
    adapt_tokenizer(ds)
    
    # Veriyi tokenize et
    def encode_fn(examples):
        tokens = vectorizer(examples["text"]).numpy().astype("int32")
        return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]}
    
    # Veri pipeline'ını oluştur
    ds = ds.map(encode_fn, batched=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# 5. JAX Data Loader
def jax_data_loader(ds):
    for batch in ds.as_numpy_iterator():
        yield jax.tree_map(jnp.asarray, batch)

# 6. Eğitim Döngüsü
@jax.pmap
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input_ids'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[..., :-1, :], batch['labels']
        ).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def train():
    # TPU/GPU Setup
    devices = jax.local_devices()
    print(f"Using {len(devices)} devices: {devices}")
    
    # Model Config
    config = {
        "vocab_size": 50000,
        "num_layers": 24,
        "dim": 4096,
        "num_heads": 32
    }
    
    # RNG Key
    rng = jax.random.PRNGKey(0)
    
    # Model State
    state = create_train_state(rng, config)
    state = jax.device_put_replicated(state, devices)
    
    # Veri Pipeline
    ds = build_data_pipeline()
    loader = jax_data_loader(ds)
    
    # Eğitim
    for epoch in range(10):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = jax.tree_map(lambda x: x.reshape(len(devices), -1, *x.shape[1:]), batch)
            state, loss = train_step(state, batch)
            total_loss += loss.mean().item()
            
            if step % 100 == 0:
                print(f"Step {step} | Loss: {loss.mean().item():.4f}")
        
        print(f"Epoch {epoch} | Avg Loss: {total_loss/(step+1):.4f}")
        
        # Model Checkpoint
        jax.checkpoint.save(f"model_epoch_{epoch}", state.params)

# 7. Ana Fonksiyon
if __name__ == "__main__":
    train()