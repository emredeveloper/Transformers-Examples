import torch
import json
from pathlib import Path

# Create checkpoints directory if it doesn't exist
Path("checkpoints").mkdir(exist_ok=True)

# Model parameters
params = {
    "dim": 128,  # Even smaller for testing
    "n_layers": 4,  # Fewer layers
    "n_heads": 8,  # Must evenly divide dim (128/8 = 16 head_dim)
    "n_kv_heads": 8,  # Match n_heads for simplicity
    "vocab_size": 128256,  # Match tokenizer.n_words
    "multiple_of": 32,
    "ffn_dim_multiplier": None,  # Remove multiplier to use default 4x
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_batch_size": 8,
    "max_seq_len": 512
}

# Save parameters
with open("checkpoints/params.json", "w") as f:
    json.dump(params, f, indent=2)

# Create a small checkpoint with random weights
state_dict = {
    "tok_embeddings.weight": torch.randn(params["vocab_size"], params["dim"]),
    "output.weight": torch.randn(params["vocab_size"], params["dim"]),
    "norm.weight": torch.ones(params["dim"]),
}

# Calculate hidden dimension for feed forward - always 4x
hidden_dim = 4 * params["dim"]  # Use 4x multiplier
hidden_dim = params["multiple_of"] * ((hidden_dim + params["multiple_of"] - 1) // params["multiple_of"])

# Add transformer layers
for i in range(params["n_layers"]):
    # Add attention weights
    state_dict.update({
        f"layers.{i}.attention.wq.weight": torch.randn(params["n_heads"] * (params["dim"] // params["n_heads"]), params["dim"]),
        f"layers.{i}.attention.wk.weight": torch.randn(params["n_heads"] * (params["dim"] // params["n_heads"]), params["dim"]),
        f"layers.{i}.attention.wv.weight": torch.randn(params["n_heads"] * (params["dim"] // params["n_heads"]), params["dim"]),
        f"layers.{i}.attention.wo.weight": torch.randn(params["dim"], params["n_heads"] * (params["dim"] // params["n_heads"])),
        f"layers.{i}.attention_norm.weight": torch.ones(params["dim"]),
        f"layers.{i}.ffn_norm.weight": torch.ones(params["dim"]),
        # Add feed-forward weights
        f"layers.{i}.feed_forward.w1.weight": torch.randn(hidden_dim, params["dim"]),
        f"layers.{i}.feed_forward.w2.weight": torch.randn(params["dim"], hidden_dim),
        f"layers.{i}.feed_forward.w3.weight": torch.randn(hidden_dim, params["dim"]),
    })

# Save checkpoint
torch.save(state_dict, "checkpoints/consolidated.00.pth")
print("Created test checkpoint and parameters in checkpoints/")