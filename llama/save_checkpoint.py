# save_checkpoint.py

import torch
from model import Transformer, ModelArgs

# Define model parameters
params = ModelArgs(
    dim=512,  # Same change to avoid mismatch
    n_layers=16,
    n_heads=16,
    vocab_size=1000,
    max_seq_len=512,
    max_batch_size=8,
)

# Build the model
model = Transformer(params)

# Save the model weights
torch.save(model.state_dict(), "checkpoints/consolidated.00.pth")
print("Model weights saved to 'checkpoints/consolidated.00.pth'.")