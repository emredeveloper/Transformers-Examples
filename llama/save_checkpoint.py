# save_checkpoint.py

import torch
from model import Transformer, ModelArgs

# Model parametrelerini tanımla
params = ModelArgs(
    dim=512,  # Same change to avoid mismatch
    n_layers=16,
    n_heads=16,
    vocab_size=1000,
    max_seq_len=512,
    max_batch_size=8,
)

# Modeli oluştur
model = Transformer(params)

# Model ağırlıklarını kaydet
torch.save(model.state_dict(), "checkpoints/consolidated.00.pth")

print("Model ağırlıkları 'checkpoints/consolidated.00.pth' olarak kaydedildi.")