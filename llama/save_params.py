# save_params.py

import json
from model import ModelArgs

# Model parametrelerini tanımla
params = ModelArgs(
    dim=512,  # Updated to be divisible by 16 heads
    n_layers=16,
    n_heads=16,
    vocab_size=1000,
    max_seq_len=512,
    max_batch_size=8,
)


# Parametreleri bir sözlük olarak kaydet
params_dict = {
    "dim": params.dim,
    "n_layers": params.n_layers,
    "n_heads": params.n_heads,
    "vocab_size": params.vocab_size,
    "max_seq_len": params.max_seq_len,
    "max_batch_size": params.max_batch_size,
}

# JSON dosyasına kaydet
with open("checkpoints/params.json", "w") as f:
    json.dump(params_dict, f, indent=4)
print("Model parametreleri 'checkpoints/params.json' olarak kaydedildi.")