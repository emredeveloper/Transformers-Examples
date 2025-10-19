"""CPU-friendly demo training pipeline for a tiny language model.

The original script in this repository demonstrated an experimental JAX/Flax
stack with Flash Attention and mixture-of-experts blocks that targets multi-GPU
or TPU environments.  While exciting, it was not practical for readers who
simply want to try out the examples on a local machine.

This file now contains a much smaller end-to-end pipeline that runs on CPU by
default.  It trains a compact Transformer on a slice of the
``tiny_shakespeare`` dataset and is intentionally configured with very small
hyper-parameters.  See ``train_heavy.md`` in this directory for guidance on how
to scale the idea back up to the original large-model setup.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import linen as nn
from flax.training import train_state
from transformers import AutoTokenizer


@dataclass
class DemoConfig:
    vocab_size: int
    seq_length: int = 64
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 8
    epochs: int = 3


DEFAULT_VOCAB_SIZE = 50257
DEFAULT_CONFIG = DemoConfig(vocab_size=DEFAULT_VOCAB_SIZE)


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.SelfAttention(num_heads=self.num_heads, dtype=jnp.float32)(x)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        return x + residual


class MiniTransformer(nn.Module):
    config: DemoConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        cfg = self.config
        x = nn.Embed(cfg.vocab_size, cfg.embed_dim)(input_ids)
        for _ in range(cfg.num_layers):
            x = TransformerBlock(
                embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, mlp_dim=cfg.mlp_dim
            )(x)
        x = nn.LayerNorm()(x)
        logits = nn.Dense(cfg.vocab_size)(x)
        return logits


def build_token_dataset(
    tokenizer, seq_length: int, *, max_sequences: int, text_limit: int
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset("tiny_shakespeare", split="train")
    joined_text = "\n".join(dataset["text"][:text_limit])
    encoded = tokenizer(
        joined_text,
        return_tensors="np",
        add_special_tokens=False,
    )["input_ids"].squeeze(0)

    total_length = (len(encoded) - 1) // seq_length * seq_length
    encoded = encoded[: total_length + 1]
    inputs = encoded[:-1].reshape(-1, seq_length)
    labels = encoded[1:].reshape(-1, seq_length)

    if max_sequences:
        inputs = inputs[:max_sequences]
        labels = labels[:max_sequences]

    return inputs.astype(np.int32), labels.astype(np.int32)


def data_iterator(
    inputs: np.ndarray, labels: np.ndarray, batch_size: int
) -> Iterator[Dict[str, jnp.ndarray]]:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_inputs = jnp.array(inputs[batch_idx])
        batch_labels = jnp.array(labels[batch_idx])
        yield {"input_ids": batch_inputs, "labels": batch_labels}


def create_train_state(rng: jax.random.KeyArray, config: DemoConfig) -> train_state.TrainState:
    model = MiniTransformer(config)
    dummy = jnp.zeros((1, config.seq_length), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]
    tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def compute_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, vocab_size)
    loss = optax.softmax_cross_entropy(logits, one_hot)
    return loss.mean()


def train_epoch(
    state: train_state.TrainState,
    iterator: Iterable[Dict[str, jnp.ndarray]],
) -> Tuple[train_state.TrainState, float]:
    epoch_loss = []

    for batch in iterator:
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch["input_ids"])
            return compute_loss(logits[:, :-1, :], batch["labels"][:, 1:])

        loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        epoch_loss.append(loss_value)

    if not epoch_loss:
        return state, 0.0

    loss_array = jax.device_get(jnp.stack(epoch_loss))
    mean_loss = float(loss_array.mean())
    return state, mean_loss


def run_demo_training(config: DemoConfig, *, text_limit: int, max_sequences: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs, labels = build_token_dataset(
        tokenizer,
        config.seq_length,
        max_sequences=max_sequences,
        text_limit=text_limit,
    )

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config)

    for epoch in range(1, config.epochs + 1):
        iterator = data_iterator(inputs, labels, config.batch_size)
        state, loss = train_epoch(state, iterator)
        print(f"Epoch {epoch}/{config.epochs} - Loss: {loss:.4f}")

    print("Training complete. The demo intentionally stops here without saving checkpoints.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-friendly demo trainer")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs, help="Number of epochs to train")
    parser.add_argument(
        "--seq-length",
        type=int,
        default=DEFAULT_CONFIG.seq_length,
        help="Sequence length for training examples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG.batch_size,
        help="Batch size for the demo run",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=200,
        help="Number of tiny_shakespeare lines to join for training data",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=512,
        help="Cap on the number of training sequences to keep (0 keeps all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = replace(
        DEFAULT_CONFIG,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    run_demo_training(config, text_limit=args.text_limit, max_sequences=args.max_sequences)
