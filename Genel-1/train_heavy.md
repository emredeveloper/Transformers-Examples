# Scaling the Demo to the Original Heavy Configuration

The previous version of `train.py` shipped with this repository showcased an
ambitious experimental stack that mixed Flash Attention, sparse Mixture of
Experts layers and multi-device `jax.pmap` training on the C4 dataset.  That
setup is fantastic for research on GPU or TPU clusters, but it is impractical
for casual experimentation—particularly in CPU-only environments.

The new `train.py` focuses on an approachable, CPU-friendly demonstration.  If
you want to recreate the larger experiment, start from the following notes:

- **Hardware expectations**: the original configuration assumes multiple A100 or
  TPU v4 chips.  Single-GPU or CPU-only setups will struggle due to the model's
  depth (24+ layers, 4K hidden size) and the 1M sample slice of C4.
- **Precision and memory**: enable bfloat16 or float16 to reduce memory pressure
  and ensure Flash Attention kernels are available (e.g., installing
  `jax[cuda12_pip]` on NVIDIA machines).
- **Dataset pipeline**: resume streaming C4 with `load_dataset("c4", "en",
  split="train", streaming=True)` and reintroduce the TensorFlow
  `TextVectorization` pipeline to build a 50k token vocabulary.
- **Model architecture**: bring back the `FlashMoeAttention` block together with
  the deep 24–32 layer `DeepSeekClone`.  Carefully shard parameters across
  devices and use `jax.pmap` to distribute batches.
- **Checkpointing**: use `jax.checkpoint.save` or Orbax to persist trained
  weights per epoch, as the heavy configuration is time-consuming to rerun.

Treat these bullets as a roadmap: port the demo-friendly pieces back into a
separate script, increase the hyper-parameters gradually, and verify stability
before scaling all the way up.
