# Detailed Walkthrough of the DiT (Diffusion Transformer) Code

## 1. Overview
This module implements a dynamic version of the **Diffusion Transformer (DiT)** architecture for image generation. It can be used for text-to-image synthesis as well as refinement tasks such as super-resolution or inpainting.

## 2. Utility Functions

### `round_to_nearest`
```python
def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
```
- **Purpose**: Round model widths so they align with the attention head count.
- **Usage**: Ensures dynamic width scaling remains compatible with the number of heads.
- **How it works**: Normalises `width_mult` with respect to `num_heads` and clamps to `min_value`.

## 3. Dynamic Linear Layers

### `DynaLinear`
```python
class DynaLinear(nn.Linear):
```
- **Goal**: Provide a linear layer whose input and output dimensions can change at runtime.
- **Highlights**:
  - `in_features` and `out_features` can be reconfigured on the fly.
  - `width_mult` controls the width multiplier.
  - `dyna_dim` determines which dimension is treated dynamically.

### `DynaQKVLinear`
```python
class DynaQKVLinear(nn.Linear):
```
- **Goal**: Produce the query, key, and value matrices with a single projection.
- **Highlights**:
  - Generates Q, K, and V in one pass.
  - Uses `einops` for concise tensor reshaping.
  - Supports dynamic width selection just like `DynaLinear`.

## 4. Attention Mechanism

### `Attention`
```python
class Attention(nn.Module):
```
- **Goal**: Implement the multi-head self-attention block.
- **Components**:
  - `qkv`: A `DynaQKVLinear` layer that emits query, key, and value tensors.
  - `q_norm`, `k_norm`: Normalisation layers applied to query and key tensors.
  - `proj`: A `DynaLinear` layer for the final projection.
  - `channel_mask`: Optional channel-wise masking for dynamic pruning.

**Execution Flow**:
1. Compute Q, K, and V from the input tensor.
2. Form attention scores and apply normalisation.
3. Optionally apply channel masks for sparsity.
4. Project the attended result back to the model dimension.

## 5. MLP (Multi-Layer Perceptron)

### `Mlp`
```python
class Mlp(nn.Module):
```
- **Goal**: Standard feed-forward network.
- **Structure**:
  - `fc1`: Expands the hidden dimension.
  - `act`: GELU activation.
  - `fc2`: Projects back to the model width.
  - Built-in support for channel masking.

## 6. Embedding Modules

### `TimestepEmbedder`
```python
class TimestepEmbedder(nn.Module):
```
- **Goal**: Encode diffusion timesteps into vector representations.
- **Technique**: Sinusoidal embeddings followed by an MLP.
- **Usage**: Provides the model with information about the current diffusion step.

### `LabelEmbedder`
```python
class LabelEmbedder(nn.Module):
```
- **Goal**: Embed class labels.
- **Highlights**:
  - Includes dropout for classifier-free guidance.
  - Randomly drops labels during training to improve unconditional generation.

## 7. Core Model Components

### `DiTBlock`
```python
class DiTBlock(nn.Module):
```
- **Goal**: The fundamental building block of the DiT architecture.
- **Components**:
  - `norm1`, `norm2`: Layer-normalisation layers.
  - `attn`: The dynamic attention module.
  - `mlp`: Feed-forward network.
  - `adaLN_modulation`: Adaptive layer-norm modulation unit.
  - `attn_rate`, `mlp_rate`: Runtime controls for channel counts.
  - `token_selection`: Mechanism for selecting the most informative tokens.

**AdaLN-Zero Conditioning**:
- Uses timestep and class embeddings to modulate the normalisation parameters.
- Applies `shift` and `scale` values to adapt each block based on the conditioning signal.

### `FinalLayer`
```python
class FinalLayer(nn.Module):
```
- **Goal**: Convert the hidden representation back to the patch format.
- **Responsibility**: Map the processed tokens into the final pixel-space patches.

## 8. The Main DiT Model

### `DiT`
```python
class DiT(nn.Module):
```
- **Goal**: Assemble the full diffusion transformer pipeline.
- **Components**:
  - `x_embedder`: Converts image patches into embeddings.
  - `t_embedder`: Encodes diffusion timesteps.
  - `y_embedder`: Encodes class labels (optional).
  - `pos_embed`: Static positional embeddings.
  - `blocks`: A stack of `DiTBlock` instances.
  - `final_layer`: Produces the reconstructed image patches.

**Forward Pass**:
1. Split the input image into patches and embed them.
2. Add timestep and (optional) label embeddings.
3. Propagate through every `DiTBlock` with adaptive conditioning.
4. Decode through the final layer.
5. Reassemble the patches into the output image tensor.

### `forward_with_cfg`
- **Goal**: Perform inference with classifier-free guidance.
- **Method**: Combine conditional and unconditional predictions to steer the output.

## 9. Positional Embedding Helpers

### `get_2d_sincos_pos_embed` and Friends
- **Goal**: Generate 2D sinusoidal positional embeddings.
- **Technique**: Use sine/cosine patterns across both spatial axes.
- **Usage**: Inject spatial coordinates into the transformer input.

## 10. Model Configurations

### Predefined Variants
```python
DiT_models = {
    'DiT-XL/2': DiT_XL_2,  # Largest model, 2x2 patches
    'DiT-L/2':  DiT_L_2,   # Large model
    'DiT-B/2':  DiT_B_2,   # Base model
    'DiT-S/2':  DiT_S_2,   # Small model
}
```

**Model Sizes**:
- **XL**: 28 layers, 1152 hidden size, 16 heads.
- **L**: 24 layers, 1024 hidden size, 16 heads.
- **B**: 12 layers, 768 hidden size, 12 heads.
- **S**: 12 layers, 384 hidden size, 6 heads.

**Patch Options**:
- `/2`: 2×2 patches (highest resolution).
- `/4`: 4×4 patches (medium resolution).
- `/8`: 8×8 patches (lowest resolution).

## 11. Dynamic Capabilities

The signature feature of this implementation is its **dynamic adaptation**:

1. **Channel Scaling**: `attn_rate` and `mlp_rate` adjust channel counts on the fly.
2. **Token Selection**: `token_selection` prunes to the most informative tokens.
3. **Adaptive Width**: `width_mult` widens or narrows layers dynamically.
4. **Conditional Execution**: `complete_model` toggles between the full and dynamic variants.

## 12. Use Cases

- **Image Generation**: Text-to-image workflows.
- **Image Editing**: Inpainting, outpainting, and super-resolution.
- **Style Transfer**: Style-aware synthesis and adaptation.
- **Conditional Generation**: Class-conditioned or prompt-conditioned sampling.

Overall, the code demonstrates how modern diffusion models can benefit from transformer architectures while leveraging dynamic adaptation to improve efficiency.
