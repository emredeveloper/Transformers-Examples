# Import required libraries
import torch  # PyTorch framework
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional operations
import math  # Mathematical utilities
import numpy as np  # Numerical computations
from typing import Optional, Tuple  # Type hints
import matplotlib.pyplot as plt  # Visualization

class TimestepEmbedding(nn.Module):
    """Generate sinusoidal embeddings for timesteps, similar to transformer positional encodings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim  # Embedding size

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device  # Device where the computation happens (CPU/GPU)
        half_dim = self.dim // 2  # Half of the embedding size
        # Create logarithmically scaled frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        # Compute the frequencies with an exponential schedule
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Multiply with timesteps to create the embedding matrix
        embeddings = timesteps[:, None] * embeddings[None, :]
        # Concatenate sine and cosine representations
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # The model dimension must be divisible by the head count
        assert d_model % n_heads == 0

        self.d_model = d_model  # Input dimension
        self.n_heads = n_heads  # Number of heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection
        self.w_v = nn.Linear(d_model, d_model)  # Value projection
        self.w_o = nn.Linear(d_model, d_model)  # Output projection

        self.dropout = nn.Dropout(dropout)  # Dropout regularization

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape  # Extract input dimensions

        # Apply linear transformations and reshape for the heads
        # Compute query, key, and value matrices and split across heads
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # Multiply queries with the transposed keys and scale
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights and apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # Apply dropout

        # Multiply attention weights with values to get the output
        attention_output = torch.matmul(attention_weights, V)

        # Merge heads and pass through the final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        return self.w_o(attention_output)  # Apply the output projection

class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # Input to hidden layer
        self.linear2 = nn.Linear(d_ff, d_model)  # Hidden layer to output
        self.dropout = nn.Dropout(dropout)        # Prevent overfitting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass: Linear -> ReLU -> Dropout -> Linear
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward layers."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)  # Multi-head attention layer
        self.feed_forward = FeedForward(d_model, d_ff, dropout)         # Feedforward network
        self.norm1 = nn.LayerNorm(d_model)  # First normalization layer
        self.norm2 = nn.LayerNorm(d_model)  # Second normalization layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention and residual connection
        attn_output = self.attention(self.norm1(x), mask)  # Normalization followed by attention
        x = x + self.dropout(attn_output)  # Residual connection and dropout

        # Feedforward network and residual connection
        ff_output = self.feed_forward(self.norm2(x))  # Normalization and feedforward
        x = x + self.dropout(ff_output)  # Second residual connection and dropout

        return x

class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer (DiT) model for image generation.

    Args:
        img_size: Size of the input images (assumes square inputs).
        patch_size: Size of the patches extracted from the image.
        d_model: Hidden size of the transformer.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        d_ff: Hidden dimension of the feedforward network.
        num_classes: Number of classes for conditional generation.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        num_classes: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.img_size = img_size  # Image size
        self.patch_size = patch_size  # Patch size
        self.d_model = d_model  # Model hidden size
        self.num_patches = (img_size // patch_size) ** 2  # Total number of patches
        self.patch_dim = 3 * patch_size ** 2  # Patch dimension for RGB images (3 channels * patch area)

        # Patch embedding layer
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # MLP for the timestep embedding
            nn.GELU(),  # Gaussian Error Linear Unit activation
            nn.Linear(d_model * 4, d_model)  # Output layer
        )

        # Class embedding for conditional generation
        self.class_embedding = nn.Embedding(num_classes, d_model)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)  # Instantiate the requested number of transformer layers
        ])

        # Output projection
        self.norm = nn.LayerNorm(d_model)  # Final normalization layer
        self.output_projection = nn.Linear(d_model, self.patch_dim)  # Project back to the patch dimension

        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image into a collection of patches."""
        batch_size, channels, height, width = x.shape

        # Reshape to patches
        x = x.reshape(
            batch_size, channels,
            height // self.patch_size, self.patch_size,
            width // self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(batch_size, self.num_patches, -1)
        
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the image from a collection of patches."""
        batch_size = x.shape[0]  # Batch size
        height = width = int(self.num_patches ** 0.5)  # Original patch grid size

        # Rearrange patches back to the original image
        x = x.reshape(
            batch_size, height, width, 3, self.patch_size, self.patch_size
        )
        # Reorder to [batch, channels, height, patch_h, width, patch_w]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        # Merge patch dimensions to recover the original image size
        x = x.reshape(batch_size, 3, height * self.patch_size, width * self.patch_size)

        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion transformer.

        Args:
            x: Input tensor shaped as (batch_size, channels, height, width).
            timesteps: Timestep per sample in the batch.
            class_labels: Optional class labels for conditional generation.

        Returns:
            Predicted noise with the same shape as the input.
        """
        batch_size = x.shape[0]  # Batch size
        device = x.device  # Device used for computation

        # Convert the image into patches
        x = self.patchify(x)

        # Project patches to the embedding dimension
        x = self.patch_embedding(x)

        # Add positional embeddings
        x = x + self.pos_embedding

        # Add timestep embedding
        t_emb = self.time_embedding(timesteps)
        t_emb = self.time_mlp(t_emb)
        x = x + t_emb.unsqueeze(1)

        # Optionally add class embeddings
        if class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            x = x + class_emb.unsqueeze(1)

        # Run through the transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Project back to patch space
        x = self.output_projection(x)

        # Convert patches back to an image
        x = self.unpatchify(x)

        return x

class DDPMScheduler:
    """DDPM noise scheduler for the diffusion process."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps  # Total number of timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)  # Beta values
        self.alphas = 1.0 - self.betas  # Alpha values (1 - beta)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # Cumulative product of alphas
        # Previous cumulative product (shifted)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Quantities for q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # Square root of the cumulative product
        # Square root of (1 - cumulative product of alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Quantities for the posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean images according to the noise schedule."""
        # Gather scaling factors for the selected timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(x_0.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(x_0.device)

        # Add noise to the clean images
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

        return x_t  # Return the noisy images

    def sample_prev_timestep(self, x_t: torch.Tensor, noise_pred: torch.Tensor, timestep: int) -> torch.Tensor:
        """Sample x_{t-1} given x_t and the predicted noise."""
        if timestep == 0:
            return x_t  # At the final step simply return x_t

        # Retrieve parameters for this timestep
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[timestep]
        beta_t = self.betas[timestep]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep]

        # Variance for the reverse process
        posterior_variance_t = self.posterior_variance[timestep]

        # Estimate x_0 from x_t and the predicted noise
        pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # Compute the mean of q(x_{t-1} | x_t, x_0)
        mean = (torch.sqrt(alpha_cumprod_prev_t) * beta_t * pred_x0 +
                torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) * x_t) / (1 - alpha_cumprod_t)

        # Sample from the posterior
        if timestep > 0:
            noise = torch.randn_like(x_t)  # Random noise
            variance = torch.sqrt(posterior_variance_t) * noise  # Apply variance term
        else:
            variance = 0  # No variance at the last step

        x_prev = mean + variance  # Combine mean and variance

        return x_prev  # Return the sample for the previous timestep

def train_step(model: DiffusionTransformer,
               scheduler: DDPMScheduler,
               x_batch: torch.Tensor,
               class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Perform a single training step for the diffusion transformer."""

    batch_size = x_batch.shape[0]  # Batch size
    device = x_batch.device  # Device used for computation

    # Select random timesteps for each image in the batch
    timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)

    # Sample the noise that will be added to the images
    noise = torch.randn_like(x_batch)

    # Add noise scaled according to the timestep schedule
    noisy_images = scheduler.add_noise(x_batch, noise, timesteps)

    # Predict the noise residual
    noise_pred = model(noisy_images, timesteps, class_labels)

    # Compute the mean squared error between predicted and true noise
    loss = F.mse_loss(noise_pred, noise)

    return loss

@torch.no_grad()
def sample_images(
    model: DiffusionTransformer,
    scheduler: DDPMScheduler,
    num_samples: int = 4,
    class_labels: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate sample images with a trained diffusion transformer."""
    model.eval()  # Switch to evaluation mode

    # Sample initial noise as the latent variable
    img_size = model.img_size  # Image size
    x_t = torch.randn((num_samples, 3, img_size, img_size), device=device)

    # If no class labels are provided for a conditional model, sample random labels
    if class_labels is None and hasattr(model, 'class_embedding'):
        num_classes = model.class_embedding.num_embeddings  # Total number of classes
        class_labels = torch.randint(0, num_classes, (num_samples,), device=device)  # Random class labels

    # Run the reverse diffusion process
    with torch.no_grad():
        # Iterate over timesteps in reverse order
        for t in reversed(range(scheduler.num_timesteps)):
            # Create a tensor filled with the current timestep index
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Predict the noise for the current latent
            noise_pred = model(x_t, timesteps, class_labels)

            # Sample the previous timestep
            x_t = scheduler.sample_prev_timestep(x_t, noise_pred, t)

    # Clamp to the valid pixel range
    x_t = torch.clamp(x_t, -1.0, 1.0)

    # Scale from [-1, 1] to [0, 1]
    x_t = (x_t + 1) / 2

    return x_t  # Return the generated samples

# Example usage and training loop
def example_usage():
    """Showcase how to use the diffusion transformer."""
    # Select the compute device (prefer GPU when available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model and scheduler
    model = DiffusionTransformer(
        img_size=32,     # Image size
        patch_size=4,    # Patch size
        d_model=256,     # Model hidden size
        n_layers=6,      # Number of transformer layers
        n_heads=8,       # Number of attention heads
        d_ff=1024,       # Hidden size of the feedforward network
        num_classes=10,  # Number of classes (e.g., CIFAR-10)
        dropout=0.1      # Dropout probability
    ).to(device)  # Move model to the selected device

    # Instantiate the noise scheduler
    scheduler = DDPMScheduler(num_timesteps=1000)

    # Create sample data
    batch_size = 4  # Batch size
    x = torch.randn(batch_size, 3, 32, 32, device=device)  # Random input images
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)  # Random timesteps
    class_labels = torch.randint(0, 10, (batch_size,), device=device)  # Random class labels

    # Forward pass
    noise_pred = model(x, timesteps, class_labels)
    print(f"Input shape: {x.shape}")
    print(f"Noise prediction shape: {noise_pred.shape}")

    # Single training step
    loss = train_step(model, scheduler, x, class_labels)
    print(f"Training loss: {loss.item():.4f}")

    # Generate sample images
    samples = sample_images(model, scheduler, num_samples=4, device=device)
    print(f"Generated samples shape: {samples.shape}")

    # Visualize the samples
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))  # Create a 1x4 grid
    for i, ax in enumerate(axes):
        # Convert from [C, H, W] to [H, W, C] and display
        ax.imshow(samples[i].permute(1, 2, 0).cpu().numpy())
        ax.axis('off')  # Hide axes
    plt.tight_layout()  # Improve layout
    plt.show()  # Display the figure

    # Start a training loop
    print("Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Switch to training mode
    model.train()

    # Perform a single training iteration
    loss = train_step(model, scheduler, x, class_labels)

    # Backpropagation and parameter update
    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

    print(f"Training loss: {loss.item():.4f}")

    # Generate additional sample images
    print("Generating sample images...")
    sample_labels = torch.arange(4, device=device)  # Create one sample for the first four classes
    generated_images = sample_images(model, scheduler, num_samples=4, class_labels=sample_labels, device=device)

    print(f"Generated images shape: {generated_images.shape}")
    print("Sampling complete!")

    return model, scheduler, generated_images  # Return the model, scheduler, and generated samples

# Run example
if __name__ == "__main__":
    model, scheduler, samples = example_usage()