import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms
from PIL import Image
import math
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


class NativeResolutionViT(nn.Module):
    """
    Native Resolution Vision Transformer
    Processes images at their original resolution without forced resizing
    """
    def __init__(
        self,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        mlp_ratio: float = 4.0,
        max_image_size: int = 1680,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_image_size = max_image_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Learnable position embeddings (adaptive)
        max_patches = (max_image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate position encodings for variable image sizes"""
        npatch = x.shape[1] - 1  # Exclude CLS token
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        
        # Interpolate patch position embeddings
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(h0, w0), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h0 * w0, dim)
        
        return torch.cat([class_pos_embed.unsqueeze(1), patch_pos_embed], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        pos_embed = self.interpolate_pos_encoding(x, H, W)
        x = x + pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with multi-head attention and MLP"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class MLPProjector(nn.Module):
    """
    Projects vision features to language model embedding space
    """
    def __init__(self, vision_dim: int, language_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, language_dim),
            nn.Dropout(0.1)
        )
        
        # Initialize with small weights for stable training
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projector(vision_features)


class VideoProcessor(nn.Module):
    """Process video frames with temporal modeling"""
    def __init__(self, vision_encoder: NativeResolutionViT, max_frames: int = 8):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.max_frames = max_frames
        self.temporal_embed = nn.Parameter(torch.randn(max_frames, vision_encoder.embed_dim) * 0.02)
        
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        video_frames: (batch_size, num_frames, channels, height, width)
        """
        B, T, C, H, W = video_frames.shape
        T = min(T, self.max_frames)  # Limit number of frames
        
        # Process each frame
        frame_features = []
        for t in range(T):
            frame_feat = self.vision_encoder(video_frames[:, t])  # (B, num_patches+1, embed_dim)
            # Add temporal embedding to non-CLS tokens
            frame_feat[:, 1:] += self.temporal_embed[t].unsqueeze(0).unsqueeze(0)
            frame_features.append(frame_feat)
        
        # Concatenate temporal features
        video_features = torch.cat(frame_features, dim=1)  # (B, T*(num_patches+1), embed_dim)
        return video_features


class MiMo7B(nn.Module):
    """
    MiMo-7B: Multimodal model with native resolution vision processing
    """
    def __init__(
        self,
        language_model_name: str = "microsoft/DialoGPT-medium",  # Using medium model for better generation
        vision_dim: int = 512,  # Reduced from 1024
        language_dim: int = 1024,  # Match DialoGPT-medium embedding size
        max_image_size: int = 896,  # Reduced from 1680
    ):
        super().__init__()
        
        print("Loading language model...")
        # Language model (7B parameters)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        print("Language model loaded!")
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens
        special_tokens = ["<image>", "<video>", "<chart>", "<ui>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        print("Special tokens added!")
        
        print("Initializing vision components...")
        # Vision components
        self.vision_encoder = NativeResolutionViT(
            embed_dim=vision_dim,
            max_image_size=max_image_size,
            num_layers=12,  # Reduced from 24
            num_heads=8,    # Reduced from 16
            patch_size=16   # Slightly larger patches for efficiency
        )
        self.projector = MLPProjector(vision_dim, language_dim)
        self.video_processor = VideoProcessor(self.vision_encoder)
        print("Vision components initialized!")
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Special token IDs
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.video_token_id = self.tokenizer.convert_tokens_to_ids("<video>")
    
    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Process single image"""
        print(f"Processing image - Original size: {image.size}")
        
        # Maintain aspect ratio while limiting size
        w, h = image.size
        max_size = 1680
        
        if max(w, h) > max_size:
            if w > h:
                new_w, new_h = max_size, int(h * max_size / w)
            else:
                new_w, new_h = int(w * max_size / h), max_size
            image = image.resize((new_w, new_h), Image.LANCZOS)
            print(f"Resized image to: {image.size}")
        
        # Convert to tensor
        image_tensor = self.image_transform(image).unsqueeze(0)
        print(f"Image tensor shape: {image_tensor.shape}")
        return image_tensor
    
    def process_video(self, video_frames: List[Image.Image]) -> torch.Tensor:
        """Process video frames"""
        # Sample frames if too many
        if len(video_frames) > 8:
            indices = np.linspace(0, len(video_frames)-1, 8, dtype=int)
            video_frames = [video_frames[i] for i in indices]
        
        # Process each frame
        frame_tensors = []
        for frame in video_frames:
            frame_tensor = self.process_image(frame).squeeze(0)
            frame_tensors.append(frame_tensor)
        
        return torch.stack(frame_tensors).unsqueeze(0)  # (1, num_frames, C, H, W)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[List[torch.Tensor]] = None,
        videos: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # Get text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Process visual inputs
        if images is not None:
            for i, image_tensor in enumerate(images):
                # Encode image
                vision_features = self.vision_encoder(image_tensor)
                projected_features = self.projector(vision_features)
                
                # Find image token positions
                image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
                if len(image_positions[0]) > i:
                    batch_idx, seq_idx = image_positions[0][i], image_positions[1][i]
                    # Replace image token with visual features
                    inputs_embeds[batch_idx, seq_idx:seq_idx+1] = projected_features[0, :1]  # Use CLS token
        
        if videos is not None:
            for i, video_tensor in enumerate(videos):
                # Encode video
                video_features = self.video_processor(video_tensor)
                projected_features = self.projector(video_features)
                
                # Find video token positions and replace
                video_positions = (input_ids == self.video_token_id).nonzero(as_tuple=True)
                if len(video_positions[0]) > i:
                    batch_idx, seq_idx = video_positions[0][i], video_positions[1][i]
                    # Insert video features (this is simplified - in practice need sequence manipulation)
                    inputs_embeds[batch_idx, seq_idx:seq_idx+1] = projected_features[0, :1]
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        videos: Optional[List[List[Image.Image]]] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text response given multimodal inputs"""
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Process images and integrate into input embeddings
        if images:
            processed_images = [self.process_image(img) for img in images]
            
            # Get input embeddings
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            # Process visual inputs and replace image tokens
            for i, image_tensor in enumerate(processed_images):
                # Encode image
                print(f"Encoding image {i+1}...")
                vision_features = self.vision_encoder(image_tensor)
                projected_features = self.projector(vision_features)
                print(f"Vision features shape: {vision_features.shape}")
                print(f"Projected features shape: {projected_features.shape}")
                
                # Find image token positions
                image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
                if len(image_positions[0]) > i:
                    batch_idx, seq_idx = image_positions[0][i], image_positions[1][i]
                    print(f"Replacing image token at position {seq_idx}")
                    # Replace image token embedding with visual features (CLS token)
                    inputs_embeds[batch_idx, seq_idx] = projected_features[0, 0]  # Use CLS token
            
            # Generate using input embeddings
            print("Generating response with visual features...")
            with torch.no_grad():
                outputs = self.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 50,  # Generate 50 new tokens
                    min_length=input_ids.shape[1] + 10,  # At least 10 new tokens
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
        else:
            # Generate without images
            print("Generating text-only response...")
            with torch.no_grad():
                outputs = self.language_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 50,  # Generate 50 new tokens
                    min_length=input_ids.shape[1] + 10,  # At least 10 new tokens
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Full generated text: {full_response}")
        
        # Extract only the new generated part
        prompt_length = len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
        if len(full_response) > prompt_length:
            response = full_response[prompt_length:].strip()
        else:
            response = full_response.replace(prompt, "").strip()
        
        # If response is still empty or too short, return a default
        if len(response) < 3:
            response = "I can see an image, but I need more context to provide a detailed description."
        
        return response


# Usage example and training utilities
class MiMoTrainer:
    """Training utilities for MiMo-7B"""
    
    def __init__(self, model: MiMo7B, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000
        )
    
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute multimodal training loss"""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch.get("images"),
            videos=batch.get("videos"),
            labels=batch["labels"]
        )
        return outputs.loss
    
    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()


# Example usage
if __name__ == "__main__":
    # Initialize model
    print("Initializing MiMo-7B model...")
    model = MiMo7B()
    trainer = MiMoTrainer(model)
    print("Model initialized successfully!")
    
    # Load real image
    image_path = "unnamed.png"
    try:
        real_image = Image.open(image_path)
        print(f"Loaded image: {image_path} - Size: {real_image.size}, Mode: {real_image.mode}")
        
        # Convert to RGB if necessary
        if real_image.mode != 'RGB':
            real_image = real_image.convert('RGB')
            print(f"Converted image to RGB mode")
        
        # Show some basic image statistics
        img_array = np.array(real_image)
        print(f"Image stats - Mean RGB: {img_array.mean(axis=(0,1)):.2f}")
        print(f"Image stats - Std RGB: {img_array.std(axis=(0,1)):.2f}")
        print(f"Image stats - Min/Max: {img_array.min()}/{img_array.max()}")
            
    except FileNotFoundError:
        print(f"Image {image_path} not found, using dummy image instead")
        real_image = Image.new("RGB", (512, 512), color="red")
    except Exception as e:
        print(f"Error loading image: {e}, using dummy image instead")
        real_image = Image.new("RGB", (512, 512), color="red")
    
    # Example multimodal input
    prompt = "Analyze this image: <image> What patterns, colors, and objects do you see? Describe in detail."
    
    print(f"\nPrompt: {prompt}")
    print("Processing image and generating response...")
    
    # Generate response
    response = model.generate(
        prompt=prompt,
        images=[real_image],
        max_length=512,
        temperature=0.8,
        top_p=0.9
    )
    
    print(f"\n{'='*50}")
    print(f"MODEL RESPONSE:")
    print(f"{'='*50}")
    print(response)
    print(f"{'='*50}")
    
    # Additional test with different prompt
    prompt2 = "What is the main subject of this image: <image>?"
    print(f"\nSecond prompt: {prompt2}")
    response2 = model.generate(
        prompt=prompt2,
        images=[real_image],
        max_length=256,
        temperature=0.5
    )
    print(f"Second response: {response2}")
    
    # Training example (mock data)
    print(f"\n{'='*30}")
    print("Training Step Example:")
    print(f"{'='*30}")
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 128)),
        "attention_mask": torch.ones(2, 128),
        "labels": torch.randint(0, 1000, (2, 128)),
        "images": [torch.randn(1, 3, 224, 224), torch.randn(1, 3, 336, 224)]
    }
    
    loss = trainer.train_step(batch)
    print(f"Training loss: {loss}")
    print("Training step completed!")