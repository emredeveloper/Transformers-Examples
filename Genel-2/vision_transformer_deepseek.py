import os
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import io
from datasets import load_dataset
from itertools import islice

from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

# Login to Hugging Face
try:
    login(HF_TOKEN)
except Exception as e:
    logger.error(f"Failed to login to Hugging Face: {e}")
    raise

# -------------------- Veri Yükleme --------------------
def load_textcaps_data(num_samples: int = 100):
    """Load TextCaps dataset from Hugging Face with streaming."""
    try:
        # Stream modunda veri setini yükle
        dataset = load_dataset("HuggingFaceM4/the_cauldron", "textcaps", streaming=True)
        # İlk num_samples kadar veriyi al
        train_data = list(islice(dataset["train"], num_samples))
        
        logger.info(f"Loaded {len(train_data)} samples from dataset")
        return train_data
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

# -------------------- Veri Kümesi Sınıfı --------------------
class TextCapsDataset(Dataset):
    """Dataset class for TextCaps data."""
    
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            item = self.dataset[idx]
            
            # Get text
            conversations = item.get('conversations', [])
            text = next((conv['content'] for conv in conversations 
                        if conv.get('from') == 'assistant'), "")
            
            # Handle image
            images = item.get('images', [])
            if not images:
                raise ValueError(f"No images found for item {idx}")
            
            image_data = images[0]
            image = None
            
            try:
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    image = PILImage.open(io.BytesIO(image_data['bytes'])).convert("RGB")
                elif isinstance(image_data, PILImage.Image):
                    image = image_data.convert("RGB")
                elif isinstance(image_data, (str, bytes)):
                    image = PILImage.open(io.BytesIO(image_data)).convert("RGB")
                else:
                    raise ValueError(f"Unsupported image format: {type(image_data)}")
            except Exception as e:
                logger.error(f"Failed to process image at index {idx}: {e}")
                # Create a blank image as fallback
                image = PILImage.new('RGB', (224, 224), color='gray')
            
            # Transform image
            img_tensor = self.transform(image)
            
            # Validate tensor shape
            if img_tensor.shape != (3, 224, 224):
                logger.warning(f"Unexpected image tensor shape at index {idx}: {img_tensor.shape}")
                img_tensor = torch.zeros((3, 224, 224))
            
            inputs = self.tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'pixel_values': img_tensor,
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze()
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            logger.error(f"Item structure: {item}")
            raise

# -------------------- Model Mimarisi (Aynı) --------------------
class EnhancedTextToImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        
        self.projection = nn.Linear(768, 768)  # remains the same
        # Updated decoder to output (3, 224, 224)
        self.decoder = nn.Sequential(
            nn.Linear(768, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),      # (B, 256, 7, 7)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), # (B, 128, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),  # (B, 64, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),   # (B, 32, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),   # (B, 16, 112, 112)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, output_padding=1),    # (B, 3, 224, 224)
            nn.Tanh()
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Get features
        vision_features = self.vision_encoder(pixel_values).last_hidden_state  # [B, 197, 768]
        text_features = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state  # [B, L, 768]
        
        # Cross attention
        attn_output, _ = self.cross_attn(
            vision_features.permute(1, 0, 2),
            text_features.permute(1, 0, 2),
            text_features.permute(1, 0, 2)
        )
        
        # Process features
        combined = attn_output.permute(1, 0, 2).mean(dim=1)  # [B, 768]
        projected = self.projection(combined)  # [B, 768]
        
        # Decode
        return self.decoder(projected)

# -------------------- Eğitim Fonksiyonu --------------------
def train_model(
    num_samples: int = 1000,
    batch_size: int = 8,
    epochs: int = 10,
    val_split: float = 0.2
) -> nn.Module:
    """Train the model with validation split."""
    try:
        dataset = load_textcaps_data(num_samples)
        
        # Veri setini böl
        val_size = int(len(dataset) * val_split)
        train_dataset = dataset[val_size:]
        val_dataset = dataset[:val_size]
        
        logger.info(f"Training on {len(train_dataset)} samples")
        logger.info(f"Validating on {len(val_dataset)} samples")
        
        # Create data loaders
        train_data = TextCapsDataset(train_dataset)
        val_data = TextCapsDataset(val_dataset)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnhancedTextToImageModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = criterion(outputs, inputs['pixel_values'])
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**inputs)
                    val_loss += criterion(outputs, inputs['pixel_values']).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
        
        return model
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

# -------------------- Test Fonksiyonu --------------------
def test_with_local_image(model_path, image_path):
    model = EnhancedTextToImageModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img = PILImage.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    
    inputs = BertTokenizer.from_pretrained("bert-base-uncased")(
        "Men wearing black shirts saying England are gathered together.",
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    with torch.no_grad():
        output = model(img_tensor, inputs.input_ids, inputs.attention_mask)
    
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output = (output * 0.5) + 0.5
    plt.imshow(output)
    plt.axis('off')
    plt.savefig("textcaps_output_1k.png")
    plt.show()

# -------------------- Ana İşlem --------------------
if __name__ == "__main__":
    try:
        trained_model = train_model()
        test_with_local_image("best_model.pth", "image.jpeg")
    except Exception as e:
        logger.error(f"Application failed: {e}")