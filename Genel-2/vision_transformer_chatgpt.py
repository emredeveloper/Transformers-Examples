import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import requests

# -------------------- Yeni Veri Kümesi Sınıfı --------------------
class TextCapsDataset(Dataset):
    def __init__(self, dataset, num_samples=100):
        self.dataset = dataset.select(range(num_samples))  # Sadece ilk 100 örnek
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Görüntüyü yükleme
        img_url = item['image']
        img = self.load_image_from_url(img_url)
        
        # Metin işleme
        caption = item['user']  # Kullanıcı tarafından sağlanan açıklama
        
        inputs = self.tokenizer(
            caption,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': self.transform(img),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }
    
    def load_image_from_url(self, url):
        """URL'den görseli indirip PIL formatında döndürür."""
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img


# -------------------- İyileştirilmiş Model Mimarisi --------------------
class EnhancedTextToImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Önceden eğitilmiş modeller
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        # Çapraz dikkat mekanizması
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        
        # Gelişmiş decoder
        self.decoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Unflatten(1, (32, 32)),
            nn.ConvTranspose2d(32, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Görsel özellikler
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs.last_hidden_state
        
        # Metin özellikleri
        text_outputs = self.text_encoder(input_ids=input_ids, 
                                       attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state
        
        # Çapraz dikkat
        attn_output, _ = self.cross_attn(
            vision_features.permute(1, 0, 2),
            text_features.permute(1, 0, 2),
            text_features.permute(1, 0, 2)
        )
        
        # Görüntü oluşturma
        combined = attn_output.permute(1, 0, 2).mean(dim=1)
        return self.decoder(combined.unsqueeze(-1).unsqueeze(-1))


# -------------------- Ana İşlem --------------------
if __name__ == "__main__":
    # HuggingFace M4 - The Cauldron veri setini yükle (textcaps alt kümesi)
    

    dataset = load_dataset("HuggingFaceM4/the_cauldron", "textcaps")
    
    # Örnek kullanım
    custom_dataset = TextCapsDataset(dataset)
    print(f"Toplam örnek sayısı: {len(custom_dataset)}")
    sample = custom_dataset[0]
    print("Örnek veri şekilleri:")
    print(f"Görüntü: {sample['pixel_values'].shape}")
    print(f"Metin ID: {sample['input_ids'].shape}")
