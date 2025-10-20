import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SimpleVisionLanguageModel(nn.Module):
    def __init__(self, vision_model_name="google/vit-base-patch16-224", language_model_name="bert-base-uncased"):
        super().__init__()
        
        # Vision encoder (e.g., ViT)
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        self.vision_hidden_size = self.vision_encoder.config.hidden_size
        
        # Language model (e.g., BERT)
        self.language_model = AutoModel.from_pretrained(language_model_name)
        self.language_hidden_size = self.language_model.config.hidden_size
        
        # Projection layer: map vision embeddings to the language embedding dimension
        self.projection = nn.Linear(self.vision_hidden_size, self.language_hidden_size)

        # Output layer for downstream tasks (e.g., classification)
        self.output_layer = nn.Linear(self.language_hidden_size, 1)  # Single-output example
    
    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images (torch.Tensor): [batch_size, 3, height, width] - visual inputs
            input_ids (torch.LongTensor): [batch_size, seq_len] - text inputs
            attention_mask (torch.Tensor): [batch_size, seq_len] - attention mask for the text
        
        Returns:
            logits (torch.Tensor): model outputs
        """
        # 1. Encode the visual information
        vision_outputs = self.vision_encoder(images).last_hidden_state  # [batch_size, num_patches, vision_hidden_size]
        vision_embeds = vision_outputs[:, 0, :]  # Use the CLS token (available by default in ViT)
        projected_vision_embeds = self.projection(vision_embeds)  # [batch_size, language_hidden_size]

        # 2. Encode the text information
        language_outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
        language_embeds = language_outputs.last_hidden_state  # [batch_size, seq_len, language_hidden_size]

        # 3. Combine the visual and textual representations
        combined_embeds = projected_vision_embeds.unsqueeze(1) + language_embeds[:, 0, :].unsqueeze(1)  # [batch_size, 1, language_hidden_size]

        # 4. Pass through the output layer
        logits = self.output_layer(combined_embeds.squeeze(1))  # [batch_size, 1]
        
        return logits
    
    
from PIL import Image
import requests
from transformers import AutoTokenizer, AutoFeatureExtractor
import torch

# Load the model
model = SimpleVisionLanguageModel()

# Tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Load the image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Text input
text = "This is a cat."

# Pre-processing
inputs = feature_extractor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # [1, 3, 224, 224]

text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
input_ids = text_inputs["input_ids"]
attention_mask = text_inputs["attention_mask"]

# Run the model
with torch.no_grad():
    outputs = model(pixel_values, input_ids, attention_mask)
    print("Output:", outputs)
