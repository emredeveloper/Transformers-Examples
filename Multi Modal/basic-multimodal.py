import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import torchaudio
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.io import wavfile

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data path configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multimodal_dataset")
os.makedirs(DATA_DIR, exist_ok=True)

# Build the dataset structure
def create_real_data_metadata():
    """Create metadata for real video, audio, and text files."""

    # Create the data directories
    video_dir = os.path.join(DATA_DIR, "videos")
    audio_dir = os.path.join(DATA_DIR, "audios")
    text_dir = os.path.join(DATA_DIR, "texts")

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    # Data structure for the metadata file
    data_entries = []

    # Prompt the user to upload video files
    print("\n" + "="*80)
    print("REAL DATA PREPARATION")
    print("="*80)
    print("This step uses real video, audio, and text assets.")
    print("Copy a few files into the folders below and we will generate the metadata.")
    print("\nPlease perform the following steps manually:")
    print(f"1. Copy your video files to: {video_dir}")
    print(f"2. Copy your audio files to: {audio_dir}")
    print(f"3. Copy or create text files for each sample in: {text_dir}")
    print("4. Make sure video, audio, and text file names align with matching indices.")
    print("   Example: video_1.mp4, audio_1.wav, text_1.txt")
    print("\nPress ENTER when everything is ready...")
    input()

    # Scan the files and build metadata
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    for i, video_file in enumerate(video_files):
        video_id = i
        video_path = os.path.join(video_dir, video_file)

        # Locate the matching audio file (same name or index)
        base_name = os.path.splitext(video_file)[0]
        audio_file = None
        for ext in ['.wav', '.mp3']:
            possible_audio = base_name + ext
            if os.path.exists(os.path.join(audio_dir, possible_audio)):
                audio_file = possible_audio
                break

        # If there is no audio file, request manual extraction
        audio_path = None
        if audio_file:
            audio_path = os.path.join(audio_dir, audio_file)
        else:
            # Suggest a new audio file name
            audio_path = os.path.join(audio_dir, f"{base_name}.wav")

            # Ask the user to extract audio manually (FFmpeg required)
            print(f"No audio track found for '{base_name}'.")
            print(f"Please create the audio file manually and save it to '{audio_path}'.")
            print("Press ENTER once the file is available...")
            input()

        # Locate or create the corresponding text file
        text_file = base_name + ".txt"
        text_path = os.path.join(text_dir, text_file)

        text = ""
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            # Ask the user to provide a text description if none exists
            print(f"Enter a text description for '{base_name}' (describe the video content):")
            text = input().strip()
            # Save the text file
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)

        # Add the sample to the metadata collection
        data_entries.append({
            "id": video_id,
            "video_path": os.path.relpath(video_path, DATA_DIR),
            "audio_path": os.path.relpath(audio_path, DATA_DIR),
            "text": text,
            "text_path": os.path.relpath(text_path, DATA_DIR)
        })

    # Persist metadata to JSON
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data_entries, f, ensure_ascii=False, indent=4)

    print(f"Metadata created for {len(data_entries)} samples.")
    return metadata_path


class MultiModalDataset(Dataset):
    """Multimodal dataset containing video, audio, and text."""
    
    def __init__(self, metadata_path, max_length=128):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_dir = os.path.dirname(metadata_path)
        
        # Text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.max_length = max_length

        # Video transforms
        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Larger resolution for real videos
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Audio transforms
        self.audio_transform = transforms.Compose([
            transforms.Normalize(mean=[-15], std=[40])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Text processing - prefer reading from file, fallback to inline text
        text = item.get("text", "")
        if "text_path" in item:
            try:
                text_path = os.path.join(self.data_dir, item["text_path"])
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
            except Exception as e:
                print(f"Text file read error: {e}")

        text_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Audio processing - support wav and mp3 formats
        audio_path = os.path.join(self.data_dir, item["audio_path"])
        try:
            if audio_path.lower().endswith('.wav'):
                # Use scipy.io.wavfile for WAV files
                sample_rate, audio_data = wavfile.read(audio_path)
                # Convert from integer representations to float32
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32767.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483647.0
                elif audio_data.dtype == np.uint8:
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0

                # Convert multi-channel audio to mono
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_data = np.mean(audio_data, axis=1)

                # Convert to tensor
                waveform = torch.tensor(audio_data).float().unsqueeze(0)
            else:
                # Fallback to torchaudio.load for other formats
                try:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    # Convert stereo to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                except Exception as e:
                    print(f"Audio load error: {e}")
                    # Create an empty waveform placeholder
                    waveform = torch.zeros(1, 16000 * 5)  # 5 seconds of silence
                    sample_rate = 16000
        except Exception as e:
            print(f"Audio processing error: {e}")
            waveform = torch.zeros(1, 16000 * 5)  # 5 seconds of silence
            sample_rate = 16000

        # Resample to 16 kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Normalise duration to five seconds
        target_length = 5 * 16000
        if waveform.shape[1] < target_length:
            # Pad if the audio is shorter
            padding = torch.zeros(waveform.shape[0], target_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        else:
            # Trim longer audio
            waveform = waveform[:, :target_length]

        # Create a spectrogram
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, n_mels=128
        )(waveform)
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        # Remove the channel dimension
        spectrogram = spectrogram.squeeze(0)

        # Video processing
        video_path = os.path.join(self.data_dir, item["video_path"])
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Unable to open video file: {video_path}")

            # Gather video information
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            frames = []
            frame_indices = []

            # Target number of frames
            target_frames = 16  # Capture more frames

            if total_frames <= 0:
                raise ValueError(f"Video frame count is zero or negative: {total_frames}")

            # Determine frame indices
            if total_frames <= target_frames:
                frame_indices = list(range(total_frames))
            else:
                # Sample frames at regular intervals
                step = total_frames / target_frames
                frame_indices = [int(i * step) for i in range(target_frames)]

            for frame_idx in frame_indices:
                # Seek to the desired frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.video_transform(frame)
                frames.append(frame)

            cap.release()

            # Fill any missing frames
            while len(frames) < target_frames:
                if frames:
                    frames.append(frames[-1])  # Repeat the last frame
                else:
                    # Insert an empty frame placeholder
                    frames.append(torch.zeros(3, 224, 224))

            video_tensor = torch.stack(frames[:target_frames])  # Ensure consistent length

        except Exception as e:
            print(f"Video processing error: {e}")
            # Return an empty video tensor if processing fails
            video_tensor = torch.zeros(16, 3, 224, 224)
        
        return {
            "id": item["id"],
            "text_input_ids": text_encoding["input_ids"].squeeze(0),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(0),
            "audio": spectrogram,
            "video": video_tensor,
            "raw_text": text
        }


# Model mimarisi - Multimodal Fusion
class VideoEncoder(nn.Module):
    """Video encoder module tuned for real-world videos."""
    def __init__(self, embed_dim=256, input_shape=(16, 3, 224, 224)):
        super().__init__()
        
        num_frames, channels, height, width = input_shape
        
        # 3D CNN encoder with a deeper stack
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # [B, 64, F, H/2, W/2]
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [B, 128, F/2, H/4, W/4]
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [B, 256, F/4, H/8, W/8]
            
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [B, 512, F/8, H/16, W/16]
        )
        
        # Compute the final spatial dimensions
        f_out = num_frames // 8
        h_out = height // 16
        w_out = width // 16
        
        # Global average pooling and projection
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embed_dim)
        )
        
    def forward(self, x):
        # Input x: [batch_size, frames, channels, height, width]
        # Reorder for 3D CNN: [batch_size, channels, frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        
        try:
            # Standard forward pass
            x = self.conv3d(x)
            # Global average pooling
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.projection(x)
        except RuntimeError as e:
            # On failure, report the shape and use a safer path
            print(f"VideoEncoder error: {e}")
            print(f"Input shape: {x.shape}")

            # Safe alternative: simplified processing
            batch_size = x.size(0)
            x = torch.mean(x, dim=(2, 3, 4))  # Global average pooling [B, C]
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = torch.nn.functional.linear(x,
                                          torch.randn(256, x.size(1), device=x.device))
            
        return x


class AudioEncoder(nn.Module):
    """Audio encoder module designed for high-fidelity audio."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, F/2, T/2]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 128, F/4, T/4]
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 256, F/8, T/8]
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # [B, 512, F/16, T/16]
        )
        
        # Global average pooling and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embed_dim)
        )
        
    def forward(self, x):
        # Input x: [batch_size, freq_bins, time_frames]
        x = x.unsqueeze(1)  # [batch_size, 1, freq_bins, time_frames]

        try:
            # Standard forward pass
            x = self.conv(x)
            # Global average pooling
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.projection(x)
        except RuntimeError as e:
            print(f"AudioEncoder error: {e}")
            print(f"Input shape: {x.shape}")

            # Safe fallback: simplified processing
            batch_size = x.size(0)
            x = torch.mean(x, dim=(2, 3))  # Global average pooling [B, C]
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = torch.nn.functional.linear(x,
                                          torch.randn(256, x.size(1), device=x.device))
        
        return x


class TextEncoder(nn.Module):
    """BERT-based text encoder module."""

    def __init__(self, embed_dim=256):
        super().__init__()
        # Load the Turkish BERT model
        self.bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")

        # Project BERT outputs to the shared embedding size
        self.projection = nn.Linear(self.bert.config.hidden_size, embed_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        projected = self.projection(cls_token)
        return projected


class MultiModalTransformer(nn.Module):
    """Enhanced multimodal transformer for video, audio, and text."""
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, output_dim=5):
        super().__init__()
        
        # Sub-encoders tailored for high-quality video and audio inputs
        self.video_encoder = VideoEncoder(embed_dim, input_shape=(16, 3, 224, 224))
        self.audio_encoder = AudioEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        
        # Projection layers per modality
        self.video_projection = nn.Linear(embed_dim, embed_dim)
        self.audio_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        
        # Transformer encoder blocks for cross-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention module for modality fusion
        self.modal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion feed-forward network
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Output classification head
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, output_dim)
        )
        
        # Store embedding dimension
        self.embed_dim = embed_dim
        
    def forward(self, video, audio, text_input_ids, text_attention_mask):
        # Extract features for each modality
        try:
            # Encode each modality separately
            video_emb = self.video_encoder(video)
            audio_emb = self.audio_encoder(audio)
            text_emb = self.text_encoder(text_input_ids, text_attention_mask)

            # Align features with the projection layers
            video_emb = self.video_projection(video_emb)
            audio_emb = self.audio_projection(audio_emb)
            text_emb = self.text_projection(text_emb)

            # Concatenate features and pass through the fusion network
            combined_features = torch.cat([video_emb, audio_emb, text_emb], dim=1)
            fused_features = self.fusion_layer(combined_features)

            # Produce classification logits
            output = self.output_layer(fused_features)

        except RuntimeError as e:
            print(f"MultiModalTransformer error: {e}")
            # Fall back to a simplified representation
            batch_size = video.size(0)

            # Safe fallback features
            video_mean = torch.mean(video, dim=(1, 2, 3, 4))
            audio_mean = torch.mean(audio, dim=(1, 2))
            text_mean = torch.mean(text_input_ids.float(), dim=1)

            combined = torch.cat([video_mean, audio_mean, text_mean], dim=1)
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)

            # Apply a direct linear projection for five classes
            out_dim = 5  # Default number of classes
            output = torch.nn.functional.linear(combined,
                                              torch.randn(out_dim, combined.size(1), device=video.device))
        
        return output


# Training and evaluation helpers
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=5):
    """Train the multimodal model."""
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            targets = batch["id"].to(device)  # Use the ID field as the label

            # Print tensor shapes for debugging on the first batch
            if batch_idx == 0 and epoch == 0:
                print(f"Video shape: {video.shape}")
                print(f"Audio shape: {audio.shape}")
                print(f"Text input_ids shape: {text_input_ids.shape}")
            
            # Forward pass
            try:
                outputs = model(video, audio, text_input_ids, text_attention_mask)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimisation step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            except RuntimeError as e:
                print(f"Runtime error (batch {batch_idx}): {e}")
                print(f"Video shape: {video.shape}, Audio shape: {audio.shape}")
                continue

        # Track epoch-level loss
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return train_losses

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the multimodal model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move data to device
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            targets = batch["id"].to(device)

            try:
                # Forward pass
                outputs = model(video, audio, text_input_ids, text_attention_mask)
                loss = criterion(outputs, targets)

                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            except RuntimeError as e:
                print(f"Runtime error during evaluation (batch {batch_idx}): {e}")
                print(f"Video shape: {video.shape}, Audio shape: {audio.shape}")
                continue

    # Average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# Demo data generation
def create_sample_data():
    """Create sample multimodal data: video, audio, and text (demo)."""
    # Ensure directories exist for generated files
    video_dir = os.path.join(DATA_DIR, "videos")
    audio_dir = os.path.join(DATA_DIR, "audios")
    text_dir = os.path.join(DATA_DIR, "texts")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    # Metadata container
    data_entries = []
    
    # Create a simple synthetic video (sequence of coloured squares)
    for i in range(5):
        # For each sample
        video_frames = []
        for j in range(30):  # 30-frame video
            # Generate a coloured RGB square
            if j < 10:
                frame = np.ones((64, 64, 3), dtype=np.uint8) * 50  # Dark gray
            elif j < 20:
                frame = np.ones((64, 64, 3), dtype=np.uint8) * 150  # Medium gray
            else:
                frame = np.ones((64, 64, 3), dtype=np.uint8) * 250  # Light gray

            # Adjust colour channels per sample for variety
            frame[:,:,i % 3] = 200
            video_frames.append(frame)

        # Save the video
        video_path = os.path.join(video_dir, f"sample_video_{i}.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 64))
        for frame in video_frames:
            out.write(frame)
        out.release()
        
        # Create an audio file containing a simple sine wave
        audio_path = os.path.join(audio_dir, f"sample_audio_{i}.wav")
        sample_rate = 16000
        t = np.linspace(0, 2, 2 * sample_rate, endpoint=False)
        # Use a different frequency for each sample
        frequency = 440 * (i + 1)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        # Convert to 16-bit integers for scipy
        audio_data_16bit = (audio_data * 32767).astype(np.int16)
        # Save as mono audio via scipy.io.wavfile
        wavfile.write(audio_path, sample_rate, audio_data_16bit)

        # Generate associated text descriptions
        if i == 0:
            text = "This video contains grayscale squares with a 440 Hz tone."
        elif i == 1:
            text = "This video contains red-tinted squares with an 880 Hz tone."
        elif i == 2:
            text = "This video contains green-tinted squares with a 1,320 Hz tone."
        elif i == 3:
            text = "This video contains blue-tinted squares with a 1,760 Hz tone."
        else:
            text = "This video contains mixed-colour squares with a 2,200 Hz tone."

        # Save text file
        text_path = os.path.join(text_dir, f"sample_text_{i}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Append to metadata entries
        data_entries.append({
            "id": i,
            "video_path": os.path.relpath(video_path, DATA_DIR),
            "audio_path": os.path.relpath(audio_path, DATA_DIR),
            "text": text,
            "text_path": os.path.relpath(text_path, DATA_DIR)
        })
    
    # Persist metadata to JSON
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data_entries, f, ensure_ascii=False, indent=4)
    
    print(f"Demo data created with {len(data_entries)} samples.")
    return metadata_path

# Main execution entry point
def main():
    """Main driver function."""
    print("Starting multimodal model training...")

    # Prompt the user to select the data source
    print("\nSelect the data source:")
    print("1 - Sample data (automatically generated demo dataset)")
    print("2 - Real data (actual video, audio, and text files)")

    choice = input("Your choice (1/2): ").strip()

    # Prepare data according to the chosen option
    if choice == "2":
        print("\nUsing real data...")
        metadata_path = create_real_data_metadata()
    else:
        print("\nCreating sample demo data...")
        metadata_path = create_sample_data()

    # Build the dataset
    dataset = MultiModalDataset(metadata_path)
    
    # Split into training and test subsets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # Instantiate the model
    model = MultiModalTransformer(embed_dim=256, num_heads=4, num_layers=2, output_dim=5).to(device)
    print(f"Model initialised: {model.__class__.__name__}")

    # Loss function and optimiser (with weight decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    # Train the model
    print("Starting model training...")
    train_losses = train_model(model, train_loader, optimizer, criterion, device, num_epochs=10)
    
    # Evaluate the model
    print("Running model evaluation...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

    # Summarise metrics
    print("\nModel Summary:")
    print(f"- Total Training Epochs: 10")
    print(f"- Final Training Loss: {train_losses[-1]:.4f}")
    print(f"- Test Loss: {test_loss:.4f}")
    print(f"- Accuracy: {test_accuracy:.2f}%")

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DATA_DIR, "training_loss.png"))
    plt.show()
    
    # Save the trained model
    model_path = os.path.join(DATA_DIR, "multimodal_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Visualise a sample from the test set using the original dataset
    print("Visualising a sample example...")
    visualize_example(model, dataset, device)
    
    return model, test_accuracy

def visualize_example(model, dataset, device):
    """Display a sample prediction from the test set."""
    # Pick a random example
    idx = np.random.randint(len(dataset))
    sample = dataset[idx]

    # Switch model to evaluation mode
    model.eval()

    # Convert to tensors and move to device
    video = sample["video"].unsqueeze(0).to(device)
    audio = sample["audio"].unsqueeze(0).to(device)
    text_input_ids = sample["text_input_ids"].unsqueeze(0).to(device)
    text_attention_mask = sample["text_attention_mask"].unsqueeze(0).to(device)

    # Log tensor shapes
    print(f"Sample visualisation - Video shape: {video.shape}")
    print(f"Sample visualisation - Audio shape: {audio.shape}")

    # Make a prediction
    predicted_class = None
    try:
        with torch.no_grad():
            output = model(video, audio, text_input_ids, text_attention_mask)
            _, predicted_class = torch.max(output, 1)
    except RuntimeError as e:
        print(f"Error during sample visualisation: {e}")
        predicted_class = torch.tensor([-1]).to(device)  # Invalid class on error

    # Ground truth label
    true_class = sample["id"]

    # Present the results
    print(f"\nSample Visualisation (Index {idx}):")
    print(f"True class: {true_class}")
    if predicted_class is not None and predicted_class.item() != -1:
        print(f"Predicted class: {predicted_class.item()}")
    else:
        print("Prediction unavailable (model error)")

    # Plot a few frames from the video
    plt.figure(figsize=(15, 5))
    for i in range(min(5, video.size(1))):
        plt.subplot(1, 5, i+1)
        frame = video[0, i].cpu().permute(1, 2, 0)
        # Revert normalisation
        frame = frame * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        frame = torch.clamp(frame, 0, 1)
        plt.imshow(frame)
        plt.title(f"Frame {i}")
        plt.axis('off')
    plt.savefig(os.path.join(DATA_DIR, "sample_frames.png"))
    plt.show()
    
    # Display the audio spectrogram
    plt.figure(figsize=(10, 4))
    # Ensure the spectrogram is 2D
    audio_data = sample["audio"].cpu()
    if len(audio_data.shape) == 1:
        # Expand 1D tensor to 2D
        audio_data = audio_data.unsqueeze(0)
    elif len(audio_data.shape) > 2:
        # Use the first slice if extra dimensions exist
        audio_data = audio_data[0]

    plt.imshow(audio_data, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Filter Banks')
    plt.savefig(os.path.join(DATA_DIR, "sample_spectrogram.png"))
    plt.show()
    
    # Retrieve the decoded text, falling back if tokenizer access is limited
    raw_text = ""
    try:
        # Attempt to reach the original dataset when working with Subset
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'tokenizer'):
            # When wrapped by Subset
            tokenizer = dataset.dataset.tokenizer
            raw_text = tokenizer.decode(sample["text_input_ids"].tolist(), skip_special_tokens=True)
        else:
            # When using the original dataset directly
            raw_text = dataset.tokenizer.decode(sample["text_input_ids"].tolist(), skip_special_tokens=True)
    except Exception as e:
        # Strip out special token IDs if the tokenizer is unavailable
        text_tokens = sample["text_input_ids"].tolist()
        # Filter out special token IDs such as 0, 101, and 102 (BERT specials)
        text_tokens = [t for t in text_tokens if t > 102 and t != 0]
        raw_text = f"IDs: {text_tokens} (raw text unavailable without tokenizer access)"

    print(f"Text: {raw_text}")


# Run the main program
if __name__ == "__main__":
    torch.manual_seed(42)  # Ensure reproducibility
    try:
        model, accuracy = main()
        print(f"Final test accuracy: {accuracy:.2f}%")
        print("Program finished successfully!")
    except Exception as e:
        import traceback
        print(f"An error occurred while running the program: {e}")
        traceback.print_exc()
        print("\nAn error occurred, but you can still inspect saved results if available.")