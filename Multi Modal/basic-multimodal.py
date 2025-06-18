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

# Cihaz yapılandırması
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Veri yolu ayarları
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multimodal_dataset")
os.makedirs(DATA_DIR, exist_ok=True)

# Veri seti yapısını oluşturalım
def create_real_data_metadata():
    """Gerçek video, ses ve metin dosyaları için metadata oluşturur"""
    
    # Veri dizinlerini oluştur
    video_dir = os.path.join(DATA_DIR, "videos")
    audio_dir = os.path.join(DATA_DIR, "audios")
    text_dir = os.path.join(DATA_DIR, "texts")
    
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    # Metadata dosyası için veri yapısı
    data_entries = []
    
    # Kullanıcıdan video dosyalarını yüklemesini iste
    print("\n" + "="*80)
    print("GERÇEK VERİ HAZIRLIĞI")
    print("="*80)
    print("Bu adımda gerçek video, ses ve metin dosyalarını kullanacağız.")
    print("Bunun için birkaç video dosyasını belirtilen klasörlere kopyaladıktan sonra metadatasını oluşturacağız.")
    print("\nAşağıdaki işlemleri manuel olarak yapmanız gerekiyor:")
    print(f"1. Video dosyalarınızı şu klasöre kopyalayın: {video_dir}")
    print(f"2. Ses dosyalarınızı şu klasöre kopyalayın: {audio_dir}")
    print(f"3. Her video/ses için metin dosyalarını şu klasöre kopyalayın: {text_dir}")
    print("4. Video, ses ve metin dosyalarının isimlerini eşleşecek şekilde numaralandırın.")
    print("   Örnek: video_1.mp4, audio_1.wav, text_1.txt")
    print("\nHazır olduğunuzda ENTER tuşuna basın...")
    input()
    
    # Dosyaları tara ve metadata oluştur
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for i, video_file in enumerate(video_files):
        video_id = i
        video_path = os.path.join(video_dir, video_file)
        
        # İlgili ses dosyasını bul (aynı isimde veya numarada olan)
        base_name = os.path.splitext(video_file)[0]
        audio_file = None
        for ext in ['.wav', '.mp3']:
            possible_audio = base_name + ext
            if os.path.exists(os.path.join(audio_dir, possible_audio)):
                audio_file = possible_audio
                break
        
        # Ses dosyası bulunamadıysa videodan ses çıkar
        audio_path = None
        if audio_file:
            audio_path = os.path.join(audio_dir, audio_file)
        else:
            # Yeni bir ses dosyası ismi oluştur
            audio_path = os.path.join(audio_dir, f"{base_name}.wav")
            
            # Videodan ses çıkar (FFmpeg gerekir - kullanıcıya bilgi ver)
            print(f"'{base_name}' için ses dosyası bulunamadı.")
            print(f"Ses dosyasını manuel olarak oluşturup '{audio_path}' konumuna kaydedin.")
            print("Hazır olduğunuzda ENTER tuşuna basın...")
            input()
        
        # İlgili metin dosyasını bul veya oluştur
        text_file = base_name + ".txt"
        text_path = os.path.join(text_dir, text_file)
        
        text = ""
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            # Metin dosyası yoksa kullanıcıdan metin girmesini iste
            print(f"'{base_name}' için metin açıklaması girin (video içeriğini açıklayan metin):")
            text = input().strip()
            # Metin dosyasını kaydet
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Metadatalara ekle
        data_entries.append({
            "id": video_id,
            "video_path": os.path.relpath(video_path, DATA_DIR),
            "audio_path": os.path.relpath(audio_path, DATA_DIR),
            "text": text,
            "text_path": os.path.relpath(text_path, DATA_DIR)
        })
    
    # JSON dosyasına kaydet
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data_entries, f, ensure_ascii=False, indent=4)
    
    print(f"Metadata oluşturuldu. Toplam {len(data_entries)} örnek.")
    return metadata_path


class MultiModalDataset(Dataset):
    """Multimodal veri seti: video, ses ve metin içeren bir dataset"""
    
    def __init__(self, metadata_path, max_length=128):
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_dir = os.path.dirname(metadata_path)
        
        # Metin tokenizeri
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.max_length = max_length
        
        # Video dönüşümleri
        self.video_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Ses dönüşümleri
        self.audio_transform = transforms.Compose([
            transforms.Normalize(mean=[-15], std=[40])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Metin işleme
        text_encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Ses işleme
        audio_path = os.path.join(self.data_dir, item["audio_path"])
        sample_rate, audio_data = wavfile.read(audio_path)
        # Int16'dan float32'ye dönüştür
        audio_data = audio_data.astype(np.float32) / 32767.0
        # Tensöre çevir ve mono olarak şekillendir
        waveform = torch.tensor(audio_data).float().unsqueeze(0)
        # Sabit uzunluğa getir (2 saniye)
        target_length = 2 * 16000
        if waveform.shape[1] < target_length:
            # Padding ekle
            padding = torch.zeros(waveform.shape[0], target_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        else:
            # Kes
            waveform = waveform[:, :target_length]
        
        # Spektrogram oluştur
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, n_mels=64
        )(waveform)
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        # İlk boyutu sıkıştır - MelSpectrogram çıkışı [channels, mel_bins, time] şeklinde,
        # biz sadece [mel_bins, time] formatına ihtiyacımız var
        spectrogram = spectrogram.squeeze(0)
        
        # Video işleme
        video_path = os.path.join(self.data_dir, item["video_path"])
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV BGR'den RGB'ye çevir
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.video_transform(frame)
            frames.append(frame)
        cap.release()
        
        # Sabit frame sayısı (10 frame)
        target_frames = 10
        if len(frames) > target_frames:
            # Düzenli aralıklarla örnekleme yap
            step = len(frames) // target_frames
            frames = [frames[i * step] for i in range(target_frames)]
        else:
            # Eksik frame'leri son frame ile doldur
            last_frame = frames[-1] if frames else torch.zeros(3, 64, 64)
            while len(frames) < target_frames:
                frames.append(last_frame)
        
        video_tensor = torch.stack(frames)
        
        return {
            "id": item["id"],
            "text_input_ids": text_encoding["input_ids"].squeeze(0),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(0),
            "audio": spectrogram,
            "video": video_tensor
        }


# Model mimarisi - Multimodal Fusion
class VideoEncoder(nn.Module):
    """Video kodlayıcı modül"""
    def __init__(self, embed_dim=256):
        super().__init__()
        # 3D CNN tabanlı enkoder
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        
        # Video özelliklerini projekte etmek için
        self.projection = nn.Sequential(
            nn.Linear(128 * 5 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
    def forward(self, x):
        # x girişi: [batch_size, frames, channels, height, width]
        # 3D CNN için: [batch_size, channels, frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = x.reshape(x.size(0), -1)
        x = self.projection(x)
        return x


class AudioEncoder(nn.Module):
    """Ses kodlayıcı modül"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Spektrogram örnek boyutları: Mel-bin=64, frame sayısı yaklaşık 161
        # 3 evrişim + havuzlama katmanı sonrası boyutlar: yaklaşık [batch, 128, 8, 20]
        # Lineer katman için boyutu ayarla
        self.projection = nn.Sequential(
            nn.Linear(128 * 8 * 20, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
    def forward(self, x):
        # x girişi: [batch_size, freq_bins, time_frames]
        x = x.unsqueeze(1)  # [batch_size, 1, freq_bins, time_frames]
        x = self.conv(x)
        # Burada boyut [batch_size, channels, height, width] olacak
        # Düzleştirmeden önce tam boyutları hesapla
        current_shape = x.size()
        x = x.reshape(x.size(0), -1)  # Düzleştirilmiş tensor
        # Linear katmanının giriş boyutunu kontrol et ve düzelt
        expected_linear_input = self.projection[0].in_features
        if x.size(1) != expected_linear_input:
            # Bu durum test sırasında oluşursa, kodun burada çalışmasını sağlar
            print(f"Uyarı: Ses özelliklerinin boyut uyuşmazlığı: Beklenen {expected_linear_input}, Mevcut {x.size(1)}")
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), expected_linear_input).squeeze(1)
        x = self.projection(x)
        return x


class TextEncoder(nn.Module):
    """Metin kodlayıcı modül - BERT tabanlı"""
    def __init__(self, embed_dim=256):
        super().__init__()
        # Türkçe BERT modelini kullan
        self.bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
        
        # BERT çıktısını projekte etmek için
        self.projection = nn.Linear(self.bert.config.hidden_size, embed_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        projected = self.projection(cls_token)
        return projected


class MultiModalTransformer(nn.Module):
    """Çoklu modal transformer modeli - Video, Ses ve Metin"""
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, output_dim=5):
        super().__init__()
        
        self.video_encoder = VideoEncoder(embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        
        # Transformer blokları
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Çıkış katmanı
        self.output_layer = nn.Linear(embed_dim*3, output_dim)
        
        # Embedding boyutu
        self.embed_dim = embed_dim
        
    def forward(self, video, audio, text_input_ids, text_attention_mask):
        # Modaliteleri ayrı ayrı kodla
        video_emb = self.video_encoder(video)
        audio_emb = self.audio_encoder(audio)
        text_emb = self.text_encoder(text_input_ids, text_attention_mask)
        
        # Özellikleri birleştir (concatenate)
        combined_features = torch.cat([video_emb, audio_emb, text_emb], dim=1)
        
        # Sınıflandırma çıktısı
        output = self.output_layer(combined_features)
        
        return output


# Eğitim ve değerlendirme fonksiyonları
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=5):
    """Model eğitim fonksiyonu"""
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Veriyi cihaza taşı
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            targets = batch["id"].to(device)  # ID'leri hedef olarak kullan
            
            # Veri boyutlarını yazdır (hata ayıklama için)
            if batch_idx == 0 and epoch == 0:
                print(f"Video boyutu: {video.shape}")
                print(f"Audio boyutu: {audio.shape}")
                print(f"Text input_ids boyutu: {text_input_ids.shape}")
            
            # Forward pass
            try:
                outputs = model(video, audio, text_input_ids, text_attention_mask)
                loss = criterion(outputs, targets)
                
                # Backward pass ve optimize et
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            except RuntimeError as e:
                print(f"Hata oluştu (batch {batch_idx}): {e}")
                print(f"Video shape: {video.shape}, Audio shape: {audio.shape}")
                continue
            
        # Epoch sonunda ortalama kaybı hesapla
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return train_losses

def evaluate_model(model, test_loader, criterion, device):
    """Model değerlendirme fonksiyonu"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Veriyi cihaza taşı
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            targets = batch["id"].to(device)
            
            try:
                # Forward pass
                outputs = model(video, audio, text_input_ids, text_attention_mask)
                loss = criterion(outputs, targets)
                
                # İstatistikleri hesapla
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            except RuntimeError as e:
                print(f"Değerlendirme sırasında hata oluştu (batch {batch_idx}): {e}")
                print(f"Video shape: {video.shape}, Audio shape: {audio.shape}")
                continue
    
    # Ortalama kayıp ve doğruluk oranı
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# Demo için örnek veri oluşturma
def create_sample_data():
    """Örnek multimodal veri oluşturur: video, ses ve metin (demo için)"""
    # Veri dosyalarını oluşturmak için dizinleri kontrol et
    video_dir = os.path.join(DATA_DIR, "videos")
    audio_dir = os.path.join(DATA_DIR, "audios")
    text_dir = os.path.join(DATA_DIR, "texts")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    # Metadata dosyası için veri yapısı
    data_entries = []
    
    # Örnek bir video oluştur (basit renkli kareler dizisi)
    for i in range(5):
        # Her örnek için
        video_frames = []
        for j in range(30):  # 30 frame'lik video
            # Renkli bir kare oluştur (RGB)
            if j < 10:
                frame = np.ones((64, 64, 3), dtype=np.uint8) * 50  # Koyu gri
            elif j < 20:
                frame = np.ones((64, 64, 3), dtype=np.uint8) * 150  # Orta gri
            else:
                frame = np.ones((64, 64, 3), dtype=np.uint8) * 250  # Açık gri
                
            # Her örnek için farklı bir renk bileşeni ekle
            frame[:,:,i % 3] = 200
            video_frames.append(frame)
        
        # Videoyu kaydet
        video_path = os.path.join(video_dir, f"sample_video_{i}.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 64))
        for frame in video_frames:
            out.write(frame)
        out.release()
        
        # Basit bir sinüs dalgası içeren ses dosyası oluştur
        audio_path = os.path.join(audio_dir, f"sample_audio_{i}.wav")
        sample_rate = 16000
        t = np.linspace(0, 2, 2 * sample_rate, endpoint=False)
        # Her örnek için farklı bir frekans
        frequency = 440 * (i + 1)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        # Stereo'ya çevir - scipy için 16-bit int'e dönüştür
        audio_data_16bit = (audio_data * 32767).astype(np.int16)
        # Mono ses olarak kaydet (scipy.io.wavfile ile)
        wavfile.write(audio_path, sample_rate, audio_data_16bit)
        
        # İlişkili metin oluştur
        if i == 0:
            text = "Bu video gri tonlamalı karelerden oluşmaktadır ve 440 Hz'lik bir ses içerir."
        elif i == 1:
            text = "Bu video kırmızı tonlarında karelerden oluşmaktadır ve 880 Hz'lik bir ses içerir."
        elif i == 2:
            text = "Bu video yeşil tonlarında karelerden oluşmaktadır ve 1320 Hz'lik bir ses içerir."
        elif i == 3:
            text = "Bu video mavi tonlarında karelerden oluşmaktadır ve 1760 Hz'lik bir ses içerir."
        else:
            text = "Bu video karışık tonlardaki karelerden oluşmaktadır ve 2200 Hz'lik bir ses içerir."
            
        # Metin dosyası kaydet
        text_path = os.path.join(text_dir, f"sample_text_{i}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Metadatalara ekle
        data_entries.append({
            "id": i,
            "video_path": os.path.relpath(video_path, DATA_DIR),
            "audio_path": os.path.relpath(audio_path, DATA_DIR),
            "text": text,
            "text_path": os.path.relpath(text_path, DATA_DIR)
        })
    
    # JSON dosyasına kaydet
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data_entries, f, ensure_ascii=False, indent=4)
    
    print(f"Örnek veri oluşturuldu. Toplam {len(data_entries)} örnek.")
    return metadata_path

# Ana fonksiyon
def main():
    """Ana çalıştırma fonksiyonu"""
    print("Multimodal model eğitimine başlıyoruz...")
    
    # Kullanıcıya veri tipi seçimi yaptır
    print("\nVeri tipi seçin:")
    print("1 - Örnek veri (otomatik oluşturulan demo verisi)")
    print("2 - Gerçek veri (gerçek video, ses ve metin dosyaları)")
    
    choice = input("Seçiminiz (1/2): ").strip()
    
    # Seçime göre veri oluştur
    if choice == "2":
        print("\nGerçek veri kullanılacak...")
        metadata_path = create_real_data_metadata()
    else:
        print("\nÖrnek demo verisi oluşturuluyor...")
        metadata_path = create_sample_data()
    
    # Veri setini hazırla
    dataset = MultiModalDataset(metadata_path)
    
    # Veri setini eğitim ve test olarak ayır
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Veri yükleyicileri
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # Model oluştur
    model = MultiModalTransformer(embed_dim=256, num_heads=4, num_layers=2, output_dim=5).to(device)
    print(f"Model oluşturuldu: {model.__class__.__name__}")
    
    # Kayıp fonksiyonu ve optimizer - daha düşük öğrenme oranı ile
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Modeli eğit
    print("Model eğitimi başlıyor...")
    train_losses = train_model(model, train_loader, optimizer, criterion, device, num_epochs=10)
    
    # Modeli değerlendir
    print("Model değerlendirmesi yapılıyor...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    
    # Sınıflandırma sonuçlarını detaylı analiz et
    print("\nModel Analizi:")
    print(f"- Toplam Eğitim Epoch: 10")
    print(f"- Son Eğitim Kaybı: {train_losses[-1]:.4f}")
    print(f"- Test Kaybı: {test_loss:.4f}")
    print(f"- Doğruluk Oranı: {test_accuracy:.2f}%")
    
    # Sonuçları görselleştir
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Eğitim Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Eğitim Kaybı')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DATA_DIR, "training_loss.png"))
    plt.show()
    
    # Modeli kaydet
    model_path = os.path.join(DATA_DIR, "multimodal_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    print(f"Model kaydedildi: {model_path}")
    
    # Test veri setinden bir örnek göster
    # Görselleştirme için orijinal dataset'i kullan (Subset sorununu önlemek için)
    print("Örnek bir veriyi görselleştirme...")
    # Orijinal veri setini görselleştirme için kullan, token çözümleme sorunundan kaçınmak için
    visualize_example(model, dataset, device)
    
    return model, test_accuracy

def visualize_example(model, dataset, device):
    """Test setinden bir örnek gösterimi"""
    # Rastgele bir örnek seç
    idx = np.random.randint(len(dataset))
    sample = dataset[idx]
    
    # Modeli değerlendirme moduna al
    model.eval()
    
    # Verileri tensöre dönüştür ve cihaza taşı
    video = sample["video"].unsqueeze(0).to(device)
    audio = sample["audio"].unsqueeze(0).to(device)
    text_input_ids = sample["text_input_ids"].unsqueeze(0).to(device)
    text_attention_mask = sample["text_attention_mask"].unsqueeze(0).to(device)
    
    # Veri boyutlarını yazdır
    print(f"Örnek görselleştirme - Video boyutu: {video.shape}")
    print(f"Örnek görselleştirme - Audio boyutu: {audio.shape}")
    
    # Tahmin yap
    predicted_class = None
    try:
        with torch.no_grad():
            output = model(video, audio, text_input_ids, text_attention_mask)
            _, predicted_class = torch.max(output, 1)
    except RuntimeError as e:
        print(f"Örnek görselleştirme sırasında hata: {e}")
        predicted_class = torch.tensor([-1]).to(device)  # Hata durumunda geçersiz sınıf
    
    # Gerçek sınıf
    true_class = sample["id"]
    
    # Sonuçları göster
    print(f"\nÖrnek Görselleştirme (Örnek {idx}):")
    print(f"Gerçek sınıf: {true_class}")
    if predicted_class is not None and predicted_class.item() != -1:
        print(f"Tahmin edilen sınıf: {predicted_class.item()}")
    else:
        print("Tahmin yapılamadı (model hatası)")
    
    # Videodan birkaç frame'i göster
    plt.figure(figsize=(15, 5))
    for i in range(min(5, video.size(1))):
        plt.subplot(1, 5, i+1)
        frame = video[0, i].cpu().permute(1, 2, 0)
        # Normalize edilmiş görüntüyü geri al
        frame = frame * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        frame = torch.clamp(frame, 0, 1)
        plt.imshow(frame)
        plt.title(f"Frame {i}")
        plt.axis('off')
    plt.savefig(os.path.join(DATA_DIR, "sample_frames.png"))
    plt.show()
    
    # Ses spektrogramını göster
    plt.figure(figsize=(10, 4))
    # Spektrogram verilerini kontrol et ve 2B bir tensöre dönüştür
    audio_data = sample["audio"].cpu()
    if len(audio_data.shape) == 1:
        # 1B tensörü 2B'ye genişlet
        audio_data = audio_data.unsqueeze(0)
    elif len(audio_data.shape) > 2:
        # İlk boyutu kullan
        audio_data = audio_data[0]
    
    plt.imshow(audio_data, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spektrogram')
    plt.xlabel('Zaman Çerçeveleri')
    plt.ylabel('Mel Filtre Bantları')
    plt.savefig(os.path.join(DATA_DIR, "sample_spectrogram.png"))
    plt.show()
    
    # Metni göster - tokenizer'a direkt erişim yerine özel tokenleri çıkartan basit bir yol kullan
    raw_text = ""
    try:
        # Subset içindeki dataset.dataset erişimi ile orijinal veri setine ulaşmaya çalış
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'tokenizer'):
            # Subset olduğunda
            tokenizer = dataset.dataset.tokenizer
            raw_text = tokenizer.decode(sample["text_input_ids"].tolist(), skip_special_tokens=True)
        else:
            # Direkt veri seti olduğunda
            raw_text = dataset.tokenizer.decode(sample["text_input_ids"].tolist(), skip_special_tokens=True)
    except Exception as e:
        # Tokenizer erişimi yoksa, özel token kodlarını temizleyen basit bir çözüm uygula
        text_tokens = sample["text_input_ids"].tolist()
        # 0, 101, 102 gibi özel token ID'lerini filtrele (BERT özel tokenleri)
        text_tokens = [t for t in text_tokens if t > 102 and t != 0]
        raw_text = f"ID'ler: {text_tokens} (Tokenizer erişilemediğinden ham metin gösterilemiyor)"
    
    print(f"Metin: {raw_text}")


# Ana programı çalıştır
if __name__ == "__main__":
    torch.manual_seed(42)  # Tekrarlanabilirlik için
    try:
        model, accuracy = main()
        print(f"Final test accuracy: {accuracy:.2f}%")
        print("Program başarıyla tamamlandı!")
    except Exception as e:
        import traceback
        print(f"Program çalıştırılırken bir hata oluştu: {e}")
        traceback.print_exc()
        print("\nHata oluştu, ancak eğer model kaydedildiyse sonuçları kontrol edebilirsiniz.")