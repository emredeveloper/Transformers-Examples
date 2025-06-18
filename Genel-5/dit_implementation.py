# Gerekli kütüphaneleri içe aktar
import torch  # PyTorch kütüphanesi
import torch.nn as nn  # Sinir ağı modülleri
import torch.nn.functional as F  # Fonksiyonel operasyonlar
import math  # Matematiksel işlemler
import numpy as np  # Sayısal işlemler
from typing import Optional, Tuple  # Tip ipuçları
import matplotlib.pyplot as plt  # Görselleştirme için

class TimestepEmbedding(nn.Module):
    """Zaman adımları için sinüzoidal gömme vektörleri oluşturur, transformer pozisyonel kodlamalarına benzer"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim  # Gömme boyutu
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device  # Hesaplamanın yapılacağı cihaz (CPU/GPU)
        half_dim = self.dim // 2  # Boyutun yarısı
        # Logaritmik ölçekli frekanslar oluştur
        embeddings = math.log(10000) / (half_dim - 1)
        # Üstel fonksiyonla frekansları hesapla
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Zaman adımlarıyla çarparak gömme matrisini oluştur
        embeddings = timesteps[:, None] * embeddings[None, :]
        # Sinüs ve kosinüs değerlerini birleştir
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class MultiHeadAttention(nn.Module):
    """Çok kafalı dikkat mekanizması"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Model boyutunun kafa sayısına bölünebilir olması gerekir
        assert d_model % n_heads == 0
        
        self.d_model = d_model  # Girdi boyutu
        self.n_heads = n_heads    # Kafa sayısı
        self.d_k = d_model // n_heads  # Her kafanın boyutu
        
        # Sorgu, anahtar, değer ve çıkış dönüşümleri
        self.w_q = nn.Linear(d_model, d_model)  # Sorgu dönüşümü
        self.w_k = nn.Linear(d_model, d_model)  # Anahtar dönüşümü
        self.w_v = nn.Linear(d_model, d_model)  # Değer dönüşümü
        self.w_o = nn.Linear(d_model, d_model)  # Çıkış dönüşümü
        
        self.dropout = nn.Dropout(dropout)  # Aşırı öğrenmeyi önlemek için dropout
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape  # Girdi boyutlarını al
        
        # Çok kafalı dikkat için doğrusal dönüşümler ve yeniden şekillendirme
        # Sorgu, anahtar ve değer matrislerini hesapla ve kafalara böl
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Ölçeklendirilmiş nokta çarpımı dikkat mekanizması
        # Anahtarların transpozu ile sorguları çarp ve ölçeklendir
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Maske uygula (varsa)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Dikkat ağırlıklarını hesapla ve softmax uygula
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # Dropout uygula
        
        # Dikkat ağırlıklarını değerlerle çarparak çıktıyı hesapla
        attention_output = torch.matmul(attention_weights, V)
        
        # Kafaları birleştir ve son lineer katmandan geçir
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attention_output)  # Son lineer dönüşümü uygula

class FeedForward(nn.Module):
    """Konum bazlı ileri beslemeli ağ"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # Girişten gizli katmana
        self.linear2 = nn.Linear(d_ff, d_model)   # Gizli katmandan çıkışa
        self.dropout = nn.Dropout(dropout)         # Aşırı öğrenmeyi önlemek için
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # İleri yayılım: Lineer -> ReLU -> Dropout -> Lineer
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Dikkat ve ileri beslemeli katmanlara sahip tek bir transformer bloğu"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)  # Çok kafalı dikkat katmanı
        self.feed_forward = FeedForward(d_model, d_ff, dropout)         # İleri beslemeli ağ
        self.norm1 = nn.LayerNorm(d_model)  # İlk normalizasyon katmanı
        self.norm2 = nn.LayerNorm(d_model)  # İkinci normalizasyon katmanı
        self.dropout = nn.Dropout(dropout)   # Dropout katmanı
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Öz-dikkat mekanizması ve artık bağlantı
        attn_output = self.attention(self.norm1(x), mask)  # Normalizasyon ve dikkat
        x = x + self.dropout(attn_output)  # Artık bağlantı ve dropout
        
        # İleri beslemeli ağ ve artık bağlantı
        ff_output = self.feed_forward(self.norm2(x))  # Normalizasyon ve ileri besleme
        x = x + self.dropout(ff_output)  # İkinci artık bağlantı ve dropout
        
        return x

class DiffusionTransformer(nn.Module):
    """
    Görüntü oluşturma için Diffusion Transformer (DiT) modeli
    
    Argümanlar:
        img_size: Giriş görüntülerinin boyutu (kare olduğu varsayılır)
        patch_size: Görüntünün bölüneceği yama boyutu
        d_model: Transformer'ın gizli boyutu
        n_layers: Transformer katman sayısı
        n_heads: Dikkat başlığı sayısı
        d_ff: İleri beslemeli ağın gizli boyutu
        num_classes: Koşullu üretim için sınıf sayısı
        dropout: Dropout oranı
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
        
        self.img_size = img_size  # Görüntü boyutu
        self.patch_size = patch_size  # Yama boyutu
        self.d_model = d_model  # Modelin gizli boyutu
        self.num_patches = (img_size // patch_size) ** 2  # Toplam yama sayısı
        self.patch_dim = 3 * patch_size ** 2  # RGB yamaları için boyut (3 kanal * yama alanı)
        
        # Yama gömme katmanı
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        
        # Konumsal gömme (pozisyonel kodlama)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Zaman adımı gömme
        self.time_embedding = TimestepEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Zaman gömme için MLP
            nn.GELU(),  # Gaussian Error Linear Unit aktivasyonu
            nn.Linear(d_model * 4, d_model)  # Çıkış katmanı
        )
        
        # Sınıf gömme (koşullu üretim için)
        self.class_embedding = nn.Embedding(num_classes, d_model)
        
        # Transformer katmanları
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)  # Belirtilen sayıda transformer katmanı oluştur
        ])
        
        # Çıkış projeksiyonu
        self.norm = nn.LayerNorm(d_model)  # Son normalizasyon katmanı
        self.output_projection = nn.Linear(d_model, self.patch_dim)  # Çıkış boyutuna dönüşüm
        
        self.dropout = nn.Dropout(dropout)  # Dropout katmanı
        
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Görüntüyü yamalara dönüştür"""
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
        """Yamaları tekrar görüntüye dönüştür"""
        batch_size = x.shape[0]  # Toplu iş boyutu
        height = width = int(self.num_patches ** 0.5)  # Orijinal ızgara boyutları
        
        # Yamaları tekrar orijinal formata dönüştür
        x = x.reshape(
            batch_size, height, width, 3, self.patch_size, self.patch_size
        )
        # Boyutları yeniden düzenle: [batch, channels, height, patch_h, width, patch_w]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        # Yama boyutlarını birleştirerek orijinal görüntü boyutuna getir
        x = x.reshape(batch_size, 3, height * self.patch_size, width * self.patch_size)
        
        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Diffusion transformer'ın ileri geçişi
        
        Argümanlar:
            x: Girdi tensörü (batch_size, channels, height, width)
            timesteps: Toplu işteki her örnek için zaman adımı
            class_labels: Koşullu üretim için isteğe bağlı sınıf etiketleri
            
        Dönüş:
            Girdiyle aynı şekilde gürültü tahmini
        """
        batch_size = x.shape[0]  # Toplu iş boyutu
        device = x.device  # Hesaplama cihazı
        
        # Görüntüyü yamalara dönüştür
        x = self.patchify(x)  # Yamalara dönüştürme
        
        # Yamaları gömme boyutuna yansıt
        x = self.patch_embedding(x)  # Yama gömme
        
        # Konumsal gömme ekle
        x = x + self.pos_embedding  # Konumsal bilgi ekle
        
        # Zaman adımı gömme ekle
        t_emb = self.time_embedding(timesteps)  # Zaman adımları için gömme
        t_emb = self.time_mlp(t_emb)  # Zaman gömme için MLP'den geçir
        x = x + t_emb.unsqueeze(1)  # Zamansal bilgiyi ekle
        
        # İsteğe bağlı olarak sınıf gömme ekle
        if class_labels is not None:
            class_emb = self.class_embedding(class_labels)  # Sınıf gömme
            x = x + class_emb.unsqueeze(1)  # Sınıf bilgisini ekle
        
        # Transformer katmanlarını uygula
        for layer in self.transformer_layers:
            x = layer(x)  # Her bir transformer katmanından geçir
        
        # Son normalizasyon katmanı
        x = self.norm(x)  # Normalizasyon
        
        # Yama boyutuna geri yansıt
        x = self.output_projection(x)  # Çıkış projeksiyonu
        
        # Yamaları tekrar görüntüye dönüştür
        x = self.unpatchify(x)  # Görüntüye dönüştür
        
        return x  # Gürültü tahminini döndür

class DDPMScheduler:
    """Diffusion süreci için DDPM gürültü çizelgeleyici"""
    
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps  # Toplam zaman adımı sayısı
        
        # Doğrusal beta çizelgesi
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)  # Beta değerleri
        self.alphas = 1.0 - self.betas  # Alfa değerleri (1 - beta)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # Kümülatif çarpım
        # Önceki kümülatif çarpım (kaydırmalı)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # q(x_t | x_{t-1}) dağılımı için hesaplamalar
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # Karekök alfa kümülatif çarpım
        # Karekök (1 - alfa kümülatif çarpım)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # q(x_{t-1} | x_t, x_0) posterior dağılımı için hesaplamalar
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Temiz görüntülere gürültü çizelgesine göre gürültü ekle"""
        # İlgili zaman adımları için ölçeklendirme faktörlerini al
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(x_0.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(x_0.device)
        
        # Temiz görüntülere gürültü ekle
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        
        return x_t  # Gürültülü görüntüyü döndür
        
    def sample_prev_timestep(self, x_t: torch.Tensor, noise_pred: torch.Tensor, timestep: int) -> torch.Tensor:
        """x_t ve tahmin edilen gürültü verildiğinde x_{t-1} örnekle"""
        if timestep == 0:
            return x_t  # Son zaman adımında, sadece tahmin edilen x_0'ı döndür
            
        # Bu zaman adımı için parametreleri al
        alpha_t = self.alphas[timestep]  # Mevcut alfa
        alpha_cumprod_t = self.alphas_cumprod[timestep]  # Kümülatif alfa
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[timestep]  # Önceki kümülatif alfa
        beta_t = self.betas[timestep]  # Mevcut beta
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep]  # Karekök(1 - alfa kümülatif)
        
        # Ters süreç için varyans hesapla
        posterior_variance_t = self.posterior_variance[timestep]
        
        # x_t ve tahmin edilen gürültüden x_0'ı tahmin et
        pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        # q(x_{t-1} | x_t, x_0) dağılımının ortalamasını hesapla
        mean = (torch.sqrt(alpha_cumprod_prev_t) * beta_t * pred_x0 + 
                torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) * x_t) / (1 - alpha_cumprod_t)
        
        # q(x_{t-1} | x_t, x_0) dağılımından örnekle
        if timestep > 0:
            noise = torch.randn_like(x_t)  # Rastgele gürültü üret
            variance = torch.sqrt(posterior_variance_t) * noise  # Varyansı uygula
        else:
            variance = 0  # Son adımda varyans yok
            
        x_prev = mean + variance  # Ortalama ve varyansı topla
        
        return x_prev  # Önceki zaman adımındaki görüntüyü döndür

def train_step(model: DiffusionTransformer, 
               scheduler: DDPMScheduler, 
               x_batch: torch.Tensor, 
               class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Diffusion transformer için tek bir eğitim adımı"""
    
    batch_size = x_batch.shape[0]  # Toplu iş boyutu
    device = x_batch.device  # Hesaplama cihazı
    
    # Toplu işteki her görüntü için rastgele zaman adımları seç
    timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
    
    # Görüntülere eklenecek gürültüyü örnekle
    noise = torch.randn_like(x_batch)  # Gürültü tensörü oluştur
    
    # Temiz görüntülere, her zaman adımındaki gürültü büyüklüğüne göre gürültü ekle
    noisy_images = scheduler.add_noise(x_batch, noise, timesteps)
    
    # Gürültü artığını tahmin et
    noise_pred = model(noisy_images, timesteps, class_labels)
    
    # Kaybı hesapla (tahmin edilen gürültü ile gerçek gürültü arasındaki ortalama kare hata)
    loss = F.mse_loss(noise_pred, noise)
    
    return loss  # Hata değerini döndür

@torch.no_grad()
def sample_images(
    model: DiffusionTransformer, 
    scheduler: DDPMScheduler, 
    num_samples: int = 4, 
    class_labels: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """Eğitilmiş diffusion transformer kullanarak örnek görüntüler oluştur"""
    model.eval()  # Modeli değerlendirme moduna al
    
    # Başlangıç gizli değişkeni olarak rastgele gürültü örnekle
    img_size = model.img_size  # Görüntü boyutu
    x_t = torch.randn((num_samples, 3, img_size, img_size), device=device)
    
    # Eğer sınıf etiketleri verilmediyse ve model koşullu ise rastgele sınıf etiketleri oluştur
    if class_labels is None and hasattr(model, 'class_embedding'):
        num_classes = model.class_embedding.num_embeddings  # Toplam sınıf sayısı
        class_labels = torch.randint(0, num_classes, (num_samples,), device=device)  # Rastgele sınıf etiketleri
    
    # Modelden örnekleme yap
    with torch.no_grad():  # Gradyan hesaplaması yapma
        # Zaman adımlarını tersten dolaş
        for t in reversed(range(scheduler.num_timesteps)):
            # Zaman adımları için tensor oluştur
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Gürültüyü tahmin et
            noise_pred = model(x_t, timesteps, class_labels)
            
            # Bir önceki örneği al
            x_t = scheduler.sample_prev_timestep(x_t, noise_pred, t)
    
    # Geçerli piksel aralığına kırp
    x_t = torch.clamp(x_t, -1.0, 1.0)
    
    # [-1, 1] aralığından [0, 1] aralığına ölçekle
    x_t = (x_t + 1) / 2
    
    return x_t  # Oluşturulan görüntüleri döndür

# Example usage and training loop
def example_usage():
    """Diffusion transformer'ın nasıl kullanılacağını göster"""
    # Hesaplama cihazını ayarla (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Modeli ve çizelgeleyiciyi başlat
    model = DiffusionTransformer(
        img_size=32,     # Görüntü boyutu
        patch_size=4,    # Yama boyutu
        d_model=256,     # Modelin gizli boyutu
        n_layers=6,      # Transformer katman sayısı
        n_heads=8,       # Dikkat başlığı sayısı
        d_ff=1024,       # İleri beslemeli ağın gizli boyutu
        num_classes=10,  # Sınıf sayısı (CIFAR-10 için)
        dropout=0.1      # Dropout oranı
    ).to(device)  # Modeli uygun cihaza taşı
    
    # Gürültü çizelgeleyiciyi başlat
    scheduler = DDPMScheduler(num_timesteps=1000)
    
    # Örnek veri oluştur
    batch_size = 4  # Toplu iş boyutu
    x = torch.randn(batch_size, 3, 32, 32, device=device)  # Rastgele giriş görüntüleri
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)  # Rastgele zaman adımları
    class_labels = torch.randint(0, 10, (batch_size,), device=device)  # Rastgele sınıf etiketleri
    
    # İleri geçiş
    noise_pred = model(x, timesteps, class_labels)  # Gürültü tahmini yap
    print(f"Girdi şekli: {x.shape}")
    print(f"Gürültü tahmini şekli: {noise_pred.shape}")
    
    # Eğitim adımı
    loss = train_step(model, scheduler, x, class_labels)  # Eğitim adımını çalıştır
    print(f"Eğitim kaybı: {loss.item():.4f}")
    
    # Örnek görüntüler oluştur
    samples = sample_images(model, scheduler, num_samples=4, device=device)
    print(f"Oluşturulan örneklerin şekli: {samples.shape}")
    
    # Örnekleri görselleştir
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))  # 1x4'lük bir ızgara oluştur
    for i, ax in enumerate(axes):
        # Görüntüyü [C, H, W]'dan [H, W, C]'ye çevir ve göster
        ax.imshow(samples[i].permute(1, 2, 0).cpu().numpy())
        ax.axis('off')  # Eksenleri kapat
    plt.tight_layout()  # Görsel düzenlemeyi iyileştir
    plt.show()  # Görseli göster
    
    # Eğitim döngüsunu başlat
    print("Eğitim başlatılıyor...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Modeli eğitim moduna al
    model.train()
    
    # Tek bir eğitim adımı gerçekleştir
    loss = train_step(model, scheduler, x, class_labels)
    
    # Geri yayılım ve parametre güncelleme
    optimizer.zero_grad()  # Gradyanları sıfırla
    loss.backward()  # Gradyanları hesapla
    optimizer.step()  # Parametreleri güncelle
    
    print(f"Eğitim kaybı: {loss.item():.4f}")
    
    # Örnek görüntüler oluştur
    print("Örnek görüntüler oluşturuluyor...")
    sample_labels = torch.arange(4, device=device)  # İlk 4 sınıf için birer örnek oluştur
    generated_images = sample_images(model, scheduler, num_samples=4, class_labels=sample_labels, device=device)
    
    print(f"Oluşturulan görüntülerin boyutu: {generated_images.shape}")
    print("Örnek oluşturma tamamlandı!")
    
    return model, scheduler, generated_images  # Modeli, çizelgeleyiciyi ve oluşturulan görüntüleri döndür

# Run example
if __name__ == "__main__":
    model, scheduler, samples = example_usage()