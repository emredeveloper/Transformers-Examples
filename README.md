# Transformers Examples

Bu repository, modern derin öğrenme modellerinin farklı yönlerini gösteren Transformers kütüphanesi kullanılarak geliştirilmiş çeşitli örnekler ve implementasyonlar içerir. Dil modelleri, vision transformers, multimodal modeller ve daha fazlasını kapsar.

## 📁 Repository Yapısı

### Ana Dizinler

- **`Architecture/`** - **YENİ!** RoPE (Rotary Position Embedding) karşılaştırmaları ve transformer mimarisi örnekleri
- **`Genel-1/`** - Temel transformer implementasyonları ve konfigürasyon örnekleri
- **`Genel-2/`** - Gelişmiş transformer modelleri (vision transformers ve multimodal örnekler)
- **`Genel-3/`** - Ek transformer varyantları ve deneyler
- **`Genel-4/`** - Performans karşılaştırmaları ve fine-tuning örnekleri
- **`Genel-5/`** - İleri teknikler ve model optimizasyonları
- **`Multi Modal/`** - Video, ses ve metin için multimodal transformer implementasyonları
- **`Vision Transformers/`** - Vision transformer modelleri ve uygulamaları
- **`Time series - Transformers/`** - Transformer modelleri kullanarak zaman serisi analizi
- **`Tokenizer/`** - Özel tokenizer implementasyonları ve eğitimi
- **`llama/`** - LLaMA model implementasyonu ve utilities
- **`Qwen3/`** - Qwen 3 model örnekleri ve kullanımı
- **`finetuned-llm/`** - Fine-tuned dil modeli checkpoint'leri
- **`archive/`** - MMLU benchmark sonuçları ve arşivlenmiş dosyalar

### Önemli Dosyalar

- **`test-time-scaling.py`** - Dil modelleri için test-time scaling implementasyonu
- **`requirements.txt`** - Temel Python bağımlılıkları
- **`requirements-jax.txt`** - JAX ekosistemi için ek bağımlılıklar
- **`requirements-dev.txt`** - Geliştirme ve ileri seviye eğitim araçları
- **`setup.sh`** - Otomatik kurulum script'i
- **`.env.example`** - Çevre değişkenleri şablonu
- **`CONTRIBUTING.md`** - Katkıda bulunma rehberi

## 🚀 Hızlı Başlangıç

### Gereksinimler

Sisteminizde Python 3.7+ yüklü olduğundan emin olun.

### Kurulum

**Otomatik Kurulum (Önerilen):**

```bash
# Repository'yi klonlayın
git clone https://github.com/emredeveloper/Transformers-Examples.git
cd Transformers-Examples

# Otomatik kurulum script'ini çalıştırın (varsayılan profil: base)
chmod +x setup.sh
./setup.sh --venv
# JAX veya geliştirme bağımlılıklarını da eklemek için:
# ./setup.sh --profile jax
# ./setup.sh --profile dev
# ./setup.sh --profile all
```

**Manuel Kurulum:**

1. Repository'yi klonlayın:

```bash
git clone https://github.com/emredeveloper/Transformers-Examples.git
cd Transformers-Examples
```

2. Virtual environment oluşturun (önerilen):

```bash
python -m venv .venv
# Windows için:
.venv\Scripts\activate
# Linux/Mac için:
source .venv/bin/activate
```

3. Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
# JAX örnekleri için ek bağımlılıklar:
# pip install -r requirements-jax.txt
# Geliştirme araçları için:
# pip install -r requirements-dev.txt
```

### Bağımlılık Profilleri

- **Base (`requirements.txt`)**: PyTorch, Transformers ve çoğu örnek için gerekli çekirdek paketler.
- **JAX (`requirements-jax.txt`)**: JAX tabanlı deneyler ve örnekler için gerekli `jax`, `jaxlib` ve `flax` paketleri.
- **Development (`requirements-dev.txt`)**: Not defterleri, büyük ölçekli eğitim yardımcıları ve gelişmiş araçlar (`jupyter`, `notebook`, `fairscale`, `deepspeed`).

`setup.sh` script'i bu profilleri `--profile` parametresiyle otomatik olarak yükleyebilir. Varsayılan profil `base`'dir.

4. Çevre değişkenlerini ayarlayın:

```bash
# .env.example dosyasını .env olarak kopyalayın
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# .env dosyasını düzenleyip Hugging Face token'ınızı ekleyin
```

## 📖 Kullanım Örnekleri

### RoPE Karşılaştırması (YENİ!)

```bash
cd Architecture
python partial-rope.py
```

### Temel Transformer Kullanımı

```bash
cd Genel-1
python app.py
```

### Vision Transformers

```bash
cd "Vision Transformers"
jupyter notebook sglip2.ipynb
```

### Multimodal Örnekler

```bash
cd "Multi Modal"
python basic-multimodal.py
```

### LLaMA Modeli

```bash
cd llama
python run_cpu.py
```

### Tokenizer Eğitimi

```bash
cd Tokenizer
python tokenizer.py
```

### Test-Time Scaling

```bash
python test-time-scaling.py
```

## ⚙️ Konfigürasyon

Birçok örnek çevre değişkenleri aracılığıyla konfigürasyonu destekler:

- `HUGGINGFACE_TOKEN`: Hugging Face API token'ınız
- `CUDA_VISIBLE_DEVICES`: GPU cihaz seçimi
- `MODEL_CACHE_DIR`: İndirilen modeller için cache dizini

## 📝 Örneklere Genel Bakış

### Dil Modelleri

- GPT-2 konfigürasyonu ve fine-tuning
- DeepSeek transformer implementasyonları
- Qwen 3 model kullanımı
- Test-time scaling teknikleri
- RoPE (Rotary Position Embedding) karşılaştırmaları

### Vision Modelleri

- Vision Transformer (ViT) implementasyonları
- SGLIP-2 multimodal anlayış
- Görüntü sınıflandırma örnekleri

### Multimodal Modeller

- Video, ses ve metin işleme
- Cross-modal attention mekanizmaları
- Multimodal fusion teknikleri

### Zaman Serileri

- Transformer tabanlı zaman serisi tahmini
- Sequence-to-sequence modelleme

### İleri Teknikler

- Mixture of Experts (MoE)
- Cross-attention mekanizmaları
- Özel tokenization stratejileri
- Model optimizasyon teknikleri
- Partial RoPE implementasyonları

## 🔧 Yeni Özellikler

### Architecture Dizini

Bu dizin transformer mimarisi ile ilgili gelişmiş örnekler içerir:

- **`partial-rope.py`**: Partial RoPE vs Full RoPE performans karşılaştırması
- Detaylı benchmark sonuçları ve görselleştirmeler
- Bellek kullanımı analizleri
- Ablasyon çalışmaları

## 🤝 Katkıda Bulunma

Katkılar memnuniyetle karşılanır! Lütfen Pull Request göndermekten çekinmeyin. Büyük değişiklikler için, önce ne değiştirmek istediğinizi tartışmak üzere bir issue açın.

Detaylı bilgi için `CONTRIBUTING.md` dosyasını kontrol edin.

## 📄 Lisans

Bu proje açık kaynaklıdır ve MIT Lisansı altında mevcuttur.

## 🔍 Notlar

- Bazı örnekler özel model erişim izinleri gerektirir
- Büyük modelleri çalıştırmak için GPU önerilir
- Belirli gereksinimler için bireysel dizin README dosyalarını kontrol edin
- Hugging Face modelleri için uygun kimlik doğrulaması ayarladığınızdan emin olun
- `.env` dosyasını oluşturmayı ve API token'larınızı eklemeyi unutmayın

## 🐛 Sorun Giderme

### Yaygın Sorunlar

1. **Import hataları**: Tüm bağımlılıkların yüklü olduğundan emin olun
2. **CUDA hataları**: GPU kullanılabilirliğini ve CUDA kurulumunu kontrol edin
3. **Model erişimi**: Özel modeller için uygun izinlere sahip olduğunuzdan emin olun
4. **Bellek hataları**: Daha küçük batch boyutları veya model varyantları kullanmayı düşünün
5. **Token hataları**: `.env` dosyasında Hugging Face token'ınızın doğru ayarlandığından emin olun

Daha detaylı yardım için, lütfen belirli dizin belgelerini kontrol edin veya bir issue açın.

## 📊 Benchmark Sonuçları

Repository, çeşitli transformer varyantları için performans karşılaştırmaları içerir:

- RoPE implementasyonları arasındaki hız ve doğruluk karşılaştırmaları
- MMLU benchmark sonuçları (archive/ dizininde)
- Model optimizasyon teknikleri analizi

Detaylı sonuçlar için `Architecture/` dizinini ve generate edilen PNG dosyalarını kontrol edin. 

## ✅ Test ve Kod Kalitesi

Hafif testleri ve kod kalite kontrollerini çalıştırmak için isteğe bağlı geliştirme bağımlılıklarını yükleyin:

```bash
pip install -r requirements-dev.txt
```

Ardından aşağıdaki komutları çalıştırabilirsiniz:

```bash
pytest
ruff check tests
black --check tests
```

Sürekli entegrasyon iş akışı bu kontrolleri otomatik olarak yürütür.
