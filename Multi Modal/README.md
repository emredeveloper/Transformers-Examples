# Gelişmiş Multimodal Transformer Modeli

Bu proje, gerçek video, ses ve metin verilerini işleyebilen gelişmiş bir multimodal (çoklu-modal) transformer modeli içermektedir. Model, verilen video, ses ve metin verilerini birleştirerek sınıflandırma yapmak üzere tasarlanmıştır.

## Özellikler

- Gerçek video dosyalarını işleme
- Gerçek ses dosyalarını işleme
- İlgili metin dosyalarını işleme
- Tüm modalitelerin füzyonu ile sınıflandırma
- Video için gelişmiş 3D-CNN mimarisi
- Ses için gelişmiş spektrogram işleme
- Türkçe metin desteği (BERT tabanlı)

## Proje İçeriği

- `basic-multimodal.py`: Ana kod dosyası
- `requirements.txt`: Gerekli Python paketleri
- `multimodal_dataset/`: Veri klasörü
  - `videos/`: Video dosyaları
  - `audios/`: Ses dosyaları
  - `texts/`: Metin dosyaları
  - `metadata.json`: Veri seti metadatası

## Kurulum

Gerekli paketleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Kullanım

Model, hem örnek veri ile hem de gerçek video, ses ve metin dosyalarıyla çalışabilir:

```bash
python basic-multimodal.py
```

Program çalıştığında size iki seçenek sunacaktır:
1. Örnek veri (otomatik oluşturulan demo)
2. Gerçek veri (kendi video, ses ve metin dosyalarınız)

Gerçek veri seçeneğini seçtiğinizde:
1. Video dosyalarınızı `multimodal_dataset/videos/` klasörüne koyun
2. Ses dosyalarınızı `multimodal_dataset/audios/` klasörüne koyun
3. Her örnek için metin açıklamalarını girebilir veya `multimodal_dataset/texts/` klasörüne koyabilirsiniz

## Model Mimarisi

Bu gelişmiş multimodal model üç temel bileşenden oluşmaktadır:

1. **Video Enkoder**: 3D CNN kullanarak videolardan özellik çıkarımı yapar
   - 224x224 çözünürlük
   - 16 frame işleme
   - AdaptiveAvgPool ve dropout katmanları

2. **Ses Enkoder**: Spektrogramlar üzerinde 2D CNN kullanarak seslerden özellik çıkarımı yapar
   - Mel spektrogramları
   - 128 mel-filtre bandı
   - 5 saniyelik ses örnekleri

3. **Metin Enkoder**: Türkçe BERT modeli kullanarak metinlerden özellik çıkarımı yapar

Çoklu-modal füzyon için:
- Transformer-tabanlı cross-attention
- Multi-head attention mekanizması
- Katmanlı normalizasyon

## Çıktılar

Model eğitim sonuçları, görseller ve eğitilen model `multimodal_dataset/` dizini içinde kaydedilir.

## Gereksinimler

- Python 3.7+
- PyTorch 1.9+
- transformers
- torchaudio
- torchvision
- OpenCV
- NumPy
- Matplotlib
