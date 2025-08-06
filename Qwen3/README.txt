# QWEN3 TÜRKÇE DİL MODELİ

Bu proje, Qwen3 mimarisini baz alan bir Türkçe dil modeli implementasyonudur. Temel amacı, Türkçe finans alanı sorularına cevap verebilen, düşünme süreçlerini modelleyebilen bir yapay zeka modeli oluşturmaktır.

## MODEL ÖZELLİKLERİ

### Mimari Bileşenleri
- **Temel Yapı**: Transformer mimarisi (Encoder-Decoder değil, yalnızca Decoder tabanlı)
- **Parametre Sayısı**: 100M+ (büyük model konfigürasyonu)
- **Konumsal Kodlama**: Sinüzoidal konumsal kodlama
- **Dikkat Mekanizması**: Gruplandırılmış Sorgu Dikkati (GQA - Grouped Query Attention)
- **Normalizasyon**: LayerNorm (RMSNorm yerine basitlik için)
- **Aktivasyon Fonksiyonu**: GELU

### Özel Özellikler
1. **Düşünme Modu**: Model, cevap vermeden önce "düşünme" sürecini simüle edebilir
   - <think> ve </think> özel tokenları ile işaretlenen düşünme adımları
   - Düşünme adımları sonrası daha iyi yanıtlar üretebilme
   
2. **Türkçe Tokenizer**: Türkçe karakterler için özelleştirilmiş basit tokenizer
   - Türkçe karakter seti desteği (ç, ğ, ı, ö, ş, ü vb.)
   - Özel tokenlar için rezerve edilmiş ID'ler
   
3. **Soru-Cevap Formatlama**: Finans alanı sorularına özel QA formatı

### Model Boyutlandırma Parametreleri
- **Vocab Boyutu**: 50,000 token
- **Gizli Boyut (Hidden Size)**: 1024
- **Katman Sayısı**: 24
- **Q Başlık Sayısı**: 16
- **KV Başlık Sayısı**: 8
- **FFN Boyutu**: 4096
- **Maksimum Dizi Uzunluğu**: 2048 token

## VERİ SETİ

- **Kaynak**: umarigan/turkiye_finance_qa (HuggingFace)
- **İçerik**: 428 Türkçe finans soru-cevap çifti
- **Format**: "Soru: {soru}\nCevap: {cevap}"

## EĞİTİM ÖZELLİKLERİ

- **Optimizer**: AdamW (öğrenme oranı: 1e-5)
- **Batch Boyutu**: 2 (büyük model için hafıza optimizasyonu)
- **Gradyan Clipping**: 1.0 maksimum norm
- **Dropout**: 0.1
- **Text Generation**: 
  - Top-k sampling (k=50)
  - Top-p sampling (p=0.9)
  - Sıcaklık (temperature): 0.7

## KULLANIM

Model şu şekilde kullanılabilir:
1. Standart metin üretimi için `generate_text` fonksiyonu
2. Düşünme modunda üretim için `think_mode=True` parametresi

## TEKNİK DETAYLAR

### Gruplandırılmış Sorgu Dikkati (GQA)
Q için 16 başlık, KV için 8 başlık kullanılarak bellek verimliliği sağlanmıştır. 
Her KV başlığı birden fazla Q başlığı tarafından paylaşılır.

### Veri İşleme
1. Veri tokenize edilir
2. Batch halinde gruplandırılır
3. Attention mask oluşturulur
4. Causal maskeleme uygulanır

### Otomatik Regresif Üretim
Model, bir sonraki token tahminlerini daha önce üretilen tokenleri kullanarak gerçekleştirir.

## PERFORMANS

Model eğitim sonrası finans alanındaki soruları cevaplayabilme yeteneğine sahiptir. 
Düşünme modu aktivasyonu ile daha karmaşık sorularda iyileştirilmiş yanıtlar sağlayabilir.

## SINIRLAMALAR

- CPU ile eğitim uzun sürebilir
- Türkçe karakterler için özel tokenizer basit olduğundan büyük dil modellerindeki subword tokenizer kadar etkili değildir
- Veri seti küçük olduğundan (428 örnek) modelin genelleme yeteneği sınırlı olabilir
