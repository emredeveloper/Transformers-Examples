# DiT (Diffusion Transformer) Kodunun Detaylı Açıklaması

## 1. Genel Bakış
Bu kod, görsel üretimi için kullanılan **Diffusion Transformer (DiT)** mimarisinin dinamik bir versiyonudur. Temel amacı, metinden görsel üretmek veya görsel iyileştirme yapmaktır.

## 2. Yardımcı Fonksiyonlar

### `round_to_nearest` Fonksiyonu
```python
def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
```
- **Amacı**: Ağın boyutlarını attention head sayısına göre uygun şekilde yuvarlar
- **Kullanımı**: Dinamik genişlik ayarlaması için
- **Çalışma Prensibi**: width_mult parametresini num_heads'e göre normalize eder

## 3. Dinamik Linear Katmanlar

### `DynaLinear` Sınıfı
```python
class DynaLinear(nn.Linear):
```
- **Amacı**: Çalışma zamanında boyutu değişebilen linear katman
- **Özellikler**:
  - `in_features` ve `out_features` dinamik olarak ayarlanabilir
  - `width_mult` parametresi ile genişlik kontrolü
  - `dyna_dim` ile hangi boyutların dinamik olacağı belirlenir

### `DynaQKVLinear` Sınıfı
```python
class DynaQKVLinear(nn.Linear):
```
- **Amacı**: Attention mekanizması için Query, Key, Value matrislerini üreten dinamik katman
- **Özellikler**:
  - QKV'yi tek seferde hesaplar (3 ayrı matris)
  - `einops` kütüphanesi ile tensor reshape işlemleri
  - Dinamik boyut ayarlaması

## 4. Attention Mekanizması

### `Attention` Sınıfı
```python
class Attention(nn.Module):
```
- **Amacı**: Multi-head self-attention mekanizması
- **Bileşenler**:
  - `qkv`: Query, Key, Value üretimi için DynaQKVLinear
  - `q_norm`, `k_norm`: Query ve Key normalizasyonu
  - `proj`: Çıkış projeksiyonu için DynaLinear
  - `channel_mask`: Dinamik kanal maskeleme

**Çalışma Prensibi**:
1. Input tensor'dan Q, K, V üretir
2. Attention skorlarını hesaplar
3. Channel mask uygulanabilir
4. Sonucu project eder

## 5. MLP (Multi-Layer Perceptron)

### `Mlp` Sınıfı
```python
class Mlp(nn.Module):
```
- **Amacı**: Feed-forward ağ
- **Yapısı**:
  - `fc1`: İlk linear katman (genişletme)
  - `act`: Aktivasyon fonksiyonu (GELU)
  - `fc2`: İkinci linear katman (daraltma)
  - Channel masking desteği

## 6. Embedding Katmanları

### `TimestepEmbedder` Sınıfı
```python
class TimestepEmbedder(nn.Module):
```
- **Amacı**: Diffusion timestep'lerini vektör formatına çevirir
- **Yöntem**: Sinusoidal embedding + MLP
- **Kullanımı**: Diffusion process'inde hangi adımda olduğumuzu modele söyler

### `LabelEmbedder` Sınıfı
```python
class LabelEmbedder(nn.Module):
```
- **Amacı**: Sınıf etiketlerini embedding'e çevirir
- **Özellikler**:
  - Classifier-free guidance için dropout desteği
  - Training sırasında random etiket düşürme

## 7. Ana Model Bileşenleri

### `DiTBlock` Sınıfı
```python
class DiTBlock(nn.Module):
```
- **Amacı**: DiT'in temel yapı taşı
- **Bileşenler**:
  - `norm1`, `norm2`: Layer normalization
  - `attn`: Attention mekanizması
  - `mlp`: Feed-forward ağ
  - `adaLN_modulation`: Adaptive Layer Norm modülasyonu
  - `attn_rate`, `mlp_rate`: Dinamik oran kontrolleri
  - `token_selection`: Token seçim mekanizması

**AdaLN-Zero Kondisyonlama**:
- Timestep ve class bilgilerini kullanarak normalizasyon parametrelerini ayarlar
- `shift` ve `scale` parametreleri ile kondisyonlama yapar

### `FinalLayer` Sınıfı
```python
class FinalLayer(nn.Module):
```
- **Amacı**: Son çıkış katmanı
- **Görevi**: Hidden state'i patch formatına dönüştürür

## 8. Ana DiT Modeli

### `DiT` Sınıfı
```python
class DiT(nn.Module):
```
- **Amacı**: Ana diffusion transformer modeli
- **Bileşenler**:
  - `x_embedder`: Görsel patch embedding
  - `t_embedder`: Timestep embedding
  - `y_embedder`: Label embedding
  - `pos_embed`: Pozisyonel embedding (sabit)
  - `blocks`: DiTBlock'ların listesi
  - `final_layer`: Son çıkış katmanı

**Forward Pass**:
1. Görsel input'u patch'lere böler ve embedding'e çevirir
2. Timestep ve label embedding'lerini hesaplar
3. Tüm DiTBlock'lardan geçirir
4. Final layer ile çıkış üretir
5. Patch'leri geri görsel formatına çevirir

### `forward_with_cfg` Metodu
- **Amacı**: Classifier-free guidance ile çıkarım
- **Yöntem**: Conditional ve unconditional prediction'ları birleştirir

## 9. Pozisyonel Embedding Fonksiyonları

### `get_2d_sincos_pos_embed` ve İlgili Fonksiyonlar
- **Amacı**: 2D pozisyonel embedding'ler oluşturur
- **Yöntem**: Sinüs-cosinüs tabanlı encoding
- **Kullanımı**: Spatial pozisyon bilgisini modele verir

## 10. Model Konfigürasyonları

### Önceden Tanımlı Modeller
```python
DiT_models = {
    'DiT-XL/2': DiT_XL_2,  # En büyük model, 2x2 patch
    'DiT-L/2':  DiT_L_2,   # Büyük model
    'DiT-B/2':  DiT_B_2,   # Orta model
    'DiT-S/2':  DiT_S_2,   # Küçük model
}
```

**Model Boyutları**:
- **XL**: 28 katman, 1152 hidden size, 16 head
- **L**: 24 katman, 1024 hidden size, 16 head
- **B**: 12 katman, 768 hidden size, 12 head
- **S**: 12 katman, 384 hidden size, 6 head

**Patch Boyutları**:
- `/2`: 2x2 patch (yüksek çözünürlük)
- `/4`: 4x4 patch (orta çözünürlük)
- `/8`: 8x8 patch (düşük çözünürlük)

## 11. Dinamik Özellikler

Bu kod'un en önemli özelliği **dinamik adaptasyon** kabiliyeti:

1. **Dinamik Kanal Sayısı**: `attn_rate` ve `mlp_rate` ile kanal sayısı çalışma zamanında ayarlanır
2. **Token Seçimi**: `token_selection` ile önemli token'lar seçilir
3. **Adaptive Genişlik**: `width_mult` parametresi ile model genişliği ayarlanır
4. **Conditional Execution**: `complete_model` parametresi ile tam model veya dinamik model seçimi

## 12. Kullanım Alanları

- **Görsel Üretimi**: Text-to-image generation
- **Görsel Düzenleme**: Image inpainting, super-resolution
- **Stil Transfer**: Style-aware image generation
- **Conditional Generation**: Class-conditional image synthesis

Bu kod, modern diffusion modellerinin transformer mimarisi ile kombinasyonunu gösteriyor ve dinamik adaptasyon ile efficiency'yi artırmaya odaklanıyor.