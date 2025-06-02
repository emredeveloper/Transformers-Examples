# Gelişmiş Görüntü İşleme Uygulaması

Bu proje, büyük görüntüleri daha küçük parçalara bölen, bu parçalara çeşitli filtreler uygulayan ve sonrasında görüntüyü tekrar birleştiren bir Streamlit uygulamasıdır.

## Özellikler

- Büyük görüntüleri parçalara ayırma
- Her parçaya farklı görüntü filtreleri uygulama
- İşlenen parçaları orijinal boyutlarında birleştirme
- Kullanıcı dostu arayüz
- İşlenmiş görüntüyü indirme imkanı

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
   ```
   pip install streamlit numpy torch Pillow
   ```

2. Uygulamayı çalıştırın:
   ```
   streamlit run image_processor_app.py
   ```

## Kullanım

1. Sol taraftaki menüden bir görüntü yükleyin
2. İstediğiniz filtreyi seçin
3. Örtüşme payını ve maksimum parça sayısını ayarlayın
4. "Görüntüyü İşle" butonuna tıklayın
5. İşlenmiş görüntüyü inceleyip indirebilirsiniz

## Kullanılan Filtreler

- Normal: Orijinal görüntü
- Siyah-Beyaz: Gri tonlamalı görüntü
- Blur: Bulanıklaştırma efekti
- Kontur: Kenar belirleme
- Keskinleştir: Görüntüyü keskinleştirme

## Geliştirme

Bu proje, büyük görüntüleri işlemek için parçalama ve birleştirme işlemlerini gösteren bir örnektir. Daha fazla özellik ekleyerek genişletebilirsiniz:

- Daha fazla filtre seçeneği
- Parça boyutlarını özelleştirme
- Toplu işlem yapabilme
- Farklı kaydetme formatları

## Lisans

MIT
