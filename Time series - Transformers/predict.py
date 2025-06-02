import torch
import numpy as np
import pandas as pd
import argparse
from train import TimeSeriesTransformer

def predict():
    # Argümanlar
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.pth',
                      help='Eğitilmiş model dosya yolu')
    parser.add_argument('--data', type=str, default='daily-total-female-births.csv',
                      help='Veri dosya yolu')
    parser.add_argument('--steps', type=int, default=10,
                      help='Tahmin adedi')
    args = parser.parse_args()
    
    # Cihaz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Modeli yükle
    print(f"Model yükleniyor: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = TimeSeriesTransformer(
        input_dim=checkpoint['input_dim'],
        d_model=128,
        n_heads=4,
        num_layers=2
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Scaler'ı yükle
    scaler = checkpoint['scaler']
    seq_length = checkpoint['seq_length']
    
    # Veriyi yükle
    print(f"Veri yükleniyor: {args.data}")
    df = pd.read_csv(args.data)
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        dates = pd.RangeIndex(start=0, stop=len(df))
    
    # Son sequence'i al ve normalize et
    data = scaler.transform(df.values)
    last_sequence = torch.FloatTensor(data[-seq_length:]).unsqueeze(0).to(device)
    
    # Tahmin yap
    print(f"{args.steps} adım tahmin yapılıyor...")
    predictions = []
    with torch.no_grad():
        current_sequence = last_sequence
        for step in range(args.steps):
            # Tahmin yap
            pred = model(current_sequence)
            pred_value = pred.item()
            predictions.append(pred_value)
            
            # Yeni sequence oluştur (sadece ilk sütunu güncelle)
            next_step = torch.zeros_like(current_sequence[:, 0:1])
            next_step[0, 0] = pred_value  # Sadece ilk özelliği güncelle
            
            # Yeni sequence: mevcut sequence'nin son seq_length-1 adımını al + yeni tahmin
            current_sequence = torch.cat([
                current_sequence[:, 1:],  # İlk adımı çıkar
                next_step.unsqueeze(1)    # Yeni tahmini ekle
            ], dim=1)
    
    # Tahminleri orijinal ölçeğe çevir
    dummy = np.zeros((len(predictions), data.shape[1]))
    dummy[:, 0] = predictions
    predictions = scaler.inverse_transform(dummy)[:, 0]
    
    # Sonuçları yazdır
    print("\nTahminler:")
    last_date = dates[-1] if 'dates' in locals() else len(dates) - 1
    for i, pred in enumerate(predictions, 1):
        if 'dates' in locals():
            pred_date = last_date + pd.DateOffset(days=i)
            print(f"{pred_date.strftime('%Y-%m-%d')}: {pred:.2f}")
        else:
            print(f"Adım {i}: {pred:.2f}")

if __name__ == '__main__':
    predict()
