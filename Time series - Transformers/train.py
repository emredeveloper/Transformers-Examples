import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import argparse

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.encoder(src)
        return self.output_linear(output[:, -1, :])

def train():
    # Argümanlar
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='daily-total-female-births.csv')
    parser.add_argument('--seq_length', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_path', type=str, default='model.pth')
    args = parser.parse_args()

    # Cihaz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Veriyi yükle ve işle
    df = pd.read_csv(args.data)
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    # Normalizasyon
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)
    
    # Sequence oluştur
    X, y = [], []
    for i in range(len(data) - args.seq_length):
        X.append(data[i:i + args.seq_length])
        y.append(data[i + args.seq_length, 0])  # İlk sütunu tahmin et
    
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).unsqueeze(-1)
    
    # DataLoader
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.9 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    
    # Model
    model = TimeSeriesTransformer(
        input_dim=X.shape[2],
        d_model=128,
        n_heads=4,
        num_layers=2
    ).to(device)
    
    # Eğitim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
        
        val_loss /= len(val_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Val Loss: {val_loss:.6f}')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'seq_length': args.seq_length,
                'input_dim': X.shape[2]
            }, args.model_path)
    
    print(f'Model kaydedildi: {args.model_path}')

if __name__ == '__main__':
    train()
