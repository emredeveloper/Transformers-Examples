import torch
import numpy as np
import pandas as pd
import argparse
from train import TimeSeriesTransformer

def predict():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.pth',
                      help='Path to the trained model file')
    parser.add_argument('--data', type=str, default='daily-total-female-births.csv',
                      help='Path to the data file')
    parser.add_argument('--steps', type=int, default=10,
                      help='Number of forecast steps')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = TimeSeriesTransformer(
        input_dim=checkpoint['input_dim'],
        d_model=128,
        n_heads=4,
        num_layers=2
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Restore scaler configuration
    scaler = checkpoint['scaler']
    seq_length = checkpoint['seq_length']
    
    # Load the data
    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data)
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        dates = pd.RangeIndex(start=0, stop=len(df))
    
    # Extract the latest sequence and normalise
    data = scaler.transform(df.values)
    last_sequence = torch.FloatTensor(data[-seq_length:]).unsqueeze(0).to(device)
    
    # Perform the forecast
    print(f"Generating {args.steps} step predictions...")
    predictions = []
    with torch.no_grad():
        current_sequence = last_sequence
        for step in range(args.steps):
            # Predict the next value
            pred = model(current_sequence)
            pred_value = pred.item()
            predictions.append(pred_value)
            
            # Build the next sequence (update only the first feature)
            next_step = torch.zeros_like(current_sequence[:, 0:1])
            next_step[0, 0] = pred_value  # Update only the first feature
            
            # New sequence: drop the oldest step and append the prediction
            current_sequence = torch.cat([
                current_sequence[:, 1:],  # Remove the first timestep
                next_step.unsqueeze(1)    # Append the new forecast
            ], dim=1)

    # Rescale predictions back to the original domain
    dummy = np.zeros((len(predictions), data.shape[1]))
    dummy[:, 0] = predictions
    predictions = scaler.inverse_transform(dummy)[:, 0]
    
    # Display predictions
    print("\nPredictions:")
    last_date = dates[-1] if 'dates' in locals() else len(dates) - 1
    for i, pred in enumerate(predictions, 1):
        if 'dates' in locals():
            pred_date = last_date + pd.DateOffset(days=i)
            print(f"{pred_date.strftime('%Y-%m-%d')}: {pred:.2f}")
        else:
            print(f"Step {i}: {pred:.2f}")

if __name__ == '__main__':
    predict()
