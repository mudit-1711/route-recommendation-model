import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from model import HybridRecommender

class TravelDataset(Dataset):
    def __init__(self, df):
        self.trans = torch.tensor(df['Transport_Type_encoded'].values, dtype=torch.long)
        self.mode = torch.tensor(df['Mode_encoded'].values, dtype=torch.long)
        self.src = torch.tensor(df['Source_encoded'].values, dtype=torch.long)
        self.dst = torch.tensor(df['Destination_encoded'].values, dtype=torch.long)
        
        # Continuous features
        self.cont = torch.tensor(df[['Distance', 'Day', 'Month', 'Weekday', 'Is_Weekend']].values, dtype=torch.float32)
        
        # Targets
        self.price = torch.tensor(df['Price'].values, dtype=torch.float32).unsqueeze(1)
        self.duration = torch.tensor(df['Duration'].values, dtype=torch.float32).unsqueeze(1)
        
        # Pseudo-target for comfort
        def calc_comfort(row):
            t = row['Transport_Type']
            m = str(row['Mode']).lower()
            if t == 'Flight': return 0.95
            elif t == 'Cab' and 'premier' in m: return 0.85
            elif t == 'Cab' and 'xl' in m: return 0.80
            elif t == 'Cab': return 0.75
            elif t == 'Train': return 0.60
            else: return 0.40
            
        comfort_scores = df.apply(calc_comfort, axis=1).values
        self.comfort = torch.tensor(comfort_scores, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.price)

    def __getitem__(self, idx):
        return (self.trans[idx], self.mode[idx], self.src[idx], self.dst[idx], self.cont[idx],
                self.price[idx], self.duration[idx], self.comfort[idx])

def train_model(data_dir="d:/far away data set/ml model"):
    artifacts_dir = os.path.join(data_dir, "artifacts")
    print("Loading preprocessed data...")
    df = pd.read_csv(os.path.join(artifacts_dir, 'processed_data.csv'))
    
    with open(os.path.join(artifacts_dir, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
        
    num_transport_types = len(encoders['Transport_Type'].classes_)
    num_modes = len(encoders['Mode'].classes_)
    num_sources = len(encoders['Source'].classes_)
    num_dests = len(encoders['Destination'].classes_)
    num_continuous = 5 # Distance, Day, Month, Weekday, Is_Weekend
    
    # Simple Train/Val split
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    train_dataset = TravelDataset(train_df)
    val_dataset = TravelDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = HybridRecommender(num_transport_types, num_modes, num_sources, num_dests, num_continuous).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss() # For comfort since it's 0-1
    
    epochs = 10
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            trans, mode, src, dst, cont, price, dur, comf = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            p_pred, d_pred, c_pred = model(trans, mode, src, dst, cont)
            
            loss_price = criterion_mse(p_pred, price)
            loss_dur = criterion_mse(d_pred, dur)
            loss_comf = criterion_bce(c_pred, comf)
            
            # Weighted loss
            loss = loss_price + loss_dur + (0.5 * loss_comf)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                trans, mode, src, dst, cont, price, dur, comf = [b.to(device) for b in batch]
                p_pred, d_pred, c_pred = model(trans, mode, src, dst, cont)
                l_p = criterion_mse(p_pred, price)
                l_d = criterion_mse(d_pred, dur)
                l_c = criterion_bce(c_pred, comf)
                val_loss += (l_p + l_d + 0.5 * l_c).item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(artifacts_dir, 'model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    print("Training finished. Best Model saved.")

if __name__ == "__main__":
    train_model()
