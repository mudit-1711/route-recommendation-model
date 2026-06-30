import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridRecommender(nn.Module):
    def __init__(self, num_transport_types, num_modes, num_sources, num_dests, num_continuous, emb_dim=16):
        super(HybridRecommender, self).__init__()
        
        # Embeddings for categorical features
        self.trans_emb = nn.Embedding(num_transport_types, emb_dim)
        self.mode_emb = nn.Embedding(num_modes, emb_dim)
        self.src_emb = nn.Embedding(num_sources, emb_dim)
        self.dst_emb = nn.Embedding(num_dests, emb_dim)
        
        # Calculate total dimension after concatenation
        self.total_dim = (emb_dim * 4) + num_continuous
        
        # Shared dense layers
        self.fc1 = nn.Linear(self.total_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        
        # Multi-objective Output Heads
        # 1. Price
        self.out_price = nn.Linear(64, 1)
        # 2. Duration
        self.out_duration = nn.Linear(64, 1)
        # 3. Comfort (0 to 1 score)
        self.out_comfort = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, trans_idx, mode_idx, src_idx, dst_idx, cont_features):
        # Embeddings
        trans_v = self.trans_emb(trans_idx)
        mode_v = self.mode_emb(mode_idx)
        src_v = self.src_emb(src_idx)
        dst_v = self.dst_emb(dst_idx)
        
        # Concat all features
        x = torch.cat([trans_v, mode_v, src_v, dst_v, cont_features], dim=1)
        
        # Shared layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        
        # Output heads
        price = self.out_price(x)
        duration = self.out_duration(x)
        comfort = self.out_comfort(x)
        
        return price, duration, comfort
