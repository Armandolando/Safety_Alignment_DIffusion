import torch
import torch.nn as nn

class ResidualTranslationMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LayerNorm(512), # UPGRADE 1: LayerNorm preserves weight magnitude scales better
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
        
        # UPGRADE 2: Zero-Initialization
        # This guarantees the model starts by predicting exactly 0 change.
        # It only learns to shift the weights where absolutely necessary for safety.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        return self.net(x)

class TranslationMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # A small but deep MLP to learn non-linear safety mappings
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
    
    def forward(self, x):
        return self.net(x)

class WeightAutoencoder(nn.Module):
    def __init__(self, dim, latent_dim=256):
        super().__init__()
        # Encoder: Compresses the weight vector
        self.encoder = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        # Decoder: Reconstructs the 'Safe' weight vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)