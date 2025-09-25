import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# ------------------------------------------------------------------
# Projector / Predictor networks
# ------------------------------------------------------------------
def projector(in_dim=128, out_dim=64):
    """
    Simple 2-layer MLP used for both the projection head and the predictor.
    Adjust in_dim to match Encoder output.
    """
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, out_dim)
    )

# ------------------------------------------------------------------
# Audio Encoder
# ------------------------------------------------------------------
class Encoder(nn.Module):
    """
    A small CNN encoder for Mel-spectrogram inputs (1 × freq × time).
    Output is a 128-dimensional embedding.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))   # → (B, 32, 1, 1)
        )
        self.fc = nn.Linear(32, 128)       # final embedding size = 128

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)          # flatten to (B, 32)
        return self.fc(x)                  # (B, 128)

# ------------------------------------------------------------------
# BYOL Lightning Module
# ------------------------------------------------------------------
class BYOLModule(pl.LightningModule):
    """
    Self-supervised BYOL module.
    Online encoder is trained; target encoder updated with EMA.
    """
    def __init__(self, lr: float = 1e-3, ema_decay: float = 0.99):
        super().__init__()
        self.save_hyperparameters()

        self.online_enc = Encoder()
        self.target_enc = Encoder()
        self.proj      = projector(in_dim=128, out_dim=64)
        self.pred      = projector(in_dim=64,  out_dim=64)
        self.lr        = lr
        self.ema_decay = ema_decay

        # target encoder not directly optimized
        for p in self.target_enc.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.online_enc(x)

    def _update_target_network(self):
        """Exponential Moving Average update for target encoder."""
        with torch.no_grad():
            for p_t, p_o in zip(self.target_enc.parameters(),
                                self.online_enc.parameters()):
                p_t.data = self.ema_decay * p_t.data + (1 - self.ema_decay) * p_o.data

    def training_step(self, batch, _):
        x1, x2 = batch

        # Online network
        z1, z2 = self.online_enc(x1), self.online_enc(x2)
        p1, p2 = self.pred(self.proj(z1)), self.pred(self.proj(z2))

        # Target network (no grad)
        with torch.no_grad():
            t1, t2 = self.proj(self.target_enc(x1)), self.proj(self.target_enc(x2))

        # BYOL loss
        loss = 2 - (
            F.cosine_similarity(p1, t2).mean() +
            F.cosine_similarity(p2, t1).mean()
        )
        self.log("train_loss", loss, prog_bar=True)

        # EMA update
        self._update_target_network()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
