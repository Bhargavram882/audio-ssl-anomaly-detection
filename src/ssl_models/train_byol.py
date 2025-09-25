from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.dataset import MelPairDataset
from src.ssl.byol_module import BYOLModule


def main():
    ds = MelPairDataset()
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    model = BYOLModule()
    trainer = pl.Trainer(max_epochs=5, accelerator="cpu")
    trainer.fit(model, dl)
    trainer.save_checkpoint("checkpoints/byol.ckpt")

if __name__ == "__main__":
    main()
