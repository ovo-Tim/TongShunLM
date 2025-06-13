from .tongshun_model import RWKV as _TongShun
import lightning as L
import torch.nn as nn
import torch

class TongShun(L.LightningModule, _TongShun):
    def __init__(self, lr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Pass rate
        self.log("90_pass_rate", ((torch.abs(y_hat - y) < 0.1).sum().item() / len(y)))
        self.log("80_pass_rate", ((torch.abs(y_hat - y) < 0.2).sum().item() / len(y)))

        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    model = TongShun(lr=1e-5)
    trainer = L.Trainer()
    trainer.fit(model)
