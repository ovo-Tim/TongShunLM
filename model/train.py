from tongshun_model import tongshun as _TongShun
import lightning as L
from lightning.pytorch.tuner.tuning import Tuner
import torch.nn as nn
import torch

class TongShun(L.LightningModule):
    def __init__(self, lr, model_args, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = _TongShun(model_args)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(*x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat)
        loss = self.loss_fn(y_hat, y)
        # print(loss)
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
    from dataloader.data_model import TongShunDataModule, DataConfig
    from configs.test1_20M import args as model_args

    data_conf = DataConfig(
        train_datas=["./train_datas"],
        val_datas=["./val_datas"],
    )
    data_module = TongShunDataModule(data_conf)

    model = TongShun(lr=1e-5, model_args=model_args)
    trainer = L.Trainer(accelerator="gpu")
    # tuner = Tuner(trainer)
    # tuner.lr_find(model)
    trainer.fit(model, datamodule=data_module)
