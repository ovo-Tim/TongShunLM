from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn as nn
import torch
from torch.nn import functional as F

class TongShunTrain(L.LightningModule):
    def __init__(self, lr, model: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
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
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Pass rate
        y_hat = F.sigmoid(y_hat)
        self.log("90_pass_rate", ((torch.abs(y_hat - y) < 0.1).sum().item() / len(y)))
        self.log("95_pass_rate", ((torch.abs(y_hat - y) < 0.05).sum().item() / len(y)))

        y_pred = (y_hat > 0.5).float()
        acc = (y_pred == y).float().mean()
        self.log("val_accuracy", acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    from dataloader.data_model import TongShunDataModule, DataConfig
    # from configs.test1_20M import args as model_args

    model_name = "configs.test7_124k"
    model_def = "tongshun_model_rwkv"
    validation_frequency = 100
    checkpoint_frequency = 500
    data_conf = DataConfig(
        train_datas=["./train_datas"], val_datas=["./val_datas"], batch_size=128
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # 保存路径
        filename=f"model-{model_name}" + "-{epoch:02d}-{step}-{val_loss:.4f}",
        every_n_train_steps=checkpoint_frequency,  # 每隔多少步保存一次
    )
    logger = TensorBoardLogger(".", version=model_name)
    model_args = getattr(__import__(model_name, fromlist=["args"]), "args")
    model = getattr(__import__(model_def, fromlist=["tongshun"]), "tongshun")(model_args)
    data_module = TongShunDataModule(data_conf)
    model = TongShunTrain(lr=1e-5, model=model)
    trainer = L.Trainer(accelerator="gpu", val_check_interval=validation_frequency, callbacks=[checkpoint_callback], limit_val_batches=1, logger=logger)
    tuner = Tuner(trainer)
    trainer.validate(model, data_module)
    tuner.lr_find(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
