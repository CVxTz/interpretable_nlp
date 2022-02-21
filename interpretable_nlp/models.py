import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DistilBertModel, get_cosine_schedule_with_warmup


class DistilBert(pl.LightningModule):
    def __init__(
        self,
        n_classes=1,
        lr=1e-4,
        dropout=0.2,
        keep_layers=("transformer.layer.5",),
        loss_fn="bce",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        self.loss_fn = loss_fn

        self.n_classes = n_classes

        self.distil_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for name, param in self.distil_bert.named_parameters():
            if not any(name.startswith(a) for a in keep_layers):
                param.requires_grad = False

        self.do = nn.Dropout(p=dropout)

        self.out_linear = nn.Linear(self.distil_bert.config.dim, n_classes)

    def forward(self, x):
        x = self.distil_bert(x).last_hidden_state  # [batch, seq_len, config.dim]

        x = self.do(x[:, 0, :])

        out = self.out_linear(x)

        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="test")

    def _step(self, batch, batch_idx, name="train"):
        x, y = batch

        y_hat = self(x)

        y_hat = y_hat.view(-1)
        y = y.view(-1)

        if self.loss_fn == "bce":
            loss = F.binary_cross_entropy(
                y_hat, torch.clip(y, 0.01, 0.99), reduction="mean"
            )
        else:
            loss = F.l1_loss(y_hat, torch.clip(y, 0.01, 0.99), reduction="mean")

        acc = ((y > 0.5) == (y_hat > 0.5)).type(torch.float).mean()

        self.log(f"{name}_loss", loss)
        self.log(f"{name}_acc", acc)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_schedulers = {
            "scheduler": get_cosine_schedule_with_warmup(
                optimizer=opt, num_warmup_steps=1000, num_training_steps=7700
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [lr_schedulers]
