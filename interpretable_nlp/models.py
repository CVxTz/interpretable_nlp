import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup


class PositionalEncoding(nn.Module):
    #  https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0:, :, 0::2] = torch.sin(position * div_term)
        pe[0:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x + self.pe[:, : x.size(1)]

        return self.dropout(x)


class TokenEmbedding(nn.Module):
    #  https://pytorch.org/tutorials/beginner/translation_transformer.html
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Transformer(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        channels=256,
        dropout=0.3,
        n_outputs=3,
        lr=1e-4,
    ):
        super().__init__()

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.n_outputs = n_outputs

        self.embeddings = TokenEmbedding(vocab_size=self.vocab_size, emb_size=channels)

        self.pos_encoder = PositionalEncoding(d_model=channels, dropout=dropout)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            batch_first=True, d_model=channels, nhead=4, dim_feedforward=4 * channels
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=6
        )

        self.linear = torch.nn.Linear(channels, self.n_outputs)

        self.do = nn.Dropout(p=self.dropout)

    def encode(self, x):

        x = self.embeddings(x)
        x = self.pos_encoder(x)

        x = self.encoder(x)

        x = x[:, 0, :]

        return x

    def forward(self, x):
        x = self.do(self.encode(x))
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="test")

    def _step(self, batch, batch_idx, name="train"):
        x, y = batch

        y_hat = self(x)

        y_hat = y_hat
        y = y.view(-1)

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)

        acc = (y == predicted).double().mean()

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
