import argparse
import os
import random
from functools import partial
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from interpretable_nlp.models import Transformer
from interpretable_nlp.nlp_utils import MAX_LEN, PAD_IDX, VOCAB_SIZE, tokenize


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.rating_to_indexes = {i: dataframe.index[dataframe['star_rating'] == i].tolist() for i in range(1, 6)}

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, _):
        label_ = random.randint(1, 5)
        idx = random.choice(self.rating_to_indexes[label_])

        text = df.loc[idx, "review_body"]
        label = df.loc[idx, "star_rating"] - 1

        x = tokenize(text).ids
        y = label

        x = torch.tensor(x, dtype=torch.long)

        return x, y


def generate_batch(data_batch, pad_idx):
    x_input, y_output = [], []
    for (x, y) in data_batch:
        x_input.append(x[:MAX_LEN])
        y_output.append(y)
    x_input = pad_sequence(x_input, padding_value=pad_idx, batch_first=True)
    y_output = torch.tensor(y_output, dtype=torch.long)

    return x_input, y_output


if __name__ == "__main__":
    batch_size = 32
    epochs = 200

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default=f"/media/{os.environ['USER']}/Data/ML/nlp/amazon_reviews/amazon_reviews_us.tsv",
    )
    args = parser.parse_args()

    OUTPUT_PATH = Path(__file__).parents[1] / "outputs"

    CSV_PATH = Path(args.csv_path)

    base_path = Path(__file__).parents[1]

    df = pd.read_csv(CSV_PATH, sep="\t", on_bad_lines="skip")

    print(df.shape)
    print(df[["star_rating", "review_body"]])

    train_val, test = train_test_split(df, random_state=1337, test_size=0.2)
    train, val = train_test_split(train_val, random_state=1337, test_size=0.2)

    train_data = Dataset(dataframe=train)
    val_data = Dataset(dataframe=val)
    test_data = Dataset(dataframe=test)

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))
    print("len(test_data)", len(test_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=False,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=False,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )

    model = Transformer(lr=1e-4, n_outputs=5, vocab_size=VOCAB_SIZE)

    # model.load_state_dict(torch.load(model_path)["state_dict"])

    logger = TensorBoardLogger(
        save_dir=str(base_path / "logs"),
        name="sentiment",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=base_path / "models",
        filename="sentiment",
        save_weights_only=True,
    )

    early_stopping = EarlyStopping(monitor="valid_loss", mode="min", patience=10)

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=1,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, dataloaders=test_loader)

    # DATALOADER: 0
    # TEST
    # RESULTS
    # {'test_loss': xxxx}
