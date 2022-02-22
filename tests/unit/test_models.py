import torch

from interpretable_nlp.models import Transformer
from interpretable_nlp.nlp_utils import VOCAB_SIZE, tokenize


def test_classifier():
    model = Transformer(vocab_size=VOCAB_SIZE)
    model.eval()

    sample_text = " ".join(["Hello"] * 128)

    ids = tokenize(sample_text).ids

    input_ids = torch.tensor([ids] * 8, dtype=torch.long)
    outputs = model(input_ids)

    assert outputs.size() == (8, 3)
