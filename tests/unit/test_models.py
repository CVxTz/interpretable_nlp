import torch

from hotel_reviews.models import DistilBert
from hotel_reviews.nlp_utils import TOKENIZER


def test_classifier():
    model = DistilBert()
    model.eval()

    sample_text = " ".join(["Hello"] * 128)

    ids = TOKENIZER.encode(sample_text)

    input_ids = torch.tensor([ids] * 8)
    outputs = model(input_ids)

    assert outputs.size() == (8, 1)
