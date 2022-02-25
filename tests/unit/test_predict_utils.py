import torch

from interpretable_nlp.models import Transformer
from interpretable_nlp.nlp_utils import VOCAB_SIZE
from interpretable_nlp.predict_utils import predict_sentiment


def test_classifier():
    model = Transformer(vocab_size=VOCAB_SIZE)
    model.eval()

    sample_text = " ".join(["Hello"] * 128)

    output = predict_sentiment(
        model=model, device=torch.device("cpu"), text=sample_text
    )

    assert output in ["Negative", "Neural", "Positive"]
