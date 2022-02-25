from typing import List
import torch

from captum.attr import LayerIntegratedGradients

from colour import Color

from interpretable_nlp.models import Transformer
from interpretable_nlp.nlp_utils import MAX_LEN, tokenize, PAD_IDX, CLS_IDX, SEP_IDX


COLOR_RANGE = list(Color("red").range_to(Color("white"), 10)) + list(
    Color("white").range_to(Color("green"), 10)
)


def predict_sentiment(text: str, model: Transformer, device: torch.device):
    """
    Predict Sentiment score (between 0 and 1)
    :param text:
    :param model: Model in eval mode
    :param device: cuda or cpu torch.device
    :return:
    """

    x = tokenize(text).ids[:MAX_LEN]
    x = torch.tensor([x], dtype=torch.long)

    x = x.to(device)

    with torch.no_grad():
        y_hat = model(x).cpu()
        _, predicted = torch.max(y_hat, 1)
        predicted_class = predicted.item()

    return {0: "Negative", 1: "Neutral", 2: "Positive"}[predicted_class]


def attribution_to_html(tokens: List, attributions: List):
    html = ""

    for token, attribution in zip(tokens, attributions):
        if attribution >= 0:
            idx = int(attribution ** 1 * 10) + 10
        else:
            idx = int((-(-attribution) ** 1 + 1) * 10)

        idx = min(idx, 19)

        color = COLOR_RANGE[idx]
        html += f""" <span style="background-color: {color.hex}">{token}</span>"""

    return html


def attribution_fun(text: str, model: Transformer, device: torch.device):
    tokenized = tokenize(text)
    tokens_idx = tokenized.ids[:MAX_LEN]
    x = torch.tensor([tokens_idx], dtype=torch.long)
    ref = torch.tensor(
        [[CLS_IDX] + [PAD_IDX] * (len(tokens_idx) - 2) + [SEP_IDX]], dtype=torch.long
    )

    x = x.to(device)
    ref = ref.to(device)

    base_class = 2

    lig = LayerIntegratedGradients(
        model,
        model.embeddings.embedding,
    )

    attributions_ig, delta = lig.attribute(
        x, ref, n_steps=500, return_convergence_delta=True, target=base_class
    )

    attributions_ig = attributions_ig[0, 1:-1, :].sum(dim=-1).cpu()
    attributions_ig = attributions_ig / attributions_ig.abs().max()

    return tokenized.tokens[1:-1], attributions_ig.tolist()
