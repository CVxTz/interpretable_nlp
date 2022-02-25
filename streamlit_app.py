from pathlib import Path

import streamlit as st
import torch

from interpretable_nlp.models import Transformer
from interpretable_nlp.nlp_utils import VOCAB_SIZE
from interpretable_nlp.predict_utils import predict_sentiment, attribution_fun, attribution_to_html

MODEL_PATH = Path(__file__).parents[0] / "models"

SENTIMENT_MODEL = MODEL_PATH / "sentiment.ckpt"


@st.cache
def get_model():
    sentiment_model_ = Transformer(lr=1e-4, n_outputs=3, vocab_size=VOCAB_SIZE)

    sentiment_model_.load_state_dict(torch.load(SENTIMENT_MODEL)["state_dict"])

    sentiment_model_.eval()

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sentiment_model_.to(device_)

    return sentiment_model_, device_


markdown = st.sidebar.markdown(
    "<h2>Detect sentiment of a review</h2>"
    "<h4>Sentiment classes:</h4>"
    "<ul>"
    "<li><span style='background-color:green'>Positive</span></li>"
    "<li><span style='background-color:white'>Neural</span></li>"
    "<li><span style='background-color:red'>Negative</span></li>"
    "</ul>",
    unsafe_allow_html=True,
)

sentiment_model, device = get_model()

txt = st.text_area(
    "Text to analyze",
    """
     I loved that movie
     """,
)

sentiment_output = predict_sentiment(txt, model=sentiment_model, device=device)
color = {"Negative": "red", "Neutral": "white", "Positive": "green"}[sentiment_output]
emoji = {"Negative": "😡", "Neutral": "😐", "Positive": "😀"}[sentiment_output]

st.write(
    "Sentiment:",
    f"{emoji} <span style='background-color:{color}'>{sentiment_output}</span>",
    unsafe_allow_html=True,
)

tokens, attributions = attribution_fun(text=txt, model=sentiment_model, device=device)

html = attribution_to_html(tokens=tokens, attributions=attributions)

st.write(
    "Interpretation:\n",
    html,
    unsafe_allow_html=True,
)
