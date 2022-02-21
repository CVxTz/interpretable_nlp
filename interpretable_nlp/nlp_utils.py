import re

from nltk.tokenize import sent_tokenize
from transformers import DistilBertTokenizer

TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
PAD_IDX = TOKENIZER.convert_tokens_to_ids("[PAD]")
MAX_LEN = 512
