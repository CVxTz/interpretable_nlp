from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"
TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
PAD_IDX = TOKENIZER.token_to_id("[PAD]")
VOCAB_SIZE = TOKENIZER.get_vocab_size()
MAX_LEN = 256


def tokenize(text):
    output = TOKENIZER.encode(text)
    return output
