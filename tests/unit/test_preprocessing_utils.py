from interpretable_nlp.nlp_utils import tokenize


def test_tokenize_1():

    x = "Hello! Its me, Mario"

    out = ["[CLS]", "hello", "!", "its", "me", ",", "mario", "[SEP]"]

    assert tokenize(x).tokens == out
