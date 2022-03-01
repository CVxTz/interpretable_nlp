# interpretable_nlp

# Data
https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt

Data License: https://s3.amazonaws.com/amazon-reviews-pds/readme.html

The file : amazon_reviews_multilingual_US_v1_00.tsv.gz

# Setup

## Pytorch

Install pytorch (torch==1.10.1) depending on your own setup: https://pytorch.org/

/!\ The training part only works with a GPU for now. 

## lib

```commandline
pip install -e .
```

## Tests

```commandline
pytest tests/
```
