import os

from setuptools import setup

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except FileNotFoundError:
    REQUIRED = []

setup(
    name="interpretable_nlp",
    version="0.0.1",
    author="Youness MANSAR",
    author_email="mansaryounessecp@gmail.com",
    description="nlp",
    license="GNU",
    keywords="nlp",
    url="https://github.com/CVxTz/interpretable_nlp",
    packages=["interpretable_nlp"],
    classifiers=[
        "Topic :: Utilities",
    ],
    install_requires=REQUIRED,
    data_files=[("interpretable_nlp", ["interpretable_nlp/tokenizer.json"])],
    include_package_data=True,
)
