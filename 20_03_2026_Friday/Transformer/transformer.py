import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Tokenizer:
    """
    This class is used to Handles turning text into numbers and back again.
    """
    def __init__(self, texts):
        all_text = " ".join(texts)
        self.words = sorted(set(all_text.split()))
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.words)

    def encode(self, text):
        """
        Swaps words for their corresponding ID numbers.
        """
        return [self.word2idx[w] for w in text.split() if w in self.word2idx]

    def decode(self, tokens):
        """
        Swaps ID numbers back into readable words.
        """
        return " ".join([self.idx2word[int(t)] for t in tokens])

def main():
    """
    The main script: prepares data, trains the model, and tests it out.
    """
    training_samples = [
        "the neural network learns patterns",
        "deep learning models need data",
        "artificial intelligence is changing technology",
        "transformers process sequences in parallel",
        "large language models generate text",
        "machine learning improves with experience",
        "python is great for data science"
    ]

    tokenizer = Tokenizer(training_samples)
    all_encoded = []
    for s in training_samples:
        all_encoded.extend(tokenizer.encode(s))

if __name__ == "__main__":
    main()