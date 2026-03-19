import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

separate = f"\n\n{'--'*50}\n\n"

class ModelRNN:
    def __init__(self):
        self.data = None
        self.sentences = None
        self.labels = None
        self.tokenizer = Tokenizer()
        self.padded_sequences = None
        self.tokenizer = Tokenizer()
        self.model = None


def main():
    rnn = ModelRNN()

if __name__ == "__main__":
    main()