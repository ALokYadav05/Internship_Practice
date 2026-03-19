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

    def load_data(self):
        try:
            self.sentences = [
                "i hate this film",
                "movie was bad",
                "i like this film",
                "this movie is amazing",
                "film was nice",
                "film was boring",
                "good acting",
                "movie was good",
                "this movie is terrible",
                "bad acting"
            ]
            self.labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])

        except Exception as e:
            print(f"Error while loading data: {e}")


def main():
    rnn = ModelRNN()
    rnn.load_data()

if __name__ == "__main__":
    main()