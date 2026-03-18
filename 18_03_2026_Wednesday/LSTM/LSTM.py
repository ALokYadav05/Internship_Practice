import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
import warnings

warnings.filterwarnings('ignore')

separator = f"\n\n{'--' * 50}\n\n"


class SentimentAnalysis:
    def __init__(self):
        self.data = None
        self.tokenizer = Tokenizer()
        self.x = None
        self.y = None
        self.model = None
        self.max_len = 5

    def load_data(self):
        """
        This method is used to load the data
        :return: None
        """
        try:
            data = {
                'Review': [
                    'I love this product',
                    'This is amazing',
                    'Very bad experience',
                    'I hate this item',
                    'Excellent quality',
                    'Worst purchase ever',
                    'Really happy with this',
                    'Not good at all',
                    'Superb performance',
                    'Terrible service'
                ],  
                'Sentiment': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
            }
            self.data = pd.DataFrame(data)
            print(self.data, end=separator)

        except Exception as e:
            print(f"Error while data loading: {e}")

    def preprocessing(self):
        """
        This method is used to preprocess the data
        :return: None
        """
        try:
            self.load_data()
            texts = self.data['Review']
            self.y = self.data['Sentiment']

            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)
            self.x = pad_sequences(sequences, maxlen=self.max_len)
            print(f" Tokenized data: \n{self.x}", end=separator)
        except Exception as e:
            print(f"Error while pre-processing: {e}")

    def build_model(self):
        """
        This method is used to build the model
        :return: None
        """
        try:
            self.preprocessing()
            vocab_size = len(self.tokenizer.word_index) + 1
            self.model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=8),
                LSTM(units=16),
                Dense(1, activation='sigmoid')
            ])

            print(self.model.summary(), end=separator)

            self.model.compile(
                optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            print(f"Error while building Model: {e}")

    def train_model(self):
        """
        This method is used to train the model
        :return: None
        """
        try:
            self.build_model()
            self.model.fit(self.x, self.y, epochs=20, verbose=2)
            print("Model trained Successfully!", end=separator)
        except Exception as e:
            print(f"Error while training model: {e}")

    def predict(self):
        """
        This method is used to predict on new unseen data
        :return: None
        """
        try:
            self.train_model()
            test_data = ['The product was good enough!']

            text = self.tokenizer.texts_to_sequences(test_data)
            padded = pad_sequences(text, maxlen=self.max_len)
            predicted = self.model.predict(padded)

            print(f"Prediction: {predicted}", end=separator)

            if predicted > 0.5:
                print(" Sentiment: Positive-Sentiments")
            else:
                print("Sentiment: Negative-Sentiments")
        except Exception as e:
            print(f"Error while Predicting: {e}")

def main():
    lstm = SentimentAnalysis()
    lstm.predict()

if __name__ == '__main__':
    main()