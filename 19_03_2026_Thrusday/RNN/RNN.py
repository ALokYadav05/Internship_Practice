import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

separate = f"\n\n{'--'*50}\n\n"

class ModelRNN:
    """
    This class is used to build RNN model
    """
    def __init__(self):
        self.data = None
        self.sentences = None
        self.labels = None
        self.tokenizer = Tokenizer()
        self.padded_sequences = None
        self.tokenizer = Tokenizer()
        self.model = None

    def load_data(self):
        """
        This method is used to load data
        :return: None
        """
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
            self.labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])     # 0 = Negative Sentiment, 1 = Positive Sentiment

        except Exception as e:
            print(f"Error while loading data: {e}")

    def preprocessing(self):
        """
        This method is used to preprocess the data
        :return: None
        """
        try:
            self.load_data()
            # Convert text words into unique integer IDs
            self.tokenizer.fit_on_texts(self.sentences)
            sequences = self.tokenizer.texts_to_sequences(self.sentences)
            # Ensure all sequences are the same length by adding zeros at the end ('post' padding)
            self.padded_sequences = pad_sequences(sequences, padding='post')

            print(f"Word-Index: \n{self.tokenizer.word_index}", end=separate)
            print(f"Sequences: \n{sequences}", end=separate)
            print(f"Padded-Sequences: \n{self.padded_sequences}", end=separate)
        except Exception as e:
            print(f"Error during Preprocessing: {e}")

    def model_building(self):
        """
        This method is used to build RNN model
        :return: None
        """
        try:
            self.preprocessing()
            vocab_size = len(self.tokenizer.word_index) + 1
            self.model = Sequential([
                # embedding turns word IDs into dense vectors of fixed size (16)
                Embedding(input_dim=vocab_size, output_dim=16),
                SimpleRNN(16),
                Dense(1, activation='sigmoid')
            ])
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            print(self.model.summary())
        except Exception as e:
            print(f"Error while building Model: {e}")

    def training(self):
        """
        This method is used to train the model
        :return: None
        """
        try:
            self.model_building()
            self.model.fit(
                self.padded_sequences,
                self.labels,
                epochs=20
            )
        except Exception as e:
            print(f"Error while training: {e}")

    def predict(self):
        """
        This method is used to predict the sentiment
        :return: None
        """
        try:
            self.training()
            test = ['Movie was good enough']
            seq = self.tokenizer.texts_to_sequences(test)
            padded = pad_sequences(seq, maxlen=self.padded_sequences.shape[1],
                                   padding='post')
            prediction = self.model.predict(padded)

            print(f"prediction: {prediction}")

            if prediction > 0.5:
                print(f"Sentiment: Positive!")
            else:
                print(f"sentiment: Negative!")
        except Exception as e:
            print(f"Error while predicting: {e}")

def main():
    rnn = ModelRNN()
    rnn.predict()

if __name__ == "__main__":
    main()