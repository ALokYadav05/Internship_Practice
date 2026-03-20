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


class TransformerBlock(layers.Layer):
    """
    This class is used to a single 'brain cell' layer that helps the model
    focus on different parts of a sentence.
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):
        """
        Processes the data through attention and cleanup layers.
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Proper causal mask for Keras MHA
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = i >= j
        mask = tf.reshape(mask, (1, seq_len, seq_len))

        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

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