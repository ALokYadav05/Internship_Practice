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
        """
        Sets up the attention mechanism and the neural network layers.
        """
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

class SentenceBuilder(tf.keras.Model):
    """
    This class is used to a tiny version of the GPT model used for generating text.
    """
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, max_len=100):
        """
        Initializes the word embeddings and stacks the transformer blocks.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_len, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, d_model * 4) for _ in range(num_layers)]
        self.dropout = layers.Dropout(0.1)
        self.final_layer = layers.Dense(vocab_size)

    def call(self, x, training=False):
        """
        Runs the input through embeddings and all transformer blocks to get a prediction.
        """
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)

        x = self.embedding(x) + self.pos_emb(positions)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            # FIX: Pass training as a keyword argument
            x = block(x, training=training)

        return self.final_layer(x)

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

    # Simple sliding window for data
    xs, ys = [], []
    seq_len = 4
    for i in range(len(all_encoded) - seq_len):
        xs.append(all_encoded[i: i + seq_len])
        ys.append(all_encoded[i + 1: i + seq_len + 1])

    X, Y = np.array(xs), np.array(ys)

    model = SentenceBuilder(vocab_size=tokenizer.vocab_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    model.fit(X, Y, epochs=100, verbose=1)
    print("--- Training Complete ---\n")

    def generate(prompt, length=4):
        """
        Predicts the next few words based on the prompt you provide.
        """
        tokens = tokenizer.encode(prompt)
        for _ in range(length):
            input_tokens = np.array([tokens])
            preds = model(input_tokens, training=False)
            next_id = tf.argmax(preds[0, -1, :]).numpy()
            tokens.append(next_id)
        return f"Input: {prompt}", tokenizer.decode(tokens)

    print(f"Result: {generate('deep learning')}")

if __name__ == "__main__":
    main()