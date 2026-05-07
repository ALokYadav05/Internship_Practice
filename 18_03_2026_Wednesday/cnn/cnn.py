from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


class MnistCNN:
    """ This class is used to implement the MNIST """
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.history = None

    def load_data(self):
        """
        This method is used to load the data, normalize it and reshape it
        :return: None
        """
        try:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

            # Normalizing values
            self.x_train = self.x_train / 255.0
            self.x_test = self.x_test / 255.0

            self.x_train = self.x_train.reshape(-1, 28, 28, 1)
            self.x_test = self.x_test.reshape(-1, 28, 28, 1)

            print(f"Train shape: {self.x_train.shape}, Test shape: {self.x_test.shape}")
        except Exception as e:
            print(f"Error while loading data: {e}")

    def build_model(self):
        """
        This method is used to build the model
        :return: None
        """
        try:
            self.load_data()
            self.model = Sequential([
                Conv2D(filters=32,kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(10, activation='softmax')
            ])
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print(self.model.summary())

        except Exception as e:
            print(f"Error while building model: {e}")

    def train_model(self):
        """
        This method is used to train the model
        :return: None
        """
        try:
            self.build_model()
            self.history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=5,
                validation_data=(self.x_test, self.y_test)
            )
        except Exception as e:
            print(f"Error while model training: {e}")

    def evaluate_model(self):
        """
        This method is used to evaluate the model
        :return: None
        """
        try:
            self.train_model()
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
        except Exception as e:
            print(f"Error while evaluation model: {e}")

    def plot_performance(self):
        """
        This method is used to plot the accuracy and loss
        :return: None
        """
        try:
            self.evaluate_model()
            plt.figure(figsize=(12, 4))
            # plot for accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            # plot for loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error while plotting: {e}")

def main():
    cnn = MnistCNN()
    cnn.plot_performance()

if __name__ == "__main__":
    main()