import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

separator = f"\n{'--'*60}\n\n"

class DLClassifier:
    def __init__(self):
        self.df = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.preprocessor = None
        self.model = None
        self.input_shape = (9,)     # 4 (numeric feature) + 2 (gender feature) + 3 (Embarked feature)
                                  # This feature are after OHE
        self.history = None

    def read_data(self):
        """
        This method is used to read the data from the csv file
        :return: None
        """
        try:
            self.df = pd.read_csv("../Dataset/Titanic_dataset.csv")
            print(self.df.head(), end=separator)
        except FileNotFoundError as e:
            print("File not found",e)
        except Exception as e:
            print(f"Error while reading data: {e}")

    def preprocess(self):
        """
        This method is used to preprocess the data
        :return: None
        """
        try:
            self.read_data()
            print(f"Columns: {self.df.columns}", end=separator)
            self.df.info()
            print(end=separator)
            print(f"Stats: \n\n{self.df.describe()}", end=separator)
            print(f"Null-values: \n\n{self.df.isnull().sum()}", end=separator)
            print(f"Total-Duplicates: {self.df.duplicated().sum()}", end=separator)

            # handling null-values
            self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
            self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])

            # dropping unnecessary columns
            self.df.drop(columns=['Name','PassengerId','Ticket','Cabin'], inplace=True)

            # Feature Engineering
            self.df["FamilySize"] = self.df["SibSp"] + self.df["Parch"] + 1
            self.df.drop(columns=["SibSp", "Parch"], inplace=True)

            print("Preprocessing Done")
            print(self.df.head(), end=separator)
        except Exception as e:
            print(f"Error while preprocessing: {e}")

    def visualize(self):
        """
        This method is used to visualize the data
        :return: None
        """
        try:
            self.preprocess()
            plt.figure(figsize=(10, 8))
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title("Feature Correlation Heatmap", weight='semibold', color='red', fontsize=15)
            plt.show()
        except Exception as e:
            print(f"Error while visualization: {e}")

    def split_data(self):
        """
        This method is used to split the data into train and test
        :return: None
        """
        try:
            self.visualize()
            self.x = self.df.drop(columns="Survived")
            self.y = self.df["Survived"]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                    test_size=0.2, random_state=1)
        except Exception as e:
            print(f"Error while splitting: {e}")

    def preprocessing_pipeline(self):
        """
        This method is used to define pipeline for numeric & categorical features
        :return: None
        """
        try:
            self.split_data()
            numeric_features = ["Age", "Fare", "Pclass", "FamilySize"]
            category_features = ["Embarked", "Sex"]

            # pipeline for preprocessing numeric features
            numeric_pipeline = Pipeline([
                ("scaler", self.scaler),
            ])
            # pipeline for preprocessing categorical features
            categorical_pipeline = Pipeline([
                ("encoder", self.encoder),
            ])
            # Applying Transformation to individual features
            self.preprocessor = ColumnTransformer([
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, category_features),
            ])
            self.x_train = self.preprocessor.fit_transform(self.x_train)
            self.x_test = self.preprocessor.transform(self.x_test)
        except Exception as e:
            print(f"Error during feature transformation: {e}")

    def build_model(self):
        """
        This method is used to build the model
        :return: None
        """
        try:
            self.preprocessing_pipeline()
            self.model = Sequential([
                Dense(32, activation="relu", input_shape=self.input_shape),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),
            ])
            self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            self.model.summary()
        except Exception as e:
            print(f"Error while building the model: {e}")

    def train_model(self):
        """
        This method is used to trains the model and stores the history for plotting
        :return:None
        """
        try:
            self.build_model()
            print("Training model...")
            self.history = self.model.fit(
                self.x_train, self.y_train,
                validation_data=(self.x_test, self.y_test),
                epochs=50,
                batch_size=32,
                verbose=1
            )
        except Exception as e:
            print(f"Error during training: {e}")

    def evaluate_model(self):
        """
        This method is used to evaluates the model on the test set and prints accuracy
        :return:None
        """
        try:
            self.train_model()
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)  # this return loss and accuracy
            print(separator)
            print(f"Test Loss: {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error while evaluating model: {e}")

    def plot_history(self):
        """
        This method is used to plots training & validation accuracy and loss
        :return:None
        """
        try:
            self.evaluate_model()
            plt.figure(figsize=(12, 5))
            # Accuracy Plot
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.legend()
            # Loss Plot
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error while plotting Accuracy & loss: {e}")

def main():
    clf = DLClassifier()
    clf.plot_history()

if __name__ == "__main__":
    main()