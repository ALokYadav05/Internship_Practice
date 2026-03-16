import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

separator = f"\n{'--'*60}\n\n"

class DLClassifier:
    def __init__(self):
        self.df = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.encoder = LabelEncoder()

    def read_data(self):
        try:
            self.df = pd.read_csv("../Dataset/Titanic_dataset.csv")
            print(self.df.head(), end=separator)
        except FileNotFoundError as e:
            print("File not found")
        except Exception as e:
            print(f"Error while reading data: {e}")

    def preprocess(self):
        try:
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
        plt.figure(figsize=(10, 8))
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Feature Correlation Heatmap", weight='semibold', color='red', fontsize=15)
        plt.show()

def main():
    clf = DLClassifier()
    clf.read_data()
    clf.preprocess()
    clf.visualize()

if __name__ == "__main__":
    main()