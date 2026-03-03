import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

separate = f"\n\n{'--'*50}\n"

class Classifier:
    def __init__(self):
        self.df = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.clf = None
        self.y_predicted = None

    def load_data(self):
        try:
          self.df = pd.read_csv("../Dataset/user-data.csv")
        except Exception as e:
            print(f"Error while loading data: {e}")

    def display_data(self):
        try:
            print(f"Head: \n\n{self.df.head()}", end=separate)
            print(f"Tail: \n\n{self.df.tail()}", end=separate)
            print(f"sample: \n\n{self.df.sample(5)}", end=separate)

        except Exception as e:
            print(f"Error while displaying data: {e}")

    def data_preprocess(self):
        try:
            print("Columns : ",self.df.columns.tolist(), end=separate)
            print("Data-types: \n", self.df.dtypes, end=separate)
            print(self.df.info(), end=separate)
            print("Stats : \n", self.df.describe(), end=separate)
            # self.df.drop("user_id", axis=1, inplace=True)
        except Exception as e:
            print(f"Error while preprocessing: {e}")

    def check_null_duplicates(self):
        try:
            print("Null-Values: \n\n", self.df.isnull().sum(), end=separate)
            print("Duplicates: ", self.df.duplicated().sum(), end=separate)
        except Exception as e:
            print(f"Error while preprocessing: {e}")

    def EDA(self):
        try:
            #heat-Map
            plt.figure(figsize=(10,10))
            sns.heatmap(self.df.corr(numeric_only=True),annot=True, cmap='coolwarm')
            plt.title("Correlation-Matrix", weight=20, fontsize=20, color='red')
            plt.show()
            #pair-plot
            sns.pairplot(self.df,hue='gender', palette='rocket', corner=True)
            plt.show()
        except Exception as e:
            print(f"Error while plotting: {e}")

    def outlier_detection(self):
        try:
            self.numeric_cols = self.df.select_dtypes(include='number').drop(columns=['purchased']).columns.tolist()
            for i, col in enumerate(self.numeric_cols):
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3-q1
                lower_bound = q1-1.5*iqr
                upper_bound = q3+1.5*iqr
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                print(f"Total-outliers for {col} : {outliers.sum()}")
            print(end=separate)
        except Exception as e:
          print(f"Error while checking-outliers: {e}")

    def split(self):
        try:
            self.x = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=1)
        except Exception as e:
            print(f"Error while splitting: {e}")

    def encoding_and_scaling(self):
        try:
            ohe = OrdinalEncoder()
            self.x_train["gender"] = ohe.fit_transform(self.x_train[["gender"]])
            self.x_test["gender"] = ohe.transform(self.x_test[["gender"]])

            scaler = StandardScaler()
            self.x_train = scaler.fit_transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)
        except Exception as e:
            print(f"Error while encoding: {e}")

    def training(self):
        try:
            self.clf = SVC(kernel='linear', random_state=1)
            self.clf.fit(self.x_train, self.y_train)
            self.y_predicted = self.clf.predict(self.x_test)
        except Exception as e:
            print(f"Error while training: {e}")

    def evaluation(self):
        try:
            print(f" Accuracy-Score: {accuracy_score(self.y_test, self.y_predicted)*100:.2f}%", end=separate)
            print(f"Classification Report: \n\n{classification_report(self.y_test, self.y_predicted)}", end=separate)
            print(f"Confusion Matrix: \n\n{confusion_matrix(self.y_test, self.y_predicted)}", end=separate)
        except Exception as e:
            print(f"Error while Evaluation: {e}")

    def plot_confusion_matrix(self):
        try:
            # plotting confusion-Matrix
            cm = confusion_matrix(self.y_test, self.y_predicted)
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, cmap='rocket')
            plt.title("Confusion Matrix", weight=20, fontsize=20, color='red')
            plt.show()
        except Exception as e:
            print(f"Error while plotting after evaluation : {e}")

def main():
    c1 = Classifier()
    c1.load_data()
    c1.display_data()
    c1.data_preprocess()
    c1.check_null_duplicates()
    c1.EDA()
    c1.outlier_detection()
    c1.split()
    c1.encoding_and_scaling()
    c1.training()
    c1.evaluation()
    c1.plot_confusion_matrix()


if __name__ == '__main__':
    main()