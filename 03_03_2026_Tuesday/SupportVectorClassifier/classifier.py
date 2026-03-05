import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
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
        """
        This function is used to load the data
        :return: None
        """
        try:
          self.df = pd.read_csv("../Dataset/user-data.csv")
        except Exception as e:
            print(f"Error while loading data: {e}")

    def display_data(self):
        """
        This function is used to display top 5, last 5 and random 5 data
        :return: None
        """
        try:
            print(f"Head: \n\n{self.df.head()}", end=separate)
            print(f"Tail: \n\n{self.df.tail()}", end=separate)
            print(f"sample: \n\n{self.df.sample(5)}", end=separate)

        except Exception as e:
            print(f"Error while displaying data: {e}")

    def data_preprocess(self):
        """
        This function is used to understand the data and stats of it
        :return: None
        """
        try:
            print("Columns : ",self.df.columns.tolist(), end=separate)
            print("Data-types: \n", self.df.dtypes, end=separate)
            print(self.df.info(), end=separate)
            print("Stats : \n", self.df.describe(), end=separate)
        except Exception as e:
            print(f"Error while preprocessing: {e}")

    def check_null_duplicates(self):
        """
        This function is used to check null and duplicated values and dropped id column
        :return: None
        """
        try:
            print("Null-Values: \n\n", self.df.isnull().sum(), end=separate)
            print("Duplicates: ", self.df.duplicated().sum(), end=separate)
            self.df = self.df.drop('user_id', axis=1)  # dropping the id columns
        except Exception as e:
            print(f"Error while preprocessing: {e}")

    def EDA(self):
        """
        This function is used to plot Heatmap and pair-plot
        :return: None
        """
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
        """
        This function is used find the outliers
        :return: None
        """
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
        """
        This function is used to split the data into train and test
        :return: None
        """
        try:
            self.x = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=1)
        except Exception as e:
            print(f"Error while splitting: {e}")

    def encoding_and_scaling(self):
        """
        This function is used to encode and scale the data
        :return: None
        """
        try:
            cat_cols = ['gender']                      #categorical-cols
            num_cols = ['age', 'estimated_salary']     #numerical-cols

            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            train_cat = ohe.fit_transform(self.x_train[cat_cols])
            test_cat = ohe.transform(self.x_test[cat_cols])

            scaler = StandardScaler()
            train_num = scaler.fit_transform(self.x_train[num_cols])
            test_num = scaler.transform(self.x_test[num_cols])

            self.x_train = np.hstack([train_cat, train_num])        # combining using horizontal-stack
            self.x_test = np.hstack([test_cat, test_num])
        except Exception as e:
            print(f"Error while encoding: {e}")

    def training(self):
        """
        This function is used to train the Model
        :return: None
        """
        try:
            self.clf = SVC(random_state=1)

            params = {
             'C': [0.1, 1, 10, 25, 50, 100] ,
             'kernel': ['linear', 'poly', 'rbf'],
             'gamma' : ['scale', 'auto'],
            }
            grid = GridSearchCV(estimator=self.clf, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1,verbose=2)
            grid.fit(self.x_train, self.y_train)
            self.y_predicted = grid.predict(self.x_test)
            print(grid.best_params_,end = separate)
        except Exception as e:
            print(f"Error while training: {e}")

    def evaluation(self):
        """
        This function is used to evaluate the model
        :return: None
        """
        try:
            print(f" Accuracy-Score: {accuracy_score(self.y_test, self.y_predicted)*100:.2f}%", end=separate)
            print(f"Classification Report: \n\n{classification_report(self.y_test, self.y_predicted)}", end=separate)
            print(f"Confusion Matrix: \n\n{confusion_matrix(self.y_test, self.y_predicted)}", end=separate)
        except Exception as e:
            print(f"Error while Evaluation: {e}")

    def plot_confusion_matrix(self):
        """
        This function is used to plot the confusion matrix
        :return: None
        """
        try:
            # plotting confusion-Matrix
            cm = confusion_matrix(self.y_test, self.y_predicted)
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, cmap='rocket')
            plt.title("Confusion Matrix", weight=20, fontsize=20, color='red')
            plt.show()
        except Exception as e:
            print(f"Error while plotting after evaluation : {e}")

    def plot_boundary(self):
        """
        This function is used to plot a decision-boundary for two features
        :return: None
        """
        try:
            X_plot = self.x_train[:, :2]  # Taking Age and Salary (first two columns)
            y_plot = self.y_train

            plot_clf = SVC(kernel='linear', random_state=1)
            plot_clf.fit(X_plot, y_plot)

            #  Using Scikit-learn's built-in display tool
            disp = DecisionBoundaryDisplay.from_estimator(
                plot_clf,
                X_plot,
                response_method="predict",
                alpha=0.3,
                cmap='coolwarm'
            )
            #  Overlay the actual data points
            plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, edgecolor="k", cmap='coolwarm')
            plt.title("SVM Decision Boundary (Age vs Salary)")
            plt.xlabel("Scaled Age")
            plt.ylabel("Scaled Salary")
            plt.show()
        except Exception as e:
            print(f"Error while plotting boundary: {e}")

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
    c1.plot_boundary()


if __name__ == '__main__':
    main()