import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
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
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.clf = None
        self.y_predicted = None
        self.preprocessor_pipeline = None
        self.pipeline = None
        self.model = None

    def load_data(self):
        """
        This method is used to load the data
        :return: None
        """
        try:
          self.df = pd.read_csv("../Dataset/user-data.csv")
        except Exception as e:
            print(f"Error while loading data: {e}")

    def display_data(self):
        """
        This method is used to display top 5, last 5 and random 5 data
        :return: None
        """
        try:
            self.load_data()
            print(f"Head: \n\n{self.df.head()}", end=separate)
            print(f"Tail: \n\n{self.df.tail()}", end=separate)
            print(f"sample: \n\n{self.df.sample(5)}", end=separate)

        except Exception as e:
            print(f"Error while displaying data: {e}")

    def data_preprocess(self):
        """
        This method is used to understand the data and stats of it
        :return: None
        """
        try:
            self.display_data()
            print("Columns : ",self.df.columns.tolist(), end=separate)
            print("Data-types: \n", self.df.dtypes, end=separate)
            print(self.df.info(), end=separate)
            print("Stats : \n", self.df.describe(), end=separate)
        except Exception as e:
            print(f"Error while preprocessing: {e}")

    def check_null_duplicates(self):
        """
        This method is used to check null and duplicated values and dropped id column
        :return: None
        """
        try:
            self.data_preprocess()
            print("Null-Values: \n\n", self.df.isnull().sum(), end=separate)
            print("Duplicates: ", self.df.duplicated().sum(), end=separate)
            self.df = self.df.drop('user_id', axis=1)  # dropping the id columns
        except Exception as e:
            print(f"Error while preprocessing: {e}")

    def EDA(self):
        """
        This method is used to plot Heatmap and pair-plot
        :return: None
        """
        try:
            self.check_null_duplicates()
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
        This method is used find the outliers
        :return: None
        """
        try:
            self.EDA()
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

    def split_train_test(self):
        """
        This method is used to split the data into train and test
        :return: None
        """
        try:
            self.outlier_detection()
            self.x = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=1)
        except Exception as e:
            print(f"Error while splitting: {e}")

    def pipeline_creation(self):
        """
        This method creates pipeline for preprocessing and transforming data
        :return: None
        """
        try:
            self.split_train_test()
            numeric_features = self.df.select_dtypes(include='int64').columns.drop("purchased")
            cat_features = self.df.select_dtypes(include='object').columns

            self.preprocessor_pipeline = ColumnTransformer(
                                                transformers=[
                                                            ('num',self.scaler,numeric_features),
                                                            ('cat',self.encoder,cat_features)
                                                        ],
                                               remainder='passthrough')

            print("Preprocessing Pipeline created", end=separate)
        except Exception as e:
            print("Error while creating pipeline", e)

    def model_training(self):
        """
        This method is used to trains the model
        :return: None
        """
        try:
            self.pipeline_creation()
            self.model = KNeighborsClassifier()
            self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor_pipeline),
                                            ('classifier',self.model)])
            param_grid = {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan", "minkowski"]
            }
            print("Starting Grid Search CV... this may take a while...\n")

            grid= GridSearchCV(self.pipeline, param_grid, refit=True, verbose=2, cv=5)
            grid.fit(self.x_train, self.y_train)
            self.pipeline = grid.best_estimator_
            self.model = grid.best_estimator_.named_steps["classifier"]
            print(end=separate)
            print(f"Best Parameters:{grid.best_params_}\n")
            print("Model Training Done", end=separate)
        except Exception as e:
            print("Error while training model training", e)

    def model_performance(self):
        """
        This method is used to checks model performance on test data
        :return: None
        """
        try:
            self.model_training()
            y_prediction = self.pipeline.predict(self.x_test)
            print(f"Confusion Matrix\n")
            print(confusion_matrix(self.y_test, y_prediction))
            print("\nClassification Report\n")
            print(classification_report(self.y_test, y_prediction))
        except Exception as e:
            print("Error checking model performance", e)

    def save_model(self):
        """
        This method saves the model
        :return: None
        """
        self.model_performance()
        try:
            joblib.dump(self.pipeline, filename='../Model/Model.pkl')
            print("Model Saved Successfully!!")
        except Exception as e:
            print("Error while saving model", e)


def main():
    c1 = Classifier()
    c1.save_model()


if __name__ == '__main__':
    main()