import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

separate = f"\n\n{'--' * 100}\n\n"  # this is used only to separate print statements
pd.set_option('display.width', None)  # used to display  all columns while printing


class MSSQLDatabase:
    def __init__(self,server,database,driver ="ODBC+Driver+17+for+SQL+Server"):
        """
        This constructor is used to Initialize database connection
        """
        self.connection_string = f"mssql+pyodbc://{server}/{database}?driver={driver}"  #connection-string
        try:
            self.df = None
            self.table_name = 'diabetes'
            self.engine = create_engine(self.connection_string)
            print("Engine created successfully!")
        except ConnectionError:
            print("Connection Error")

    def loadTable(self):
        """
        This method is used to load data from MSSQL database
        :return : dataframe,table
        """
        try:
            query = f"SELECT * FROM {self.table_name}"
            self.df = pd.read_sql(query, self.engine)
            return self.df, self.table_name
        except FileNotFoundError:
            print("Table not found")

    def save_prediction(self,predict):
        """
        This function is used to save the prediction back to database
        :return: None
        """
        try:
            self.df['Predicted'] = predict
            self.df.to_sql(self.table_name, self.engine, if_exists='replace', index=False)
            print("Predictions  Added successfully in Database!!")
        except Exception as e:
            print(f"Error while saving prediction: {e}", end=separate)


class Classifier:
    def __init__(self,df):
        self.df = df
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.clf = None
        self.numerical_cols = None
        self.y_predict = None

    def displayData(self):
        """
        This function displays the data
        :return: None
        """
        try:
            print(self.df.head(), end=separate)
        except Exception as e:
            print(f"Error while displaying data: {e}", end=separate)

    def dataUnderstanding(self):
        """
        This function is used to understand the data shape, data-type and statistics
        :return: None
        """
        try:
            print("Columns: \n\n", self.df.columns.tolist(), end=separate)
            print(self.df.info(), end=separate)
            print("(Rows, columns) = ",self.df.shape, end=separate)
            print("Data-types: \n\n",self.df.dtypes, end=separate)
            print("Stats: \n\n",self.df.describe(), end=separate)
        except Exception as e:
            print(f"Error while understanding data: {e}", end=separate)

    def checkNullDuplicates(self):
        """
        This function is used to check null and duplicate values
        :return: None
        """
        try:
            print("Null values: \n\n",self.df.isnull().sum(), end=separate)
            print("Total-Duplicates :",self.df.duplicated().sum(), end=separate)
        except Exception as e:
            print(f"Error while checking null values: {e}", end=separate)

    def EDA(self):
        """
        This function is used to perform Exploratory Data Analysis, plotting boxplot, hist_plot, pair_plot and heatmap
        :return: None
        """
        try:
            self.numerical_cols = [col for col in self.df.select_dtypes(exclude=('object','str')).columns if col != 'Outcome']
            # Box-Plot
            plt.figure(figsize=(15, 6))
            plt.title("Box-plot for Numeric Columns", weight='bold', fontsize=20, color='Maroon')
            for i, col in enumerate(self.numerical_cols):
                plt.subplot(3, 3, i + 1)
                sns.boxplot(self.df[col], orient='h', palette='muted', ax=plt.gca())
                plt.xlabel(col, weight='semibold', fontsize=12, color='black')
            plt.tight_layout()
            plt.show()

            #Hist-Plot
            plt.figure(figsize=(15, 8))
            plt.title("Hist-Plot", weight='bold', fontsize=20, color='Maroon')
            for i, col in enumerate(self.numerical_cols):
                plt.subplot(3, 3, i + 1)
                sns.histplot(self.df[col], palette='dark', ax=plt.gca(), kde=True)
                plt.xlabel(col, weight='semibold', fontsize=12, color='black')
            plt.tight_layout()
            plt.show()

            # Heat-Map
            plt.figure(figsize=(18, 10))
            plt.title("Correlation Matrix", weight='bold', fontsize=20, color='Red')
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, linewidths=0.5, cmap='rocket')
            plt.show()

            # Pair-plot
            sns.pairplot(self.df, hue='Outcome', palette='viridis', corner=True)
            plt.show()

        except Exception as e:
            print(f"Error during EDA: {e}", end=separate)

    def outliers(self):
        """
        This function is used to display outliers and clipped it to lower & upper bounds!
        :return: None
        """
        try:
            for col in self.numerical_cols:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                print(f"Total-Outliers for {col} : ",outliers.sum())

        except Exception as e:
            print(f"Error while displaying outliers: {e}", end=separate)

    def splitData(self):
        """
        This function is used to split data into train and test.
        :return: None
        """
        try:
            self.x = self.df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
            self.y = self.df["Outcome"]

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3,
                                                                                    random_state=1)
            print(f"x-shape: {self.x.shape}, y-shape: {self.y.shape}")
        except Exception as e:
            print(f"Error while splitting: {e}", end=separate)

    def trainingData(self):
        """
        This function is used to train the model
        :return: None
        """
        try:
            self.clf = RandomForestClassifier(n_estimators=100, random_state=1, ccp_alpha=0.008, min_samples_split=2, max_depth=6)
            self.clf.fit(self.x_train, self.y_train)
            self.y_predict = self.clf.predict(self.x_test)
        except Exception as e:
            print(f"Error while training: {e}", end=separate)


    def evaluation_Metrics_And_SavePrediction(self):
        """
        This function is used to evaluate the model, and predict the x, so that we can insert prediction into db
        :return: predicted
        """
        try:
            print(f" Accuracy: {accuracy_score(self.y_test, self.y_predict)*100:.2f}%", end=separate)
            print(f"Confusion-Matrix: \n\n{confusion_matrix(self.y_test, self.y_predict)}", end=separate)
            print(f"Classification-report: \n\n{classification_report(self.y_test, self.y_predict)}", end=separate)

            # This is used to add prediction column into SQLSERVER-Database
            predicted = self.clf.predict(self.x)
            predicted = ["Diabetes" if i==1  else "Non-diabetes" for i in predicted]
            return predicted

        except Exception as e:
            print(f"Error while evaluation: {e}", end=separate)


def main():
    server = "localhost"
    database = "alok"

    # Initialize loader
    loader = MSSQLDatabase(server, database)  # object
    df, table_name = loader.loadTable()
    c1 = Classifier(df)
    c1.displayData()
    c1.dataUnderstanding()
    c1.checkNullDuplicates()
    c1.EDA()
    c1.outliers()
    c1.splitData()
    c1.trainingData()
    predicted = c1.evaluation_Metrics_And_SavePrediction()
    loader.save_prediction(predicted)     # this is used to save prediction in database

if __name__ == "__main__":
    main()