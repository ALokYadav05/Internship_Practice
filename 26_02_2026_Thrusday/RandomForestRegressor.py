import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename='./flight_logs.log',filemode='a')

separate = f"\n\n{'--' * 100}\n\n"  # this is used only to separate print statements
pd.set_option('display.width', None)  # used to display  all columns while printing


class Regressor:
    def __init__(self):
        self.df = pd.read_csv('flight_price_prediction.csv')
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.y_predict = None


    def display_data(self):
        """
        This function displays the data
        :return: None
        """
        try:
            logging.info("display data started")
            print(self.df.head(), end=separate)
            logging.info("display_data function executed successfully")
        except Exception as e:
            print(f"Error while displaying data: {e}", end=separate)
            logging.error(f"Error while displaying data: {e}")


    def data_understanding(self):
        """
        This function is used to understand the data shape, data-type and statistics
        :return: None
        """

        try:
            logging.info("Data-understanding Started!")
            print("Columns: \n\n", self.df.columns.tolist(), end=separate)
            print(self.df.info(), end=separate)
            print("(Rows, columns) = ", self.df.shape, end=separate)
            print("Data-types: \n\n", self.df.dtypes, end=separate)
            print("Stats: \n\n", self.df.describe(), end=separate)
            logging.info("data_understanding function executed successfully")
        except Exception as e:
            print(f"Error while understanding data: {e}", end=separate)
            logging.error(f"Error while understanding data: {e}")


    def check_null_duplicates(self):
        """
        This function is used to check null and duplicate values
        :return: None
        """
        try:
            logging.info("checking null and duplicates values started!")
            print("Null values: \n\n", self.df.isnull().sum(), end=separate)
            print("Total-Duplicates :", self.df.duplicated().sum(), end=separate)
            logging.info("Check_null_duplicates function executed successfully!")
        except Exception as e:
            print(f"Error while checking null values: {e}", end=separate)
            logging.error(f"Error while checking null values: {e}")

    def dropping_col(self):
        """
        This function is used to drop a specific columns
        :return: None
        """
        try:
            logging.info("Dropping columns function started!")
            self.df.drop(columns=['Unnamed: 0'], inplace=True)
            print(self.df.columns.tolist())
            logging.info("dropping_col function executed successfully!")
        except Exception as e:
            print(f"Error while dropping columns: {e}", end=separate)
            logging.error(f"Error while dropping columns: {e}")

    def eda(self):
        """
        This function is used to perform Exploratory Data Analysis, plotting boxplot, hist_plot, pair_plot and heatmap
        :return: None
        """
        try:
            logging.info("EDA is started!")
            self.numerical_cols = [col for col in self.df.select_dtypes(exclude=('object','str')).columns if col != 'Outcome']
            self.categorical_cols = self.df.select_dtypes(exclude=('int64','float64')).drop(columns=['flight'])

            # Box-Plot
            plt.figure(figsize=(15, 6))
            plt.title("Box-plot for Numeric Columns", weight='bold', fontsize=20, color='Red')
            for i, col in enumerate(self.numerical_cols):
                plt.subplot(1, 3, i + 1)
                sns.boxplot(self.df[col], orient='v', palette='muted', ax=plt.gca())
                plt.xlabel(col, weight='semibold', fontsize=12, color='black')
            plt.tight_layout()
            plt.show()

            # count-plot
            plt.figure(figsize=(18, 10))
            plt.title("Count-Plot", weight='bold', fontsize=20, color='Red')
            for i, cols in enumerate(self.categorical_cols):
                plt.subplot(4, 2, i + 1)
                sns.countplot(self.df[cols])
                plt.xlabel(cols, weight='semibold', fontsize=12, color='black')
            plt.tight_layout()
            plt.show()

            # Heat-Map
            plt.figure(figsize=(18, 10))
            plt.title("Correlation Matrix", weight='bold', fontsize=20, color='Red')
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, linewidths=0.5, cmap='rocket')
            plt.show()

            logging.info("EDA function executed successfully!")

        except Exception as e:
            print(f"Error during EDA: {e}", end=separate)
            logging.error(f"Error during EDA: {e}")

    def outliers(self):
        """
        This function is used to display outliers and clipped it to lower & upper bounds!
        :return: None
        """
        try:
            logging.info("detecting outliers is started!")
            for col in self.numerical_cols:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                print(f"Total-Outliers for {col} : ",outliers.sum())

                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                logging.info("Outliers function executed successfully!")

        except Exception as e:
            print(f"Error while displaying outliers: {e}", end=separate)
            logging.error(f"Error while displaying outliers: {e}")

    def split_data(self):
        """
        This function is used to split data into train and test.
        :return: None
        """
        try:
            logging.info("Splitting data is started!")
            self.x = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
            print(self.x.head(), end=separate)
            print(self.y.head(), end=separate)


            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3,
                                                                                    random_state=1)

            print(f"x-shape: {self.x.shape}, y-shape: {self.y.shape}", end=separate)
        except Exception as e:
            print(f"Error while splitting: {e}", end=separate)
            logging.error(f"Error while splitting: {e}")


    def pipeline(self):
        """
        This function is used to encode categorical features and trained the model
        :return: None
        """
        try:
            logging.info("Pipeline Started")
            ohe=OneHotEncoder(handle_unknown='ignore',drop='first')
            self.x_train = ohe.fit_transform(self.x_train)
            self.x_test = ohe.transform(self.x_test)

            logging.info("Model training Started!")
            reg = RandomForestRegressor(n_estimators=50, random_state=1, verbose=2, max_depth=10,
                                        min_samples_leaf=8, min_samples_split=10)
            reg.fit(self.x_train, self.y_train)
            logging.info("Model training Completed!")
            self.y_predict = reg.predict(self.x_test)

            # This-is used to plot tree
            plt.figure(figsize=(18, 10))
            plt.title("Tree-structure", weight='bold', fontsize=20, color='Red')
            plot_tree(reg.estimators_[0], filled=True,max_depth=3)
            plt.show()


        except Exception as e:
            print(f"Error in pipeline : {e}", end=separate)
            logging.error(f"Error in pipeline : {e}")

    def evaluation(self):
        """
        This function is used to evaluate the model performance!
        :return: None
        """
        try:
            logging.info("Evaluation Started!")
            print(f"MSE : {mean_squared_error(self.y_test, self.y_predict)}")
            print(f"MAE : {mean_absolute_error(self.y_test, self.y_predict)}")
            print(f"R2 : {r2_score(self.y_test, self.y_predict)*100:.2f}%")
        except Exception as e:
            print(f"Error in evaluation : {e}", end=separate)
            logging.error(f"Error in evaluation : {e}")

def main_func():
    """
    This is the main function, entry-point
    :return: none
    """
    try:
        r1 = Regressor()
        r1.display_data()
        r1.data_understanding()
        r1.check_null_duplicates()
        r1.dropping_col()
        r1.eda()
        r1.outliers()
        r1.split_data()
        r1.pipeline()
        r1.evaluation()

    except Exception as e:
        print("Unexpected error:", e)
        logging.error(f"Unexpected error: {e}")



if __name__ =='__main__':
    main_func()