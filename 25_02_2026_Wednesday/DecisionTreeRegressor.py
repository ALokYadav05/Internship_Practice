import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

separate = f"\n\n{'--' * 100}\n\n"  # this is used only to separate print statements


class Regression:
    def __init__(self):
        self.df = pd.read_csv('insurance.csv')
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.numerical_cols = None


    def display_data(self):
        """
        This function displays the data
        :return: None
        """
        try:
            print(self.df.head(), end=separate)
        except Exception as e:
            print(f"Error while displaying data: {e}", end=separate)


    def understanding_data(self):
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


    def check_null_duplicates(self):
        """
        This function is used to check null and duplicate values
        :return: None
        """
        try:
            print("Null values: \n\n",self.df.isnull().sum(), end=separate)
            print("Total-Duplicates Before:",self.df.duplicated().sum(), end=separate)

        except Exception as e:
            print(f"Error while checking null values: {e}", end=separate)


    def handing_duplicates(self):
        """
        This function is used to handle duplicate values
        :return: None
        """
        try:
            self.df.drop_duplicates(keep='first', inplace=True)   #droping duplicates
            print("Total-Duplicates After:",self.df.duplicated().sum(), end=separate)
        except Exception as e:
            print(f"Error while Handling duplicate values: {e}", end=separate)


    def eda(self):
        """
        This function is used to perform Exploratory Data Analysis, plotting boxplot, hist_plot, pair_plot and heatmap
        :return: None
        """
        try:
            self.numerical_cols = [col for col in self.df.select_dtypes('number').columns if col!='charges']

            #Box-Plot
            plt.figure(figsize=(15, 6))
            plt.title("Box-plot for Numeric Columns", weight='bold', fontsize=20)
            for i, col in enumerate(self.numerical_cols):
                plt.subplot(3,1,i+1)
                sns.boxplot(self.df[col],orient='h',palette='muted',ax=plt.gca())
                plt.xlabel(col, weight='semibold', fontsize=12, color='maroon')
            plt.tight_layout()
            plt.show()

            #Hist-Plot
            plt.figure(figsize=(15, 8))
            plt.title("Hist-Plot", weight='bold', fontsize=20)
            for i, col in enumerate(self.numerical_cols):
                plt.subplot(3,1,i+1)
                sns.histplot(self.df[col],palette='dark',ax=plt.gca(), kde=True)
                plt.xlabel(col, weight='semibold', fontsize=12, color='maroon')
            plt.tight_layout()
            plt.show()

            #Heat-Map
            plt.figure(figsize=(18, 10))
            plt.title("Correlation Matrix", weight='bold', fontsize=20)
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr,annot=True,linewidths=0.5,cmap='mako')
            plt.show()

            # Pair-plot
            plt.figure(figsize=(20, 10))
            sns.pairplot(self.df, hue='smoker', palette='rocket',corner=True)
            plt.show()

        except Exception as e:
            print(f"Error during EDA: {e}", end=separate)


    def outliers(self):
        """
        This function is used to display outliers, I am not handling and clipping it because it's very few in numbers!
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


    def split_train_evaluate(self):
        """
        This function is used to split data into train and test, encode it using One-hot-encoding.
        :return: None
        """

        try:
            cat_features = self.df.select_dtypes(exclude='number').columns.tolist()

            self.x = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=1)

            # It is used to transform specific columns, here it's categorical
            transformer = ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
               ], remainder='passthrough')

            model = Pipeline(steps=
                             [('Transformation',transformer), ('Regressor',
                            DecisionTreeRegressor(max_depth=3,min_samples_split=5,ccp_alpha=0.09, min_samples_leaf=10))
                            ])

            model.fit(self.x_train, self.y_train)
            y_predict = model.predict(self.x_test)

            print("MSE :",mean_squared_error(y_predict, self.y_test))
            print("MAE :",mean_absolute_error(y_predict, self.y_test))
            print(f"R2 Score : {r2_score(self.y_test, y_predict)*100:.2f}%")

            # This is used to display grown tree
            plt.figure(figsize=(20, 10))
            plot_tree(model)
            plt.show()

        except Exception as e:
            print(f"Unexpected Error : {e}", end=separate)



def main_func():
    """
    This is the main function, entry-point
    :return:
    """
    try:
        d1 = Regression()
        d1.display_data()
        d1.understanding_data()
        d1.check_null_duplicates()
        d1.handing_duplicates()
        d1.eda()
        d1.outliers()
        d1.split_train_evaluate()

    except Exception as e:
        print("Unexpected error:", e)



if __name__ =='__main__':
    main_func()