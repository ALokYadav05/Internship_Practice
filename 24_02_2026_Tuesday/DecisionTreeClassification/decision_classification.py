import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


separate = f"\n\n{'--' * 100}\n\n"
pd.set_option('display.width', None)  # used to display full-width of columns


def read_data():
    """
    This function is used to read the data from the csv file
    :return: dataframe
    """
    try:
        df = pd.read_csv('../Dataset/winequality-red.csv')
        print(df.head(), end=separate)
        return df
    except Exception as e:
        print(f"Error while loading data : {e}")


def data_understanding(df):
    """
    This function is used to understand and explore the data
    :param df: original dataframe
    :return: None
    """
    try:
        print(f" Columns:\n\n {df.columns.tolist()}", end=separate)
        print(df.info(), end=separate)
        print(f"Data-types:\n\n {df.dtypes}", end=separate)
        print(f"Stats: \n\n {df.describe()}", end=separate)
        print(f"Target:\n\n {df['quality'].value_counts()}", end=separate)

    except Exception as e:
        print(f"Error while understanding data : {e}")


def check_null_duplication(df):
    """
    This function is used to check Nulls and duplicates value in each column
    :param df: dataframe
    :return: None
    """
    try:
        print(f"NUll :\n\n {df.isnull().sum()}", end=separate)
        print(f"Duplicates : {df.duplicated().sum()}", end=separate)

    except Exception as e:
        print(f"Error While checking Null & Duplicates : {e}")

def handling_duplicates(df):
    """
    This function is used to handle duplicate values in each column
    :param df: dataframe
    :return: None
    """
    try:
        df.drop_duplicates(keep='first', inplace=True)
        print("Dropped Duplicates!!")
        print(f"Duplicates : {df.duplicated().sum()}", end=separate)

    except Exception as e:
        print(f"Error While dropping Duplicates : {e}")


def handling_target(df):
    """
    This function is used to make multi-class target into binary class , By doing -> (3,4,5) into class 0 , (6,7,8) into class 1
    :param df: dataframe
    :return: None
    """
    try:
        df['quality'] = df['quality'].apply(lambda x : 1 if x >= 6 else 0)
        print("Tuned the Target column :", end=separate)
        print(df['quality'].value_counts())

        # this is another-way to deal with class imbalance, we have muli-class classification!

        # We use 2 as the start to include the score 3
        # Bins: (2-4], (4-6], (6-8]
        # bins = [2,4,6,8]
        # labels = ['low', 'medium', 'high']
        # df['quality'] = pd.cut(df['quality'], bins = bins, labels = labels)
        # print(df['quality'].value_counts())

    except Exception as e:
        print(f"Error While Handling-Target : {e}")


def pair_plot_numeric(df):
    """
    This function is used to plot a pair-plot, that is used to compare relation of one numeric feature with another numeric feature
    :param df: dataframe
    :return: None
    """
    try:
        plt.figure(figsize=(15,5))
        sns.pairplot(data=df, hue='quality')
        plt.show()
    except Exception as e:
        print(f"Error While plotting pair_plot : {e}")


def outliers_detection(df):
    """
    This function is used to detect outliers and clip it to lower & upper bound
    :param df: dataframe
    :return: None
    """
    try:
        columns = df.columns.tolist()
        for i, cols in enumerate(columns):
            plt.subplot(4,3,i+1)
            plt.grid(alpha=0.3)
            sns.boxplot(x=cols, data=df, palette="muted")
            plt.title(cols, fontsize=10, weight='semibold')
        plt.tight_layout()
        plt.show()

        cols = [col for col in df.columns if col != 'quality']

        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_fence = q1-1.5*iqr
            upper_fence = q3+1.5*iqr
            outliers = (df[col] < lower_fence) | (df[col] > upper_fence)
            print(f"Total-outliers for column {col} : {outliers.sum()}")
            df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)
        print("Clipped Outliers Successfully!!")
        print(end=separate)

    except Exception as e:
        print(f"Error While Handling Outliers : {e}")


def correlation_matrix(df):
    """
    This function is used to plot a correlation matrix, which is heat-map
    :param df: dataframe
    :return: None
    """
    try:
        corr = df.corr(numeric_only=True, method='pearson')
        sns.heatmap(corr, annot=True)
        plt.title('Correlation Matrix', fontsize=15, weight='semibold')
        plt.show()

    except Exception as e:
        print(f"Error While Handling Correlation Matrix : {e}")


def split_train_evaluate(df):
    """
    This function is used to train-test-split, training the model and testing it accuracy using different  metrics
    :param df: dataframe
    :return: None
    """
    try:
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=69)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        classifier = DecisionTreeClassifier(max_depth=5, random_state=69, max_leaf_nodes=20,class_weight='balanced')
        classifier.fit(x_train, y_train)
        classifier.fit(x_train, y_train)
        classifier.fit(x_train, y_train)
        classifier.fit(x_train, y_train)
        y_predict = classifier.predict(x_test)
        print(f"Accuracy : {accuracy_score(y_test, y_predict)*100:.2f}%", end=separate)
        print(f"Classification Report:\n\n {classification_report(y_test, y_predict)}", end=separate)
        print(f"Confusion Matrix:\n\n {confusion_matrix(y_test, y_predict)}", end=separate)

    except Exception as e:
        print(f"Error While Split Train Evaluate : {e}")


def main_func():
    """
    This function is used to run main function, entry-point
    :return: None
    """

    try:
        df = read_data()
        if df is not None:
            data_understanding(df)
            check_null_duplication(df)
            handling_duplicates(df)
            handling_target(df)
            pair_plot_numeric(df)
            outliers_detection(df)
            correlation_matrix(df)
            split_train_evaluate(df)

    except Exception as e:
        print(f"UnExpected Error: {e}")


if __name__ =='__main__':
    main_func()

