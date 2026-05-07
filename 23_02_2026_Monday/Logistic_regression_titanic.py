import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


separate = f"\n\n{'--' * 100}\n\n"
pd.set_option('display.width',1000)    # it is used to display all columns in one-line
pd.set_option('display.max_columns', None)     # it is used to display all columns (without it , we get something like ...)


def read_data():
    """
    This function is used to read the data from csv file
    :return: dataframe
    """
    df = pd.read_csv('Titanic_dataset.csv')
    print(df.head(), end=separate)
    return df


def data_understanding(df):
    """
    This function is used to understand the data, like its data-types and statistics
    :param df: original dataframe
    :return: None
    """
    print(df.columns.tolist(), end=separate)
    print(df.dtypes, end=separate)
    print(df.info(), end=separate)
    print(df.describe(), end=separate)

def check_null_and_duplicate(df):
    """
    This function is used to check the null values and duplicates
    :param df: original dataframe
    :return: None
    """
    print("Null-Values : \n\n",df.isnull().sum(), end=separate)
    print("Duplicates:",df.duplicated().sum(), end=separate)
    print(df['Age'].value_counts(), end=separate)


def plot_age_and_fill_null(df):
    """
    This function is used to plot the age and fill null values!
    :param df: original dataframe
    :return: None
    """
    plt.figure(figsize=(10,10))
    sns.boxplot(df['Age'], palette="viridis", orient='h')
    plt.title('Box-plot for Age', fontsize=20, weight='bold')
    plt.show()

    df['Age'] = df['Age'].fillna(df['Age'].median())
    print("Null-for-age : ",df['Age'].isnull().sum(), end=separate)


def handling_cabin_col(df):
    """
    This function is used to drop the cabin column, because it has too many null-values
    :param df: original dataframe
    :return: None
    """
    df.drop('Cabin', axis=1, inplace=True) # dropped because too-many null values
    print(df.head(), end=separate)


def handling_embarked_col(df):
    """
    This function is used to fill null values in Embarked columns, using mode frequency
    :param df: original dataframe
    :return: None
    """
    print(df['Embarked'].value_counts(), end=separate)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    print('Embarked: ',df.Embarked.isnull().sum(), end=separate)


def outliers_detection(df):
    """
    This function is used to detect outliers and clip it into lower-fence and upper-fence
    :param df: original dataframe
    :return: None
    """

    numeric_cols = [col for col in df.columns if df[col].dtype != 'str' and col != 'Survived' and col != 'Parch']

    for i, col in enumerate(numeric_cols):
        plt.subplot(3,2,i+1)
        plt.title(col, fontsize=15, weight='bold')
        sns.boxplot(df[col], palette="viridis")

    plt.tight_layout()
    plt.show()

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1-1.5*iqr
        upper_fence = q3+1.5*iqr

        outlier = (df[col] < lower_fence) | (df[col] > upper_fence)
        print("Total-Outliers : ",outlier.sum())            # displaying total outliers

        df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)   #clipping outlier to lower-bound and upper-bound
    print("Clipped-outliers, Successfully", end=separate)

def heatmap_plot(df):
    """
    This function is used to plot heatmap, that shows correlation of one numeric feature with another
    :param df: original dataframe
    :return: None
    """
    plt.figure(figsize=(18,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis")
    plt.title("Correlation Matrix", fontsize=20, weight='bold')
    plt.show()

def split_train_predict(df):
    """
    This function is used to split ,train and test the model
    :param df: original dataframe
    :return: None
    """
    y = df['Survived']
    x = df.drop('Survived', axis=1)

    print(f" x-shape: {x.shape}, Y-shape: {y.shape}" ,end=separate)

    cat_feature = x.select_dtypes(include='str').columns.tolist()
    num_feature = x.select_dtypes(include='number').columns.tolist()

    pre_process = ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_feature),
                                  ('num', StandardScaler(), num_feature)]
                )

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)

    model= Pipeline(
                [('pre_process', pre_process),
                 ('logistic_regression', LogisticRegression())
                 ])

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_predict)*100:.2f}%", end=separate)

    con_mat = confusion_matrix(y_test, y_predict)
    print(f" Confusion Matrix:\n {con_mat}", end=separate)

    print(f" Classification Report: \n{classification_report(y_test, y_predict)}", end=separate)




if __name__ == '__main__':
    data = read_data()
    data_understanding(data)
    check_null_and_duplicate(data)
    plot_age_and_fill_null(data)
    handling_cabin_col(data)
    handling_embarked_col(data)
    outliers_detection(data)
    heatmap_plot(data)
    split_train_predict(data)


