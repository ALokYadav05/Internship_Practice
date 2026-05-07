import numpy as np
import pandas as pd
import seaborn as sns





def preprocess():
    df = sns.load_dataset('titanic')
    print(df.head())
    # print(df.info())
    # print(df.describe())
    # print(df.isnull().sum())
    df['age'] = df['age'].fillna(df['age'].mean())
    print(df['age'])
    print(f" Age : {df['age'].isna().sum()}")




if __name__ == '__main__':
    preprocess()