import numpy as np
import pandas as pd

path = "C:/Users/alok.yadav/.cache/kagglehub/datasets/juhibhojani/house-price/versions/1/house_prices.csv"
df = pd.read_csv(path)
# print(df.head())

# print(df.info())

# print(df.columns)

# print(df.describe(include='all'))

# print(df.shape)

# print(df.isna().sum())

# print(df['Title'].isnull().sum())

print(df.isnull.sum())