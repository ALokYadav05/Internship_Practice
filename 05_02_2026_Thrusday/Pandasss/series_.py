import pandas as pd

# s = pd.Series([[1,2,3], [4,5,6], [7,8,9]],index=['a','b','c'])
# print(s)

# s2  = pd.Series([34,23,50], index=['Alok','Rudra','Raj'])
# print(s2)
# print(s2['Alok'])
# print(s2.iloc[0])
# print(s2 + 5)
# print(s2)

# s1 = pd.Series([1, 2], index=["a", "b"])
# s2 = pd.Series([3, 4], index=["b", "c"])
# print(s1.values + s2.values)
# a NaN, b 6, c NaN


# ser = pd.Series([1,None,2])
# print(ser.isna())
# ser = ser.fillna(ser.mean())
# print(ser)

series1 = pd.Series([11,12,13,14,15])
series1 = series1.apply(lambda x : x+1)

# print(series1[series1 > 14 ])
print(series1.to_list())