import pandas as pd

s = pd.Series([1,2,3,4,5], index=['A','B','C','D','E'])
#
# df = pd.DataFrame(s, columns=['Series'])
df = pd.DataFrame({'series':s})
print(df)