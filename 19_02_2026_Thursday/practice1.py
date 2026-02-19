import seaborn as sns
import matplotlib.pyplot as plt

# print(sns.get_dataset_names())

df = sns.load_dataset('tips')
print(df)

sns.set_style("darkgrid")

plt.figure(figsize = (10,10))
plt.hist(df['total_bill'],color = 'blue')
plt.xlabel('Total Bill',fontsize = 15)
plt.ylabel('Frequency',fontsize = 15)
plt.title('Histogram of Total Bill',fontsize = 20, weight = 'bold', pad=15, color='red')
# sns.histplot(df['total_bill'], color = 'blue')
plt.tight_layout()
plt.show()


print(df.isnull().sum())

