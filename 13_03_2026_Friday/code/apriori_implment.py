import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

seperator = f"\n\n{'---' * 70}\n\n"

class AprioriAlgorithm:
    """
    This class is used to demonstrate implementation of apriori algorithm on groceries dataset
    """
    def __init__(self):
        self.df = None
        self.df_encoded = None
        self.basket = None
        self.transactions = None
        self.encoder = TransactionEncoder()
        self.algo = None
        self.rules = None

    def load_data(self):
        """
        This method is used to Load the  data from csv file
        returns: None
        """
        try:
            self.df = pd.read_csv("../Dataset/Groceries_Dataset.csv")
            print(f"\nDataset = \n{self.df.head()}\n")
            print("Dataset loaded successfully", end = seperator)
        except FileNotFoundError:
            print("File not found")
        except Exception as e:
            print(e)

    def preprocess(self):
        """
        This method is used to preprocess the data
        return: None
        """
        try:
            self.load_data()
            print(f"Columns : {self.df.columns.tolist()}", end=seperator)
            self.df.info()
            print(end=seperator)
            print(f"Null-Values :\n\n{self.df.isnull().sum()}", end=seperator)
            print(f"Total-Duplicates : {self.df.duplicated().sum()}", end=seperator)
            print(f"Stats: \n\n{self.df.describe()}", end=seperator)
        except Exception as e:
            print(f"Error in preprocessing: {e}")

    def group_items(self):
        """
        This method is used to Group items by transaction
        returns: None
        """
        try:
            self.preprocess()
            self.basket = self.df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()
            self.transactions = self.basket["itemDescription"].tolist()
            print(f"Transactions = {len(self.transactions)}\n")
            print("\nSuccessfully grouped items by transactions", end = seperator)
        except Exception as e:
            print("Error while grouping items by transactions : ", e)

    def feature_encoding(self):
        """
        This method is used to do One hot encoding of transactions
        returns: None
        """
        try:
            self.group_items()
            encoder_array = self.encoder.fit_transform(self.transactions)
            self.df_encoded = pd.DataFrame(encoder_array, columns = self.encoder.columns_)
            print("Feature encoding successful", end=seperator)
        except Exception as e:
            print(e)

    def run_algorithm(self):
        """
        This method is used to run Apriori Algorithm
        returns: None
        """
        try:
            self.feature_encoding()
            self.algo = apriori(
                self.df_encoded,
                min_support = 0.01,
                use_colnames = True
            )
            print(f"Total Frequent Item-sets = {self.algo.shape[0]}\n")
            print("Model Training Completed", end = seperator)
        except Exception as e:
            print(e)

    def generate_association_rules(self):
        """
        This method is used to generate association rules

        .return: None
        """
        try:
            self.run_algorithm()
            # generate rules
            rules_df = association_rules(
                self.algo,
                metric="confidence",
                min_threshold=0.1
            )
            # Check if rules_df is valid before processing
            if rules_df is None or rules_df.empty:
                print("No association rules were generated.")
                return

            # filter rules
            self.rules = rules_df[
                rules_df['antecedents'].apply(lambda x: len(x) >= 1) &
                rules_df['consequents'].apply(lambda x: len(x) >= 1)
                ]

            print("Association Rules:", self.rules.shape[0])
            print(self.rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
        except Exception as e:
            print(f"Error in generating rules: {e}")

    def visualize(self):
        """
        Visualize the results of Apriori Algorithm
        :return: None
        """
        try:
            self.generate_association_rules()
            top_items = self.df['itemDescription'].value_counts().head(10)
            plt.figure(figsize = (15,9))
            sns.barplot(top_items,palette="muted")
            plt.title("Top 10 Most Purchased Items", weight='bold', color='red', fontsize=20)
            plt.xlabel("Item", weight='semibold', fontsize=15)
            plt.ylabel("Count", weight='semibold', fontsize=15)
            plt.xticks(rotation=20)
            plt.show()
        except Exception as e:
            print(f"Error in visualization: {e}")

def main():
    algorithm = AprioriAlgorithm()
    algorithm.visualize()

if __name__ == "__main__":
    main()