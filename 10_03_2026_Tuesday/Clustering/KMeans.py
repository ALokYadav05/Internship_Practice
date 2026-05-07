import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

separator = f"\n\n{'--'*50}\n"

class Cluster:
    def __init__(self):
        self.df = pd.read_csv('../Dataset/Mall_Customers.csv')
        self.numeric_feat = None
        self.encoder = OneHotEncoder(drop='first',sparse_output=False)
        self.scaler = StandardScaler()
        self.wcss = None
        self.model = None

    def read_data(self):
        """
        This method is used to read the data
        :return: None
        """
        try:
            print(self.df.head(), end=separator)
            print(self.df.tail(), end=separator)
        except Exception as e:
            print(f"Error While reading data: {e}")

    def data_understanding(self):
        """
        This method is used to understand the data
        :return: None
        """
        try:
            self.read_data()
            print(f"Columns: {self.df.columns.tolist()}",end=separator)
            self.df.info()
            print(end=separator)
            print(f"Stats: \n\n{self.df.describe()}",end=separator)
        except Exception as e:
            print(f"Error While understanding the data: {e}")

    def check_null_duplicates(self):
        """
        This method is used to check the null and duplicate values in each feature
        :return: None
        """
        try:
            self.data_understanding()
            print(f"Null-Values:\n\n{self.df.isnull().sum()}",end=separator)
            print(f"Total-duplicates: {self.df.duplicated().sum()}",end=separator)
        except Exception as e:
            print(f"Error While checking null & duplicates values: {e}")

    def drop_unnecessary_column(self):
        """
        This method is used to drop unnecessary column
        :return: None
        """
        try:
           self.check_null_duplicates()
           self.df.drop(columns=['CustomerID'], inplace=True)
        except Exception as e:
            print(f"Error While dropping unnecessary column: {e}")

    def eda(self):
        """
        This method is used to plot boxplot, heatmap, scatterplot
        :return: None
        """
        try:
            self.drop_unnecessary_column()
            self.numeric_feat = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            plt.figure(figsize = (15,10))
            for i, col in enumerate(self.numeric_feat):
                plt.subplot(1,3,i+1)
                sns.boxplot(self.df[col], palette='muted')
                plt.title(col, fontsize = 13, weight='semibold', color='red')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize = (25,10))
            sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='viridis')
            plt.title('Correlation Matrix', fontsize = 20, weight='semibold', color='red')
            plt.show()

            plt.figure(figsize = (20,10))
            sns.scatterplot(x=self.df['Annual Income (k$)'], y=self.df['Spending Score (1-100)'], hue=self.df['Gender'])
            plt.title("Scatter plot", fontsize = 20, weight='semibold', color='red')
            plt.xlabel('Annual Income', fontsize = 13, weight='semibold')
            plt.ylabel('Spending Score', fontsize=13, weight='semibold')
            plt.show()
        except Exception as e:
            print(f"Error While eda: {e}")

    def feature_encoding(self):
        """
        This method is used to encode categorical feature
        :return: None
        """
        try:
             self.eda()
             gender_encoded = self.encoder.fit_transform(self.df[['Gender']])
             column_name = self.encoder.get_feature_names_out(['Gender'])[0]
             self.df[column_name] = gender_encoded
             self.df.drop(columns=['Gender'], inplace=True)

        except Exception as e:
            print(f"Error While feature encoding: {e}")

    def feature_scaling(self):
        """
        This method is used to scale features
        :return: None
        """
        try:
            self.feature_encoding()
            scaled = self.scaler.fit_transform(self.df)
            self.df = pd.DataFrame(scaled, columns=self.df.columns)
        except Exception as e:
            print(f"Error While feature scaling: {e}")

    def elbow_method(self):
        """
        This method is used to plot the elbow method to find the optimal k-value
        :return: None
        """
        try:
            self.feature_scaling()
            self.wcss = []
            plt.figure(figsize=(15,10))
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=1)
                kmeans.fit(self.df)
                self.wcss.append(kmeans.inertia_)
            sns.lineplot(self.wcss, marker='o', color='red')
            plt.title("Elbow Method", color='red', fontsize=20, weight='bold')
            plt.xlabel("Number of clusters", fontsize=10, weight='semibold')
            plt.ylabel("WCSS", fontsize=10, weight='semibold')
            plt.show()
        except Exception as e:
            print("Error in elbow method", e)

    def training(self):
        """
        This method is used to train the model
        :return: None
        """
        try:
            self.elbow_method()
            self.model = KMeans(n_clusters=6, random_state=1)
            self.df["Clusters_formed"] = self.model.fit_predict(self.df)
        except Exception as e:
            print("Error in training the model", e)

    def evaluation(self):
        """
        This method is used to evaluate the model
        :return: None
        """
        try:
            self.training()
            score = silhouette_score(self.df, self.df['Clusters_formed'])
            print(f"Silhouette score: {score}", end=separator)
        except Exception as e:
            print("Error in evaluation", e)

    def visualize_clusters(self):
        """
        This method is used to visualize the clusters
        :return: None
        """
        try:
            self.evaluation()
            plt.figure(figsize=(15, 10))
            sns.scatterplot(x=self.df["Annual Income (k$)"], y=self.df["Spending Score (1-100)"],
                            hue=self.df["Clusters_formed"], palette="viridis")
            centers = self.model.cluster_centers_
            plt.scatter(centers[:, 2], centers[:, 3], s=300, marker='*', c='black', label='Centroids')
            plt.title("Customer Segments", color='red', fontsize=20, weight='bold')
            plt.legend()
            plt.show()
        except Exception as e:
            print("Error in visualizing clusters", e)

def main():
    c1 = Cluster()
    c1.visualize_clusters()

if __name__ == "__main__":
    main()