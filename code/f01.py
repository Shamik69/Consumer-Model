import pandas as pd

path = 'C:/Users/User/PycharmProjects/Consumer-Model'
df = pd.read_csv(f'{path}/data/Mall_Customers.csv')


def fno1(dataframe: pd.DataFrame):
    df = dataframe
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    df.insert(loc=2, column='Male',
              value=[x + 1 if x == 0 else x - 1 for x in df['Gender']])
    df.insert(loc=3, column='Female',
              value=[x for x in df['Gender']])
    df.insert(loc=4, column='Annual Income',
              value=[x*1000 for x in df['Annual Income (k$)']])
    df = df.drop(['CustomerID', 'Gender', 'Annual Income (k$)'], axis=1)
    return df


df= pd.read_csv(f'{path}/data/modified.csv')

from sklearn import preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
kmeans= cluster.KMeans(5)
df['clusters']= kmeans.fit_predict(
        preprocessing.scale(df[['Annual Income', 'Spending Score (1-100)']]))
df.to_csv(f'{path}/data/modified.csv', index=False)
sns.scatterplot(x='Annual Income',y='Spending Score (1-100)',hue='clusters',palette=sns.color_palette("hls",5),data=df,legend="full")
plt.xlabel('Income (annual)')
plt.ylabel('Spending')