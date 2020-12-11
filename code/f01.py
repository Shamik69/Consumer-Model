import pandas as pd
from sklearn import preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

path = 'C:/Users/User/PycharmProjects/Consumer-Model'
df = pd.read_csv(f'{path}/data/Mall_Customers.csv')


def fno1(dataframe: pd.DataFrame):
    df = dataframe
    map_dict = {'Male': 0, 'Female': 1}
    df['Gender'] = df['Gender'].map(map_dict)

    df.insert(loc=2, column='Male',
              value=[x + 1 if x == 0 else x - 1 for x in df['Gender']])
    df.insert(loc=3, column='Female',
              value=[x for x in df['Gender']])
    df.insert(loc=4, column='Annual Income',
              value=[x * 1000 for x in df['Annual Income (k$)']])
    df = df.drop(['CustomerID', 'Annual Income (k$)'], axis=1)
    df['Gender'] = df['Gender'].map({map_dict[x]: x for x in map_dict.keys()})
    df.to_csv(f'{path}/data/modified.csv', index=False)
    return df


fno1(df)
df = pd.read_csv(f'{path}/data/modified.csv')


def clustering(df: pd.DataFrame, x: 'independent variable', run_counter,
               y: 'dependant variable' = 'Spending Score (1-100)',
               c: 'clustering variable' = None):
    sns.set()

    scaled = preprocessing.scale(df[[x, y]])
    knee= KneeLocator(x=[i + 1 for i in range(10)],
                      y=[cluster.KMeans(i + 1).fit(scaled).inertia_ for i in range(10)], direction='decreasing').knee
    print(knee)
    kmeans = cluster.KMeans(knee)
    clusters = kmeans.fit_predict(scaled)
    df['clusters'] = clusters
    print(df['clusters'].unique().shape[0])
    df.to_csv(f'{path}/data/modified.csv', index=False)
    c_var = 'clusters' if c is None else c
    sns.scatterplot(x=x, y=y, hue=c_var,
                    data=df, legend="full")
    plt.xlabel(x)
    plt.ylabel('Spending')
    plt.savefig(f'{path}/figs/fig{run_counter}({x}-{y}-{c}).jpeg')
    plt.close()


run = 0
for i in 'Age', 'Annual Income':
    for j in 'Gender', None:
        run += 1
        clustering(df=df, x=i, c=j, run_counter=run)
