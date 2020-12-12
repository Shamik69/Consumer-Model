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
    if c is None:
        x_ = [i + 1 for i in range(10)]
        y_ = [cluster.KMeans(i + 1).fit(scaled).inertia_ for i in range(10)]
        knee = x_[y_.index(KneeLocator(x=y_, y=x_, direction='decreasing').knee) - 1]
        plt.plot(x_, y_)
        plt.vlines(x=knee, ymin=min(y_), ymax=max(y_))
        plt.show()
        kmeans = cluster.KMeans(knee)
        clusters = kmeans.fit_predict(scaled)
        df[f'{x} clusters'] = clusters
    c_var = f'{x} clusters' if c is None else c
    sns.scatterplot(x=x, y=y, hue=c_var,
                    data=df, legend="full")
    plt.xlabel(x)
    plt.ylabel('Spending')
    plt.show()
    plt.savefig(f'{path}/figs/fig{run_counter}({x}-{y}-{x if c is None else c}).jpeg')
    plt.close()


def regression_anal()
