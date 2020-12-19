import pandas as pd
from sklearn import preprocessing, cluster
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
import statsmodels.api as sm
import os

path = 'C:/Users/User/PycharmProjects/Consumer-Model'


def modification(dataframe: pd.DataFrame):
    df = dataframe
    map_dict = {'Male': 0, 'Female': 1}
    df['Gender'] = df['Gender'].map(map_dict)

    df.insert(loc=2, column='Male',
              value=[x + 1 if x == 0 else x - 1 for x in df['Gender']])
    df.insert(loc=3, column='Female',
              value=[x for x in df['Gender']])
    df.insert(loc=4, column='Annual Income',
              value=[x for x in df['Annual Income (k$)']])
    df.insert(loc=list(df.columns).index('Spending Score (1-100)'),
              column='Spending Score',
              value=[i for i in df['Spending Score (1-100)']])
    df = df.drop(['CustomerID', 'Annual Income (k$)', 'Spending Score (1-100)'], axis=1)
    df['Gender'] = df['Gender'].map({map_dict[x]: x for x in map_dict.keys()})
    df.to_csv(f'{path}/data/modified.csv', index=False)
    return df


def demography(df: pd.DataFrame, y: str, x='Gender'):
    data = pd.DataFrame(data=[(i, df[df['Gender'] == i][y].mean(),
                               df[df['Gender'] == i][y].var(),
                               df[df['Gender'] == i][y].shape[0]) for i in ['Male', 'Female']],
                        columns=['Gender', f'Average {y}', f'Variance in {y}', 'Population'])
    for i in data.columns[1:]:
        plt.bar(x=data[x], height=data[i], data=data)
        plt.xlabel(x)
        plt.ylabel(i)
        plt.savefig(f'{path}/figs/(demography {y}- {i}).jpeg')
        plt.close()


def clustering(df: pd.DataFrame, x: 'independent variable', run_counter=0,
               y: 'dependant variable' = 'Spending Score',
               c: 'clustering variable' = None,
               return_clusters: bool = False):
    sns.set()

    scaled = preprocessing.scale(df[[x, y]])
    if c is None:
        x_ = [i + 1 for i in range(10)]
        y_ = [cluster.KMeans(i + 1).fit(scaled).inertia_ for i in range(10)]
        knee = x_[y_.index(KneeLocator(x=y_, y=x_, direction='decreasing').knee) - 1]
        kmeans = cluster.KMeans(knee)
        clusters = kmeans.fit_predict(scaled)
        df[f'{x} clusters'] = clusters
    c_var = f'{x} clusters' if c is None else c
    if not return_clusters:
        sns.scatterplot(x=x, y=y, hue=c_var,
                        data=df, legend="full")
        plt.xlabel(x)
        plt.ylabel('Spending')
        plt.savefig(f'{path}/figs/fig{run_counter}({x}-{y}-{x if c is None else c}).jpeg')
        plt.close()
    else:
        try:
            return clusters
        except:
            pass


def reg(df: pd.DataFrame, y, x,
        return_coefs: bool = False):
    y = df[y]
    x = sm.add_constant(df[x])
    model = sm.OLS(y, x).fit()
    summary = model.summary()
    print(summary)
    html = summary.tables[1].as_html()
    df0 = pd.read_html(html, header=0, index_col=0)[0]
    x = list(df0['coef'])
    sign = ['+', '', '+']
    for i in x:
        if i < 0:
            sign[x.index(i)] = ''
    if not return_coefs:
        return f'y={sign[1]}{x[1]}*x{sign[0]}{x[0]}'
    else:
        return {'var': x[1], 'constant': x[0]}


def plot(df: pd.DataFrame, run_counter, x: 'independent variable',
         y: 'dependant variable' = 'Spending Score'):
    clusters = clustering(df, x=x, y=y, return_clusters=True)
    coefs = reg(df, x=x, y=y, return_coefs=True)
    df['c'] = clusters
    sns.scatterplot(x=x, y=y, hue='c',
                    data=df, legend="full")
    x_ = range(min(df[x]), max(df[x]) + 1)
    plt.plot(x_, [float(coefs['var']) * i + float(coefs['constant']) for i in x_])
    plt.xlabel(x)
    plt.ylabel('Spending')
    plt.savefig(f'{path}/figs/fig{run_counter}({x}-{y}-{x} with reg line).jpeg')
    plt.close()


def call(call_var: int):
    if call_var == 0:
        modification(pd.read_csv(f'{path}/data/Mall_Customers.csv'))
    elif call_var == 1:
        run = 0
        for i in 'Annual Income', 'Age':
            for j in 'Gender', None:
                run += 1
                clustering(pd.read_csv(f'{path}/data/modified.csv'), x=i, c=j, run_counter=run)
    elif call_var == 2:
        x = ['Male', 'Female', 'Annual Income', 'Age']
        y = 'Spending Score'
        reg_lines = [reg(pd.read_csv(f'{path}/data/modified.csv'), y=y, x=i) for i in x]
        print(i for i in reg_lines)
        pd.DataFrame(data={
                'independent variable (x)': x,
                'dependent variable (y)': [y] * len(x),
                'regression lines': reg_lines
        }).to_csv(f'{path}/outputs/reg_output.csv', index=False)
    elif call_var == 3:
        df = pd.read_csv(f'{path}/data/modified.csv')
        x = ['Annual Income', 'Age']
        y = 'Spending Score'
        bipasa = 0
        for i in x:
            plot(df=df, x=i, y=y, run_counter=bipasa)
    elif call_var == 4:
        df = pd.read_csv(f'{path}/data/modified.csv')
        for i in df.columns[3:]:
            demography(df, y= i)

call(4)
