import pandas as pd
df = pd.read_csv('data/PRSA_data_2010.1.1-2014.12.31.csv')
df = df.dropna(axis=0)
X_cols = ['year',  'month', 'day',  'hour', 'DEWP','TEMP','PRES', 'Iws','Is','Ir']
Y_cols = ['pm2.5']
X = df[X_cols]
Y = df[Y_cols]

train_size = 500
test_size = 100

with open(f'data/PRSA_{train_size}_{test_size}.txt', 'w') as f:
    f.write(f"{train_size} {X.shape[1]}\n")
    X.iloc[:train_size, :].to_csv(f, index=False, header=False, sep=' ')
    f.write(f"{train_size} 1\n")
    Y.iloc[:train_size, :].to_csv(f, index=False, header=False, sep=' ')

    f.write(f"{test_size} {X.shape[1]}\n")
    X.iloc[train_size:train_size+test_size, :].to_csv(f, index=False, header=False, sep=' ')
    f.write(f"{test_size} 1\n")
    Y.iloc[train_size:train_size+test_size, :].to_csv(f, index=False, header=False, sep=' ')

