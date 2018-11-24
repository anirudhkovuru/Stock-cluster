import pandas as pd
import numpy as np
import glob

files = glob.glob('./stocks/*')

tickers = [name[9:-4] for name in files]

columns = ['date', 'o/c', 'volume', 'low', 'high', 'name']
for i, f in enumerate(files):
    t_data = pd.read_csv(f)
    o = np.asarray(t_data['open'].tolist())
    c = np.asarray(t_data['close'].tolist())
    h = np.asarray(t_data['high'].tolist())
    l = np.asarray(t_data['low'].tolist())
    volume = np.asarray(t_data['volume'].tolist())
    oc = (o-c)/o
    norm_vol = volume/np.average(volume)
    norm_h = h/o
    norm_l = l/o

    df = pd.DataFrame(columns=columns)
    df['date'] = t_data['date']
    df['o/c'] = oc
    df['volume'] = norm_vol
    df['high'] = norm_h
    df['low'] = norm_l
    df['name'] = t_data['Name']

    df.to_csv('./preprocessed_stocks/' + t_data['Name'][0] + '.csv', index=False)
