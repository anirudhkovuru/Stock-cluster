import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='anirudhkovuru', api_key='tAqAyR74bwrdbpQ02JxS')

t_data = pd.read_csv('./stocks/AAL.csv')
t_data = t_data.iloc[1:6, :]
print(t_data)

trace = go.Candlestick(x=t_data.date,
                       open=t_data.open,
                       high=t_data.high,
                       low=t_data.low,
                       close=t_data.close)
data = [trace]
layout = {
    'title': 'Trends in AAL stocks',
    'yaxis': {'title': 'AAL Stock'},
    'xaxis': {
        'title': 'Dates',
        'rangeslider': {
            'visible': False
        }
    },
}

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='simple_candlestick')
