import os
import datetime
import pandas as pd
import pandas_datareader
import talib
import matplotlib.pyplot as plt
import seaborn as sns

START_DATE = datetime.date(2011, 1, 1)
END_DATE = datetime.date(2020, 12, 31)


def get_stock(ticker, start_date, end_date):
    '''
    get stock data from Yahoo Finance
    '''
    dirname = 'data'
    os.makedirs(dirname, exist_ok=True)
    fname = f'{dirname}/{ticker}.pkl'
    df_stock = pd.DataFrame()
    if os.path.exists(fname):
        df_stock = pd.read_pickle(fname)
        start_date = df_stock.index.max() + datetime.timedelta(days=1)
    if end_date > start_date:
        df = pandas_datareader.data.DataReader(
            ticker, 'yahoo', start_date, end_date)
        df_stock = pd.concat([df_stock, df[~df.index.isin(df_stock.index)]])
        df_stock.to_pickle(fname)
    return df_stock


def main():
    short_term = 25
    long_term = 75

    df = get_stock('^N225', START_DATE, END_DATE)

    # 移動平均を求める
    df['short_ma'] = talib.SMA(df['Close'], timeperiod=short_term)
    df['long_ma'] = talib.SMA(df['Close'], timeperiod=long_term)

    # クロスしている箇所を見つける
    df['diff'] = df['long_ma'] - df['short_ma']
    df_cross = df[df['diff'] * df['diff'].shift() < 0]
    df['golden_cross'] = df.index.isin(df_cross[df_cross['diff'] < 0].index)
    df['dead_cross'] = df.index.isin(df_cross[df_cross['diff'] > 0].index)

    # グラフで表示する
    sns.set()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df['short_ma'], label=f'short ({short_term} days)')
    ax.plot(df['long_ma'], label=f'long ({long_term} days)')
    ax.set_title('Golden Cross / Dead Cross')
    ax.legend()
    for index, row in df[df['golden_cross']].iterrows():
        ax.annotate('G', xy=(index, row['short_ma']),
                    xytext=(index, row['short_ma'] + 1500),
                    size=10, color='red',
                    arrowprops={'arrowstyle': '->', 'color': 'red'})
    for index, row in df[df['dead_cross']].iterrows():
        ax.annotate('D', xy=(index, row['short_ma']),
                    xytext=(index, row['short_ma'] - 1500),
                    size=10, color='blue',
                    arrowprops={'arrowstyle': '->', 'color': 'blue'})
    plt.savefig('cross1.png')
    plt.show()


if __name__ == '__main__':
    main()
