import os
import math
import datetime
import numpy as np
import pandas as pd
import pandas_datareader
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

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


def pct_change(before, after):
    return (after / before) - 1.0


def analyze_golden_cross(df):
    '''
    ゴールデンクロスの翌日から株価(終値)がどう変化したか求める
    デッドクロスになったらそこで処理を終える
    '''
    results = []
    for date, _ in df[df['golden_cross']].iterrows():
        result = []
        close_at_golden_cross = df.loc[date]['Close']
        row = df.index.get_loc(date)
        for _, s in df.iloc[row+1:].iterrows():
            result.append(pct_change(close_at_golden_cross, s['Close']))
            if s['dead_cross']:
                results.append(result)
                result = []
                break
    if len(result) > 0:
        results.append(result)

    # 平均変化率を求める
    all_result = []
    for result in results:
        all_result.extend(result)
    return np.array(all_result)


def analyze_dead_cross(df):
    '''
    デッドクロスの翌日から株価(終値)がどう変化したか求める
    ゴールデンクロスになったらそこで処理を終える
    '''
    results = []
    for date, _ in df[df['dead_cross']].iterrows():
        result = []
        close_at_golden_cross = df.loc[date]['Close']
        row = df.index.get_loc(date)
        for _, s in df.iloc[row+1:].iterrows():
            result.append(pct_change(close_at_golden_cross, s['Close']))
            if s['golden_cross']:
                results.append(result)
                result = []
                break
    if len(result) > 0:
        results.append(result)

    # 平均変化率を求める
    all_result = []
    for result in results:
        all_result.extend(result)
    return np.array(all_result)


def main():
    df = get_stock('^N225', START_DATE, END_DATE)

    sns.set(font='IPAexGothic')

    golden_cross_result = []
    dead_cross_result = []

    for long_term in np.arange(3, 251):
        for short_term in np.arange(2, long_term):
            # 移動平均を求める
            df['short_ma'] = talib.SMA(df['Close'], timeperiod=short_term)
            df['long_ma'] = talib.SMA(df['Close'], timeperiod=long_term)

            # クロスしている箇所を見つける
            df['diff'] = df['long_ma'] - df['short_ma']
            df_cross = df[df['diff'] * df['diff'].shift() < 0]
            df['golden_cross'] = df.index.isin(
                df_cross[df_cross['diff'] < 0].index)
            df['dead_cross'] = df.index.isin(
                df_cross[df_cross['diff'] > 0].index)

            result = analyze_golden_cross(df)
            golden_cross_avg = np.average(result)
            golden_cross_result.append(
                {'long': long_term, 'short': short_term, 'avg': golden_cross_avg})

            result = analyze_dead_cross(df)
            dead_cross_avg = np.average(result)
            dead_cross_result.append(
                {'long': long_term, 'short': short_term, 'avg': dead_cross_avg})
            print(
                f'long={long_term}, short={short_term}, g-avg={golden_cross_avg}, d-avg={dead_cross_avg}')

    # 移動平均期間の組み合わせによる変化率(ゴールデンクロス)
    results = pd.DataFrame(golden_cross_result).pivot('short', 'long', 'avg')
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(results, square=True, cmap='Reds', ax=ax)
    ax.set_title('移動平均期間の組み合わせによる変化率(ゴールデンクロス)')
    ax.invert_yaxis()
    ax.grid()
    plt.savefig('cross3-1.png')
    plt.show()

    # 移動平均期間の組み合わせによる変化率(デッドクロス)
    results = pd.DataFrame(dead_cross_result).pivot('short', 'long', 'avg')
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(results, square=True, cmap='Blues_r', ax=ax)
    ax.set_title('移動平均期間の組み合わせによる変化率(デッドクロス)')
    ax.invert_yaxis()
    ax.grid()
    plt.savefig('cross3-2.png')
    plt.show()


if __name__ == '__main__':
    main()
