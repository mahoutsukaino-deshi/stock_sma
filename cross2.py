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

    fig, ax = plt.subplots(figsize=(12, 8))
    for result in results:
        ax.plot(result)
    ax.set_title('ゴールデンクロス後の株価変化率推移')
    plt.xlabel('日数')
    plt.ylabel('変化率')
    plt.savefig('cross2-1.png')
    plt.show()

    # 平均変化率を求める
    all_result = []
    for result in results:
        all_result.extend(result)
    all_result = np.array(all_result)
    print(pd.DataFrame(all_result).describe())
#                  0
# count  1487.000000
# mean      0.080796
# std       0.118025
# min      -0.123680
# 25%       0.005977
# 50%       0.050059
# 75%       0.122404
# max       0.709272

    # 平均変化率の分布図
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = math.ceil(math.log(len(all_result), 2) + 1)
    ax.hist(all_result, bins=bins)
    ax.set_title('ゴールデンクロス後の平均変化率の分布')
    plt.savefig('cross2-2.png')
    plt.show()


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

    fig, ax = plt.subplots(figsize=(12, 8))
    for result in results:
        ax.plot(result)
    ax.set_title('デッドクロス後の株価変化率推移')
    plt.xlabel('日数')
    plt.ylabel('変化率')
    plt.savefig('cross2-3.png')
    plt.show()

    # 平均変化率を求める
    all_result = []
    for result in results:
        all_result.extend(result)
    all_result = np.array(all_result)
    print(pd.DataFrame(all_result).describe())
#                 0
# count  839.000000
# mean    -0.018099
# std      0.059415
# min     -0.267749
# 25%     -0.050816
# 50%     -0.014894
# 75%      0.020195
# max      0.153848

    # 平均変化率の分布図
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = math.ceil(math.log(len(all_result), 2) + 1)
    ax.hist(all_result, bins=bins)
    ax.set_title('デッドクロス後の平均変化率の分布')
    plt.savefig('cross2-4.png')
    plt.show()


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

    sns.set(font='IPAexGothic')

    analyze_golden_cross(df)
    analyze_dead_cross(df)


if __name__ == '__main__':
    main()
