######################################################### IMPORTS #####################################################
import dash
from dash import dcc, callback, dash_table
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import yfinance
import ta as ta
import talib as taa
import datetime
from datetime import date
import string
from sklearn.linear_model import LinearRegression
import requests
import sort_dataframeby_monthorweek as sd

#### Datasets
info = pd.DataFrame()
info['Coin'] = ['ADA', 'ATOM', 'AVAX', 'AXS', 'BTC', 'ETH', 'LINK', 'LUNA1', 'MATIC', 'SOL']
info['Text'] = [
    'Lauched in 2017 by Ethereum’s co-founder Charles Hoskinson, ADA is the cryptocurrency of Cardano, a decentralized PoS blockchain platform created to serve as an alternative to PoW networks. This type of structure provides a more sustainable and ascendable blockchain reality opposed to the proof of work, which leads to large amounts of energy wasted. It is considered the largest proof of stake cryptocurrency and by the end of its launch year it reached a market cap of 10 billion dollars.',
    'Launched in 2017, ATOM it is the main cryptocurrency of Cosmos, a distributed blockchain network that allows developers to construct their personal interoperable blockchains. Cosmos is slowly growing into “the internet of blockchains” since it allows for free sharing of data and tokens across its environment. ATOM is expected to become more valuable as the number of blockchains built within the Cosmos network increases.',
    'Initially released in 2020, AVAX it is the main cryptocurrency of Avalanche, an open-source proof of stake blockchain platform. In 2021, Avalanche had a massive impact on the market, having an investment return of 3460%. The founder, Emin Gün Sirer, made it a priority to reduce the power needed to implement decentralized networks, making Avalanche one of the most eco-friendly networks in the world.',
    'Launched in 2020, AXS it is the native corporation token of Axie Infinity network. This platform was created by Sky Mavis, a Vietnamese company, and it functions as a non-fungible token-based online video game. Axie Infinity is the current trend in the metaverse space since investors’ interest in decentralized, blockchain-based virtual worlds is increasing rapidly. AXS can be used within the gameplay or capitalized in the crypto marketplace.',
    'Launched in 2009, Bitcoin is one of the first cryptocurrencies to hit the market. It consists of a decentralized peer-to-peer electronic exchange, initially created as a transactional alternative to government or financial institutions. It relies on the proof of work method to keep track of user transactions. Presently, it is the world’s most popular cryptocurrency with a single bitcoin being worth thousands of dollars.',
    'Launched in 2015, Ethereum is a blockchain-based network that relies on multiple independent computer units to manage and verify transactions through their historical data. Its native cryptocurrency is the Ether. Similarly to Bitcoin, Ethereum follows a proof of work method to control its transactions. When it was first released, 72 million coins were produced, which represented 65% of the coins in the network in 2020.',
    'Launched in 2017 by its founder Sergey Nazarov, Chainlink in a decentralized blockchain-based cryptocurrency network and it uses its token, LINK, to pay for smart contract transactions inside its network. Until 2019 the cryptocurrency traded for less than one dollar and its market capitalization was reasonably low. However, since 2020 its value has been increasing steadily, possibly due to cryptocurrency popularity.',
    'Created in 2018 by Daniel Shin and Do Kwon, Terra is an algorithm-regulated, share style stable coin platform. Its main goal is to facilitate the world-wide adoption of cryptocurrencies and the blockchain structure. LUNA1 is the native token of the platform. Its contribution substitutes the payment value chain with a simple blockchain layer and allows the merchants access to a considerably lower transaction fee.',
    'Launched in 2017, Polygon is a pioneer as a straightforward platform for Ethereum scaling and infrastructure development. The network follows a proof of stake design linking a multi-chain system with the benefits of Ethereum’s security and ecosystem. Its native currency is MATIC. This token is released monthly, and more than four billion units are currently circulating in the market.',
    'Founded in 2017 by Anatoly Yakovenko, Solana is a public blockchain system and operates as an open source. It manages transactions with the use of its native token, SOL. To ensure the validity of exchanges in the network, it combines both proof of stake and proof of history architectures. The latter eliminates the time required to verify transaction orders and allows the blockchain transaction process to be easily automated.'
]
info['img'] = ['ADA.png', 'ATOM.png', 'AVAX.png', 'AXS.png', 'BTC.png', 'ETH.png', 'LINK.png', 'LUNA1.png', 'MATIC.png',
               'SOL.png']
info['Color'] = ['#3468D1', '#2E3148', '#E84142', '#0055D5', '#FF9416', '#627EEA', '#335DD2', '#172852', '#8247E5',
                 '#100339']
info['extend'] = ['Cardano – ADA', 'Cosmos – ATOM', 'Avalanche – AVAX', 'Axie Infinity – AXS', 'Bitcoin – BTC',
                  'Ethereum – ETH', 'Chainlink – LINK', 'Terra – LUNA1', 'Polygon – MATIC', 'Solana – SOL']
info.set_index('Coin', inplace=True)

crypto_options = [
    dict(label='Coin ' + crypto, value=crypto)
    for crypto in info.index.unique()]


##### Functions
def EMA_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # exponential moving average
    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            low_ = low[i]
            high_ = high[i]
            indicator = ta.trend.EMAIndicator(close_, window).ema_indicator()
            EMA = (indicator)
            df[i] = EMA
        tech_indicators[f'{name}_{window}'] = df.stack()

    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def SMA_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # Simple Moving Average
    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            low_ = low[i]
            high_ = high[i]
            indicator = taa.SMA(close_, timeperiod=window)

            SMA = (indicator)
            df[i] = SMA
        tech_indicators[f'{name}_{window}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def RSI_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # relative stenght index, measures momentum/trend
    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            low_ = low[i]
            high_ = high[i]
            indicator = taa.RSI(close_, timeperiod=window)
            RSI = (indicator)
            df[i] = RSI
        tech_indicators[f'{name}_{window}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def ATR_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # average true range, measures volatility
    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            low_ = low[i]
            high_ = high[i]
            indicator = taa.NATR(high_, low_, close_, timeperiod=window)
            ATR = (indicator)
            df[i] = ATR
        tech_indicators[f'{name}_{window}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def Stoch_indicator(name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    tech_indicators = pd.DataFrame()
    df = (close - low) / (high - low)
    tech_indicators[f'{name}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def Volume_Trend(name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # volume was divided by 1 Million to facilitate interpretation
    tech_indicators = pd.DataFrame()
    df = (((close - close.shift(1)) / close.shift(1)) * (volume.values / 1000000))
    tech_indicators[f'{name}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def STD_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # std of the daily % change
    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            indicator = taa.STDDEV((close_ - close_.shift(1)) / (close_.shift(1)), timeperiod=window)
            std = (indicator)
            df[i] = std
        tech_indicators[f'{name}_{window}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])

    for col in tech_indicators.columns:
        for i, ii in zip(df_engineer.groupby('ticker').mean()[f'{col}'].values,
                         df_engineer.groupby('ticker').mean()[f'{col}'].index):
            df_engineer.loc[df_engineer.index.get_level_values(0) == ii, f'{col}_Adj'] = df_engineer[f'{col}'] / i
    return (df_engineer)


def KAMA_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # Kaufman's Adaptive Moving Average (KAMA) is a moving average designed to account for market noise or volatility
    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            low_ = low[i]
            high_ = high[i]
            indicator = taa.KAMA(high_, timeperiod=window)
            kama = (indicator)
            df[i] = kama
        tech_indicators[f'{name}_{window}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])

    return (df_engineer)


def MOM_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            low_ = low[i]
            high_ = high[i]
            indicator = taa.MOM(close_, timeperiod=window)

            MOM = (indicator)
            df[i] = MOM
        tech_indicators[f'{name}_{window}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def OBV_indicator(df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    tech_indicators = pd.DataFrame()
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    for i in close.columns:
        close_ = close[i]
        low_ = low[i]
        high_ = high[i]
        volume_ = volume[i]
        indicator = taa.OBV(close_, volume_)
        OBV = (indicator)
        df[i] = OBV / 100000000
    tech_indicators[f'OBV'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def ROC_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # rate of change = % of change
    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            low_ = low[i]
            high_ = high[i]
            indicator = taa.ROC(high_, timeperiod=window)
            ROC = (indicator)
            df[i] = ROC
        tech_indicators[f'{name}_{window}'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])

    return (df_engineer)


def AROON_indicator(lag_window, name, df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # The indicator consists of the "Aroon up" line, which measures the strength of the uptrend,
    # and the "Aroon down" line, which measures the strength of the downtrend.
    tech_indicators = pd.DataFrame()
    for window in lag_window:
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        for i in close.columns:
            close_ = close[i]
            low_ = low[i]
            high_ = high[i]
            ar1, ar2 = taa.AROON(high_, low_, timeperiod=window)

            df[i] = ar1
            df2[i] = ar2
        tech_indicators[f'{name}_down_{window}'] = df.stack()
        tech_indicators[f'{name}_up_{window}'] = df2.stack()

    tech_indicators.index.names = ['date', 'ticker']
    #     tech_indicators = tech_indicators.reset_index()
    #     tech_indicators['date'] = pd.to_datetime(tech_indicators['date'])
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])
    return (df_engineer)


def avg_price(df_orig):
    data = df_orig.copy()
    close = data['close'].unstack(level='ticker').copy()
    high = data['high'].unstack(level='ticker').copy()
    low = data['low'].unstack(level='ticker').copy()
    open_ = data['open'].unstack(level='ticker').copy()
    volume = data['volume'].unstack(level='ticker').copy()

    # this function compares the current price with the average price
    tech_indicators = pd.DataFrame()

    df = pd.DataFrame()
    df2 = pd.DataFrame()
    for i in close.columns:
        open__ = open_[i]
        close_ = close[i]
        low_ = low[i]
        high_ = high[i]
        indicator = (close_ / taa.AVGPRICE(open__, high_, low_, close_)) - 1
        avg = (indicator)
        df[i] = avg
    tech_indicators[f'Avg_Price_Disparity'] = df.stack()
    tech_indicators.index.names = ['date', 'ticker']
    df_engineer = pd.merge(df_orig.reset_index().copy(), tech_indicators.reset_index(), how='left',
                           on=['ticker', 'date']).set_index(['ticker', 'date'])

    return (df_engineer)


def ft_engineer(data):
    a = EMA_indicator([2, 7, 20], 'EMA', data)
    a = ATR_indicator([2, 7, 20], 'ATR', a)
    a = RSI_indicator([2, 7, 20], 'RSI', a)
    a = MOM_indicator([2, 7, 20], 'MOM', a)
    a = STD_indicator([2, 7, 20], 'STD', a)
    a = Stoch_indicator('Stoch', a)
    a = KAMA_indicator([2, 7, 20], 'KAMA', a)
    a = OBV_indicator(a)
    a = ROC_indicator([2, 7, 20], 'ROC', a)
    a = AROON_indicator([2, 7, 20], 'AROON', a)
    a = avg_price(a)
    a = SMA_indicator([2, 7, 20], 'SMA', a)

    df = a.copy()
    df['KAMA_Disparity'] = (df['KAMA_7'] / df['KAMA_20']) - 1

    return (df)


def yahoo_time(tickers):
    stock_price = pd.DataFrame()
    for i in tickers:
        df = yfinance.download(f'{i}-USD').rename(
            columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
        df = df.drop(columns='Adj Close').reset_index().rename(columns={'Date': 'date'})
        df['ticker'] = i
        stock_price = pd.concat([df, stock_price], axis=0)

    stock_price['date'] = pd.to_datetime(stock_price['date'])
    stock_price = stock_price.set_index(['ticker', 'date'])
    stock_price.index.names = 'ticker', 'date'
    stock_price = stock_price.sort_values(['ticker', 'date'])
    return (stock_price)


def gen_daily_data():
    import time
    start = time.time()
    key = '54594d7278e0fa3c0831a72c60e04b8d'
    tickers = ['ADA', 'ATOM', 'AVAX', 'AXS', 'BTC', 'ETH', 'LINK', 'LUNA1', 'MATIC',
               'SOL']
    # calls the api
    df = pd.DataFrame()
    for i in tickers:
        #         daily_df = real_time_crypto(key, [f'{i}']).dropna()
        daily_df = yahoo_time([f'{i}']).dropna()
        data = daily_df.copy()
        # creates auxiliary datasets
        # performs feature engineering
        df_daily = ft_engineer(data.copy())
        df = pd.concat([df, df_daily], axis=0)
    end = time.time()
    print(' ')
    print(f'Data generation lasted: {end - start} seconds')
    return (df)


def real_time_crypto_1h(key, companies):
    # import datetime
    stock_price = pd.DataFrame()
    for i in companies:
        if i == 'MATIC':
            url = f'https://financialmodelingprep.com/api/v3/historical-chart/1hour/{i}USD?from={date.today() - datetime.timedelta(days=50)}&to={date.today()}&apikey={key}'
            r = requests.get(url)
            data = r.json()
            d = pd.DataFrame.from_dict(data)
            # d = d.drop(columns='date')
            d['date'] = pd.to_datetime(d['date']) + datetime.timedelta(minutes=1)
            d['ticker'] = i
            stock_price = pd.concat([stock_price, d.dropna()], axis=0)

        if i != 'MATIC':
            url = f'https://financialmodelingprep.com/api/v3/historical-chart/1hour/{i}USD?from={date.today() - datetime.timedelta(days=50)}&to={date.today()}&apikey={key}'
            r = requests.get(url)
            data = r.json()
            d = pd.DataFrame.from_dict(data)
            d['ticker'] = i
            stock_price = pd.concat([stock_price, d.dropna()], axis=0)

    # stock_price.drop(['label'],axis=1)
    stock_price['date'] = pd.to_datetime(stock_price['date'])
    stock_price = stock_price.set_index(['ticker', 'date'])
    stock_price.index.names = 'ticker', 'date'
    stock_price = stock_price.sort_values(['ticker', 'date'])
    return (stock_price)


def gen_hour_data():
    import time
    start = time.time()
    key = '54594d7278e0fa3c0831a72c60e04b8d'
    tickers = ['ADA', 'ATOM', 'AVAX', 'AXS', 'BTC', 'ETH', 'LINK', 'LUNA', 'MATIC',
               'SOL']

    # calls the api
    df = pd.DataFrame()
    for i in tickers:
        daily_df = real_time_crypto_1h(key, [f'{i}']).dropna()
        data = daily_df.copy()

        # performs feature engineering
        df_daily = ft_engineer(daily_df.copy())
        df = pd.concat([df, df_daily], axis=0)
    end = time.time()
    print(' ')
    print(f'Data generation lasted: {end - start} seconds')
    return (df)


df_daily = gen_daily_data()


def make_predictions(df, range_):
    df = df.dropna()
    # set subset
    pct_final_subset = ['ATR_2', 'RSI_2', 'ATR_20', 'Avg_Price_Disparity', 'AROON_down_2', 'AROON_up_2', 'ROC_2'
        , 'OBV', 'RSI_7', 'ROC_20']

    # create dicts (keys are the tickers)
    # create a dict for predictions
    predictions_ = {}
    # create a dict for the models
    models = {}

    # get the tickers in the df
    tickers = df.index.get_level_values(0).unique()

    # iterate all tickers
    for i in tickers:
        # create a temporary df to hold predictions
        hold_predictions = pd.DataFrame()

        # create lists to store predictions information (future predictions)
        preds = []
        dates = []

        # create lists to store predictions information (past predictions)
        old_preds = pd.DataFrame()
        dates_o = []
        preds_o = []
        for ii in np.arange(1, range_):

            # define x and y (y is the daily % change)
            x = df
            y = (df.groupby('ticker')['close'].shift(-ii) - df['close']) / (df['close'])
            y = y.dropna()
            x = x.loc[x.index.isin(y.index)]

            # define training set (all data, except the last observtion)
            x_tr = x.loc[x.index.get_level_values(0) == i]
            y_tr = y.loc[y.index.get_level_values(0) == i]
            x_t = df.loc[df.index.get_level_values(0) == i]

            # fit the model
            model = LinearRegression()
            model.fit(x_tr[pct_final_subset], y_tr)

            # get predictions
            pred = model.predict(x_t[pct_final_subset])
            pred = pred[-1]
            # get closing price to convert the predictions to normal value
            close = x_t['close'].iloc[-1]

            # convert predictions from % to normal value
            pred_ = (pred * close) + close

            # append information to the lists
            preds.append(pred_)
            date = x_t.reset_index()['date'].iloc[-1] + datetime.timedelta(days=(int(ii) - 1))
            dates.append(date)

            # condition to get past predictions
            if ii == 1:
                dates_o = x_tr.reset_index()['date']
                # predict training data
                pred_o = model.predict(x_tr[pct_final_subset])
                # convert % values to normal values
                pred_oo = (pred_o * x_tr['close'].values) + x_tr['close'].values
                close_oo = x_tr['close'].values
        # add values to the dfs
        old_preds['Date'] = dates_o
        old_preds['Prediction'] = close_oo
        old_preds['Format'] = 'Price'
        hold_predictions['Date'] = dates
        hold_predictions['Prediction'] = preds
        hold_predictions['Format'] = 'Prediction'

        # concat both dfs and add to the dict of predictions
        predictions_[f'{i}'] = pd.concat([old_preds, hold_predictions], axis=0).set_index('Date')
        models[f'{i}'] = model

    return predictions_, models


def make_predictions_hour(df, range_):
    df = df.dropna()
    # set subset
    pct_final_subset = ['ATR_2', 'RSI_2', 'ATR_20', 'Avg_Price_Disparity', 'AROON_down_2', 'AROON_up_2', 'ROC_2'
        , 'OBV', 'RSI_7', 'ROC_20']

    # create dicts (keys are the tickers)
    # create a dict for predictions
    predictions_ = {}
    # create a dict for the models
    models = {}

    # get the tickers in the df
    tickers = df.index.get_level_values(0).unique()

    # iterate all tickers
    for i in tickers:
        # create a temporary df to hold predictions
        hold_predictions = pd.DataFrame()

        # create lists to store predictions information (future predictions)
        preds = []
        dates = []

        # create lists to store predictions information (past predictions)
        old_preds = pd.DataFrame()
        dates_o = []
        preds_o = []
        for ii in np.arange(1, range_):

            # define x and y (y is the daily % change)
            x = df
            y = (df.groupby('ticker')['close'].shift(-ii) - df['close']) / (df['close'])
            y = y.dropna()
            x = x.loc[x.index.isin(y.index)]

            # define training set (all data, except the last observtion)
            x_tr = x.loc[x.index.get_level_values(0) == i]
            y_tr = y.loc[y.index.get_level_values(0) == i]
            x_t = df.loc[df.index.get_level_values(0) == i]

            # fit the model
            model = LinearRegression()
            model.fit(x_tr[pct_final_subset], y_tr)

            # get predictions
            pred = model.predict(x_t[pct_final_subset])
            pred = pred[-1]
            # get closing price to convert the predictions to normal value
            close = x_t['close'].iloc[-1]

            # convert predictions from % to normal value
            pred_ = (pred * close) + close

            # append information to the lists
            preds.append(pred_)
            date = x_t.reset_index()['date'].iloc[-1] + datetime.timedelta(hours=(int(ii) - 1))
            dates.append(date)

            # condition to get past predictions
            if ii == 1:
                dates_o = x_tr.reset_index()['date']
                # predict training data
                pred_o = model.predict(x_tr[pct_final_subset])
                # convert % values to normal values
                pred_oo = (pred_o * x_tr['close'].values) + x_tr['close'].values
                close_oo = x_tr['close'].values
        # add values to the dfs
        old_preds['Date'] = dates_o
        old_preds['Prediction'] = close_oo
        old_preds['Format'] = 'Price'
        hold_predictions['Date'] = dates
        hold_predictions['Prediction'] = preds
        hold_predictions['Format'] = 'Prediction'

        # concat both dfs and add to the dict of predictions
        predictions_[f'{i}'] = pd.concat([old_preds, hold_predictions], axis=0).set_index('Date')
        models[f'{i}'] = model

    return predictions_, models


def real_time_(key, companies):
    import datetime
    stock_price = pd.DataFrame()
    for i in companies:
        try:
            url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{i}?apikey={key}'
            r = requests.get(url)
            data = r.json()
            d = pd.DataFrame.from_dict(data['historical'])
            d['ticker'] = i
            stock_price = pd.concat([stock_price, d.dropna()], axis=0)
            stock_price.drop(['label'], axis=1)
            stock_price['date'] = pd.to_datetime(stock_price['date'])
            stock_price.set_index(['ticker', 'date'])
        except:
            pass

    stock_price['date'] = pd.to_datetime(stock_price['date'])
    stock_price = stock_price.sort_values(['ticker', 'date'])
    stock_price = stock_price.set_index(['ticker', 'date'])
    stock_price.index.names = 'ticker', 'date'

    return (stock_price)


def gen_commodities():
    import time
    start = time.time()

    key = '54594d7278e0fa3c0831a72c60e04b8d'
    df = real_time_(key, ['CLUSD', 'ZGUSD', 'NGUSD', 'KWUSX']).reset_index()

    for i, ii in zip(['CLUSD', 'ZGUSD', 'NGUSD', 'KWUSX'], ['oil', 'gold', 'gas', 'wheat']):
        df.loc[df['ticker'] == i, 'ticker'] = ii
    #     df = df['close'].unstack('ticker')
    #     df = df.rename(columns = {'CLUSD':'oil','ZGUSD':'gold','NGUSD':'gas','KWUSX':'wheat'})

    end = time.time()
    print(' ')
    print(f'Data generation lasted: {end - start} seconds')
    return (df.set_index('ticker', 'date'))


commodities = gen_commodities()

import pendulum

timezones = {
    f'Lisbon, Portugal': datetime.datetime.strftime(datetime.datetime.now(pendulum.timezone("Etc/GMT-1")), "%d %b %y | %H:%M:%S"),
    f'Zurich, Switzerland': datetime.datetime.strftime(datetime.datetime.now(pendulum.timezone("Etc/GMT-2")),
                                               "%d %b %y | %H:%M:%S"),
    f'London, England': datetime.datetime.strftime(datetime.datetime.now(pendulum.timezone("Etc/GMT-1")), "%d %b %y | %H:%M:%S"),
    f'Tokyo, Japan': datetime.datetime.strftime(datetime.datetime.now(pendulum.timezone("Etc/GMT-9")), "%d %b %y | %H:%M:%S"),
    f'New York, USA': datetime.datetime.strftime(datetime.datetime.now(pendulum.timezone("US/Eastern")), "%d %b %y | %H:%M:%S")}
keys = ['Lisbon, Portugal', 'Zurich, Switzerland', 'London, England', 'Tokyo, Japan','New York, USA']


def real_time_forex(key, companies):
    stock_price = pd.DataFrame()
    for i in companies:
        try:
            url = f'https://financialmodelingprep.com/api/v3/historical-price-full/USD{i}?apikey={key}'
            r = requests.get(url)
            data = r.json()
            d = pd.DataFrame.from_dict(data['historical'])
            d['ticker'] = i
            stock_price = stock_price.append(d)

        except:
            pass
    stock_price.drop(['label'], axis=1)
    stock_price = stock_price.set_index(['ticker', 'date'])
    return (stock_price)


def gen_fx():
    import time
    start = time.time()

    key = '54594d7278e0fa3c0831a72c60e04b8d'
    fx = ['EUR', 'GBP', 'CHF', 'JPY']
    df = real_time_forex(key, fx)['close'].unstack('ticker')

    end = time.time()
    print(' ')
    print(f'Data generation lasted: {end - start} seconds')
    return (df)


fx = gen_fx()
fx['USD'] = 1

###################### APP Structure
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='main')
])
app.config.suppress_callback_exceptions = True

#### Main page
mainpage = html.Div([
    html.Div([  ### header box
        html.Div([
            html.Img(src='assets/datalin.png', width=250, style={'float': 'right'}),
            html.H1('Cryptocurrency Analysis Dashboard - Overview'),
            html.P('Business Case 5: Group E',
                   style={'padding-left': '0px', 'margin-top': '25px'}),
            html.Img(src='assets/nova_ims.png', width=85, style={'float': 'right','margin-right':'80px','margin-top':'30px'}),
            dbc.Button(
                "Detailed Analysis",
                id="button1",
                n_clicks=0,
                outline=True, color="secondary",
                className="me-1"
                , href='/secpage',style={'margin-bottom':'10px'}
            ), ], id='Title Box'),
        html.Br(),
        html.P('Choose the currency: ', style={'display': 'inline-block','margin-right':'10px'}),
        dcc.RadioItems(fx.columns, id='currency', value=fx.columns[-1], inline=True,
                       style={'padding-right': '5px', 'display': 'inline-block', 'padding-bottom':'40px'}, className='radio', ),
    ], className='header'),
    html.Br(),
    html.Div([  ## page content container

        html.Div([
            html.Div([
                #### Clock & Date
                html.Div([
                    html.H4('Current Date and Time'),
                    html.Br(),
                    html.P(keys[0], className='title'),
                    html.P(timezones[keys[0]], className='clock'),
                    html.P(keys[1], className='title'),
                    html.P(timezones[keys[1]], className='clock'),
                    html.P(keys[2], className='title'),
                    html.P(timezones[keys[2]], className='clock'),
                    html.P(keys[3], className='title'),
                    html.P(timezones[keys[3]], className='clock'),
                    html.P(keys[4], className='title'),
                    html.P(timezones[keys[4]], className='clock'),
                ], style={'text-align': 'center'}),
                html.Br(),
                html.Br(),
                html.H4('Latest values registered for the coins', style={'padding-top':'20px','text-align': 'center','border-top': '4px dashed #dbd9d9'}),
                html.Br(),
                html.Br(),
                html.Div([], id='table'),
            ]),
        ], className='boxes', style={'width': '40%','margin-bottom':'20px'}),

        html.Div([
            #### Heatmap + Bar Plot
            html.Div([
                html.H4('Monthly returns in the last 3 years', style={'text-align': 'center'}),
                dcc.Graph(id='returns_heatmap'),
            ]),
            html.Br(),
            html.Br(),
            html.Div([
                html.H4('Volume Analysis', style={'text-align': 'center'}),
                dcc.Graph(id='plot'),
            ],style={'border-top': '4px dashed #dbd9d9','padding-top':'20px'}),
        ], className='boxes', style={'width': '40%','margin-bottom':'20px'}),

        html.Div([
            html.H4('Commodities Overview', style={'text-align': 'center'}),
            html.Br(),
            html.P("Select one commodity from the dropdown to further analyze"),
            dcc.Dropdown(options=commodities.index.unique(),
                         placeholder="Select a commodity to further analyze",
                         searchable=True,
                         id='com_drop',
                         value='oil',
                         style={'border-color': 'gray', 'position': 'relative'}
                         ),
            dcc.Graph(id='fig_',style={'border-bottom': '4px dashed #dbd9d9'}),
            html.Br(),
            html.H4('Commodities Comparison', style={'text-align': 'center'}),
            html.Br(),
            html.P("Select 2 commodities to compare"),
            dcc.Dropdown(options=commodities.index.unique(),
                         placeholder="Select 2 commodities to further analyze",
                         searchable=True,
                         id='com_drop2',
                         multi=True,
                         value=['oil', 'gas'],
                         style={'border-color': 'gray', 'position': 'relative'}
                         ),
            dcc.Graph(id='subfig_')
        ], className='boxes', style={'width': '40%', 'margin-right': '20px','margin-bottom':'20px'}),

    ], style={'display': 'flex'}),
]),

secpage = html.Div([  # main container
    #### Header container
    html.Div([
        html.Div([
            html.Img(src='assets/datalin.png', width=250, style={'float': 'right'}),
            html.H1('Cryptocurrency Analysis Dashboard'),
            html.P('Business Case 5: Group E',
                   style={'padding-left': '0px', 'margin-top': '25px'}),
        ], id='Title Box'),
        html.Img(src='assets/nova_ims.png', width=85, style={'float': 'right','margin-right':'80px','margin-top':'30px'}),
        dbc.Button(
            "Back to Overview",
            id="button",
            n_clicks=0,
            outline=True, color="secondary",
            className="me-1"
            , href='/',
            style={'margin-bottom': '20px'}
        ),
        dcc.Dropdown(options=crypto_options,
                     placeholder="Select a crypto to further analyze",
                     searchable=True,
                     id='crypto_drop',
                     value='ADA',
                     style={'border-color': 'gray', 'position': 'relative','margin-bottom':'10px','width':'50%'}
                     ),
        html.P('Choose the currency: ', style={'display': 'inline-block','margin-right':'10px'}),
        dcc.RadioItems(fx.columns, id='currencysec', value=fx.columns[-1], inline=True,
                       style={'margin-right': '5px', 'display': 'inline-block'}, className='radio'),
    ], className='header'),
    html.Br(),
    #### First row
    html.Div(
        [
            #### Information box about the coin + cards
            html.Div(
                [
                    html.Div(id='img'),
                    html.H4(id='title'),
                    html.Br(),
                    html.Br(),
                    html.P(id='text', style={'margin-bottom': '50px'}),
                    html.Div([html.H4('Important Values',style={'border-top': '4px dashed #dbd9d9','padding-top':'20px'}),
                              html.Br(),
                              html.P('% of change from the last price available', className='title1'),
                              html.P(id='pct_change', className='box'),
                              html.P('% of change from a month ago', className='title1'),
                              html.P(id='pct_changem', className='box'),
                              html.P('Value of the coin today', className='title1'),
                              html.P(id='valuetd', className='box', style={'text-align': 'center'}),
                              html.P('Value of the coin a month ago', className='title1'),
                              html.P(id='valuem', className='box'), ], style={'text-align': 'center'}),
                    html.Br(),
                    html.Br(),
                    html.H4('Sources',style={'text-align':'center','border-top': '4px dashed #dbd9d9','padding-top':'20px'}),
                    html.Br(),
                    html.Div([
                        html.A(href='https://site.financialmodelingprep.com/',children=[
                        html.Img(src='assets/source1.png', width=300)])],style={'display':'flex','justify-content':'center'}),
                    html.Br(),
                    html.Br(),
                    html.Div([
                        html.A(href='https://finance.yahoo.com/', children=[
                            html.Img(src='assets/source2.png', width=200)])],
                        style={'display': 'flex', 'justify-content': 'center'}),
                    html.Br(),
                    html.Br(),
                    html.Div([
                        html.A(href='https://fireart.studio/blog/15-best-crypto-web-design-inpirations/', children=[
                            html.Img(src='assets/source3.png', width=300)])],
                        style={'display': 'flex', 'justify-content': 'center'}),
                ], className='boxes', style={'width': '30%', 'margin-right': '20px'}),
                    html.Br(),
            #### Time Series graph
            html.Div(
                [

                    html.H4('Time Series with historical data', style={'text-align': 'center'}),
                    dcc.DatePickerRange(
                        id='datepick',
                        end_date=(date.today() - datetime.timedelta(1)),
                        start_date=date(2017, 1, 1),
                        min_date_allowed=date(2014, 9, 17),
                        max_date_allowed=date.today(),
                        initial_visible_month=date.today(),
                        display_format='MMM Do, YY',
                        style={'float': 'right'}
                    ),
                    dcc.Graph(id='timeseries_', style={'margin-top': '30px', 'align': 'center'}),
                    html.Br(),
                    html.H4('Time Series with predictions', style={'text-align': 'center','border-top': '4px dashed #dbd9d9','padding-top':'20px'}),
                    html.H5('Daily', style={'text-align': 'center'}),
                    html.P('How many days to predict: ',
                           style={'display': 'inline-block', 'margin-right': '10px', 'margin-left': '50px'}),
                    dcc.Input(id='time_input', type='number', min=1, max=90, step=1, value=7,
                              style={'display': 'inline-block'}),
                    dcc.Graph(id='pred_', style={'margin-top': '30px', 'align': 'center'}),
                    html.H5('Hourly', style={'text-align': 'center'}),
                    html.P('How many hours ahead to predict: ',
                           style={'display': 'inline-block', 'margin-right': '10px', 'margin-left': '50px'}),
                    dcc.Input(id='day_input', type='number', min=1, max=24, step=1, value=8,
                              style={'display': 'inline-block'}),
                    dcc.Graph(id='pred_d', style={'margin-top': '30px', 'align': 'center'}),
                ], className='boxes', style={'width': '90%', 'margin-right': '20px'}),
        ], style={'display': 'flex'}),
html.Br(),
    html.Br(),
    html.H1('Indicator Analysis',
            style={'margin-left': '20px', 'text-align': 'center', 'text-decoration': 'underline'}),
    html.Br(),
    #### Indicators
    html.Div([
        html.Div(
            [
                html.H5('Momentum'),
                dcc.Graph(id='momentum_', style={'margin-bottom': '10px', 'align': 'center'}),
            ], className='boxes', style={'width': '35%', 'margin-bottom': '20px'}),
        html.Div(
            [
                html.H5('Overlap'),
                dcc.Graph(id='overlap_', style={'margin-bottom': '10px', 'align': 'center'}),
            ], className='boxes', style={'width': '35%', 'margin-bottom': '20px'}),
        html.Div(
            [
                html.H5('Volatility'),
                dcc.Graph(id='volatility_', style={'margin-bottom': '10px', 'align': 'center'}),
            ], className='boxes', style={'width': '35%', 'margin-right': '20px', 'margin-bottom': '20px'}),

    ], style={'display': 'flex'}),

], id='main')


# callback for the overview page
@callback(Output('fig_', 'figure'),
          Output('returns_heatmap', 'figure'),
          Output('plot', 'figure'),
          Output('table', 'children'),
          Input('currency', 'value'),
          Input('com_drop', 'value'))
def candlestick(coin, ticker):
    # converts dolar OHLC values to a specified coin
    fx1 = fx.copy()
    fx1.index = pd.to_datetime(fx.index)
    convert_df = pd.merge(commodities.reset_index(), fx1.reset_index(), on='date', how='outer').sort_values(
        ['ticker', 'date']).set_index(['ticker', 'date'])
    for i in fx1.columns:
        convert_df[f'{i}'] = convert_df[f'{i}'].fillna(method='ffill')

    for i in ['open', 'close', 'high', 'low']:
        convert_df[f'{i}'] = convert_df[f'{i}'] * convert_df[f'{coin}']

    convert_df2 = pd.merge(df_daily.reset_index(), fx1.reset_index(), on='date', how='outer').sort_values(
        ['ticker', 'date']).set_index(['ticker', 'date'])
    for i in fx1.columns:
        convert_df2[f'{i}'] = convert_df2[f'{i}'].fillna(method='ffill')

    for i in ['open', 'close', 'high', 'low']:
        convert_df2[f'{i}'] = convert_df2[f'{i}'] * convert_df2[f'{coin}']

    # CANDLESTICK
    import plotly.graph_objects as go
    convert_df1 = convert_df.loc[convert_df.index.get_level_values(0) == ticker]
    convert_df1 = convert_df1.reset_index()
    fig = go.Figure(data=[go.Candlestick(x=convert_df1['date'],
                                         open=convert_df1['open'],
                                         high=convert_df1['high'],
                                         low=convert_df1['low'],
                                         close=convert_df1['close'])])
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', },
                      title=f'{ticker.upper()} Candlestick', yaxis_title=f'{(ticker).upper()} / USD',
                      xaxis_title='Date')

    fig.update_yaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    fig.update_xaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB',
                     rangeslider_visible=True)

    ##HEATMAP
    close = convert_df2['close'].unstack('ticker').copy()
    close = close.loc[(close.index >= '2019-01-01')]
    returns = (np.log(close / close.shift()))
    m_returns = returns.resample('m').mean().reset_index()
    m_returns['year'] = (m_returns['date'].dt.year)
    m_returns['month_name'] = m_returns['date'].dt.month_name()
    m_returns['year'] = m_returns['year'].astype(str)

    m_returns = m_returns.groupby(['year', 'month_name']).mean().mean(axis=1).unstack()
    m_returns = sd.Sort_Dataframeby_Month(m_returns.T.reset_index(), 'month_name').set_index('month_name').T
    returns_heatmap = px.imshow(m_returns, title='Average returns per month',
                                labels=dict(x="Months", y="Years", color="Average Returns"),
                                color_continuous_scale='rdylgn')
    returns_heatmap.update_coloraxes(cmin=-0.04, cmid=0, cmax=0.04)
    returns_heatmap.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', })

    # VOLUME BARS
    dates = pd.to_datetime(date.today() - datetime.timedelta(weeks=10))
    close_df = df_daily['volume'].unstack('ticker').sort_index().dropna()
    close_df = close_df.loc[close_df.index > dates]
    close_df = close_df.diff().mean(axis=1).cumsum()
    close_df = close_df.sort_index().reset_index().rename(columns={0: 'Cumulative Volume', 'date': 'Date'})

    return_cm = close_df
    return_cm['Color'] = 'green'
    return_cm.loc[return_cm['Cumulative Volume'] < 0, 'Color'] = 'red'

    date1 = return_cm['Date'].min()
    plot = px.bar(return_cm, y='Cumulative Volume', x='Date', title=f'10 weeks cumulative volume',
                  color='Color',
                  color_discrete_sequence=return_cm.Color.unique())
    plot.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', })

    # TABLE
    convert_df2.dropna(inplace=True)
    a = pd.DataFrame(convert_df2.loc[:, 'close']).reset_index()
    a['date'] = a['date'].astype('string')
    b = a[pd.to_datetime(a['date']).dt.date >= (date.today() - datetime.timedelta(4))].set_index('ticker').pivot(
        columns='date').round(4)
    b.columns = b.columns.get_level_values(1)
    # b.columns = [str((datetime.datetime.strptime(b.columns[0], '%Y-%M-%d') - datetime.timedelta(1)).date()),
    #              str((datetime.datetime.strptime(b.columns[1], '%Y-%M-%d') - datetime.timedelta(1)).date()),
    #              str((datetime.datetime.strptime(b.columns[2], '%Y-%M-%d') - datetime.timedelta(1)).date()),
    #              str((datetime.datetime.strptime(b.columns[3], '%Y-%M-%d') - datetime.timedelta(1)).date())]

    return fig, returns_heatmap, plot, dash_table.DataTable(data=b.reset_index().to_dict('records'),
                                                            columns=[{"name": i, "id": i} for i in
                                                                     b.reset_index().columns],
                                                            style_cell={'textAlign': 'center',
                                                                        'font-family': 'Arial, Helvetica, sans-serif'},
                                                            style_data_conditional=[
                                                                {'if': {
                                                                    'column_id': b.columns[-1],
                                                                    'filter_query': '{' + b.columns[-1] + '} > {' + b.columns[-2] + '}'},
                                                                    'font-weight': 'bold',
                                                                    'color': '#32CD32'},
                                                                {'if': {
                                                                    'column_id': b.columns[-1],
                                                                    'filter_query': '{' + b.columns[-1]+ '} < {' + b.columns[-2] + '}'},
                                                                    'font-weight': 'bold',
                                                                    'color': 'red'},
                                                                {'if': {
                                                                    'column_id': b.columns[-1],
                                                                    'filter_query': '{' + b.columns[-1] + '} = {' +
                                                                                    b.columns[-2] + '}'},
                                                                    'color': '#EC9706',
                                                                    'font-weight': 'bold',}

                                                            ],
                                                            style_as_list_view=True,
                                                            style_header={
                                                                'backgroundColor': '#EBEBEB',
                                                                'fontWeight': 'bold'
                                                            },
                                                            )


@callback(
    Output('subfig_', 'figure'),
    Input('com_drop2', 'value'),
    Input('currency', 'value'))
def gen_commodities_plot(drop, coin, date='2019-01-01'):
    # converts dolar OHLC values to a specified coin
    fx1 = fx.copy()
    fx1.index = pd.to_datetime(fx.index)
    convert_df = pd.merge(commodities.reset_index(), fx1.reset_index(), on='date', how='outer').sort_values(
        ['ticker', 'date']).set_index(['ticker', 'date'])
    for i in fx1.columns:
        convert_df[f'{i}'] = convert_df[f'{i}'].fillna(method='ffill')

    for i in ['open', 'close', 'high', 'low']:
        convert_df[f'{i}'] = convert_df[f'{i}'] * convert_df[f'{coin}']

    ticker = drop[0]
    ticker2 = drop[1]
    df = convert_df.reset_index().loc[convert_df.index.get_level_values(0) == ticker, ['date', 'close']].set_index(
        'date').rename(columns={'close': f'{ticker}'})
    df = df.loc[df.index >= date]

    df1 = convert_df.reset_index().loc[convert_df.index.get_level_values(0) == ticker2, ['date', 'close']].set_index(
        'date').rename(columns={'close': f'{ticker2}'})
    df1 = df1.loc[df1.index >= date]

    from plotly.subplots import make_subplots
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = px.line(df[f'{ticker}'], y=f'{ticker}', title=f'{str(ticker).upper()} Price USD', labels={'y': f'{ticker}'})
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', }, showlegend=True)
    fig.update_traces(name=ticker, showlegend=True)
    fig2 = px.line(df1[f'{ticker2}'], y=f'{ticker2}', title=f'{str(ticker2).upper()} Price USD', labels=dict(y=ticker2))
    fig2.update_layout(showlegend=True)
    fig2.update_traces(yaxis="y2", name=ticker2, showlegend=True)

    subfig.add_traces(fig.data + fig2.data)
    subfig.layout.xaxis.title = "Date"
    subfig.layout.yaxis.title = f"{ticker}"

    subfig.layout.yaxis2.title = f"{ticker2}"
    # recoloring is necessary otherwise lines from fig und fig2 would share each color
    # e.g. Linear-, Log- = blue; Linear+, Log+ = red... we don't want this
    subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))

    subfig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', },
                         title=f'{ticker.upper()} and {ticker2.upper()} price evolution', showlegend=True)
    subfig.update_yaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    subfig.update_xaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB',
                        rangeslider_visible=True)
    return subfig


# callback for the secpage
@callback(Output('momentum_', 'figure'),
          Output('overlap_', 'figure'),
          Output('volatility_', 'figure'),
          Output('timeseries_', 'figure'),
          Output('title', 'children'),
          Output('img', 'children'),
          Output('text', 'children'),
          Output('pred_', 'figure'),
          Output('pred_d', 'figure'),
          Output('pct_change', 'children'),
          Output('pct_changem', 'children'),
          Output('valuetd', 'children'),
          Output('valuem', 'children'),
          Input('crypto_drop', 'value'),
          Input('datepick', 'start_date'),
          Input('datepick', 'end_date'),
          Input('time_input', 'value'),
          Input('day_input', 'value'),
          Input('currencysec', 'value'))
def update_info(crypto, start_date, end_date, timestamp, daystamp, coin):
    # converts dolar OHLC values to a specified coin
    fx1 = fx.copy()
    fx1.index = pd.to_datetime(fx.index)
    convert_df = pd.merge(df_daily.reset_index(), fx1.reset_index(), on='date', how='outer').sort_values(
        ['ticker', 'date']).set_index(['ticker', 'date'])
    for i in fx1.columns:
        convert_df[f'{i}'] = convert_df[f'{i}'].fillna(method='ffill')

    for i in ['open', 'close', 'high', 'low']:
        convert_df[f'{i}'] = convert_df[f'{i}'] * convert_df[f'{coin}']

    df = df_daily.loc[[crypto], ['MOM_7', 'ROC_7']]
    momentum = px.scatter(df, x="MOM_7", y="ROC_7", marginal_y="violin", trendline="lowess", template="simple_white", color_discrete_sequence=['#268099'])

    df2 = df_daily.loc[[crypto], ['EMA_7', 'EMA_20']]
    df2['EMA_20'] = df2['EMA_20'] - df2['EMA_7']
    df2.index = df2.index.get_level_values(1)
    overlap = px.area(df2[df2.index > '2022-01-01'], markers=True, color_discrete_sequence=['#268099','#8E4585'])

    df3 = df_daily.loc[[crypto], ['ATR_7', 'STD_7', 'RSI_7']].dropna()
    volatility = px.scatter(df3, x="ATR_7", y="STD_7", size="RSI_7", color='RSI_7', log_x=True, log_y=True, size_max=20
                            , color_continuous_scale='tealgrn')

    df4 = convert_df.loc[[crypto], ['close']].dropna()
    df4 = df4.loc[df4.index.get_level_values(0) == crypto].droplevel('ticker')[['close']]
    timeseries = px.line(df4[(df4.index <= end_date) & (df4.index >= start_date)])
    timeseries.update_traces(line_color=info['Color'][info.index == crypto].item())

    title = info['extend'][info.index == crypto]
    img = html.Img(src=f"assets/{crypto}.png", width=50, style={'float': 'right'})
    text = info['Text'][info.index == crypto]

    # removing background
    momentum.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    overlap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    volatility.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    timeseries.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # changing color of gridlines
    momentum.update_xaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    momentum.update_yaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    overlap.update_xaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    overlap.update_yaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    volatility.update_xaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    volatility.update_yaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    timeseries.update_xaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    timeseries.update_yaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')

    data = convert_df.loc[convert_df.index.get_level_values(0) == crypto]
    data = data.loc[data.index.get_level_values(1) >= '2022-05-01']

    a, b = make_predictions(data, timestamp)

    pred = px.line(a[crypto], y=a[crypto]['Prediction'], color=a[crypto]['Format'],
                   title=f'{crypto}, {timestamp - 1} days  price prediction', line_group=a[crypto]['Format']
                   , color_discrete_sequence=[info['Color'][info.index == crypto].item(), '#32CD32'])
    pred.update_yaxes(showgrid=True)

    pred.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', },
                       title=f'{timestamp - 1} days, {crypto.upper()} Predictions',
                       yaxis_title=f'{crypto.upper()} / USD',
                       xaxis_title='Date')
    pred.update_yaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    pred.update_xaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB',
                      rangeslider_visible=True)

    # hourly predictions
    hour_df = gen_hour_data().dropna()
    hour_df.index.set_levels(hour_df.index.levels[0].str.replace('LUNA', 'LUNA1'), level=0, inplace=True)
    data_h = hour_df.loc[hour_df.index.get_level_values(0) == crypto].copy()
    data_h = data_h.iloc[-24:]

    a, b = make_predictions_hour(data_h, daystamp)

    pred_d = px.line(a[f'{crypto}'], y=a[f'{crypto}']['Prediction'], color=a[f'{crypto}']['Format'],
                     title=f'{crypto}, {daystamp - 1} hours  price prediction', line_group=a[f'{crypto}']['Format']
                     , color_discrete_sequence=[info['Color'][info.index == crypto].item(), '#32CD32'])
    pred_d.update_yaxes(showgrid=True)

    pred_d.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', },
                         title=f'{daystamp - 1} hours, {crypto.upper()} Predictions',
                         yaxis_title=f'{(crypto).upper()} / USD',
                         xaxis_title='Date')
    pred_d.update_yaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB')
    pred_d.update_xaxes(showgrid=True, showline=True, gridwidth=1, gridcolor='#EBEBEB', linecolor='#EBEBEB',
                        rangeslider_visible=True)

    aux = convert_df.loc[convert_df.index.get_level_values(0) == crypto]['close']
    pct_change = str(np.round(((aux[aux.index[-1]] - aux[aux.index[-2]]) / aux[aux.index[-2]]) * 100, 4))
    pct_changem = str(np.round(((aux[aux.index.get_level_values(1) == str(
        date.today())].item() - aux[aux.index.get_level_values(1) == str(
        date.today() - datetime.timedelta(30))].item()) / (aux[aux.index.get_level_values(1) == str(
        date.today() - datetime.timedelta(30))].item()) * 100), 4))
    valuetd = str(np.round(aux[aux.index.get_level_values(1) == str(date.today())].item(), 4))
    valuem = str(np.round(aux[aux.index.get_level_values(1) == str(date.today() - datetime.timedelta(30))].item(), 4))
    return momentum, overlap, volatility, timeseries, title, img, text, pred, pred_d, pct_change, pct_changem, valuetd,valuem


@callback(Output('main', 'children'),
          Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return mainpage
    elif pathname == '/secpage':
        return secpage


if __name__ == '__main__':
    app.run_server(debug=True)

# https://dash.plotly.com/urls
