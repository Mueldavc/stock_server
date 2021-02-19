import MetaTrader5 as mt5
from datetime import timedelta
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator, STCIndicator
from ta.momentum import RSIIndicator, stochrsi_k, ultimate_oscillator
from ta.volume import volume_price_trend, VolumeWeightedAveragePrice


class StockData(object):
    def __init__(self, stock, date, window_days, timeframe):
        self.stock = stock
        self.window_days = window_days
        self.date = date
        self.timeframe = timeframe
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self._Y = None
        self.val_y = None
        self._download()

    def _download(self):
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        ctrl = 0
        date_ini = self.date - timedelta(days=1)
        date_fim = self.date
        _ = []
        while True:
            tick_frame = mt5.copy_rates_range(self.stock,
                                              self.timeframe,
                                              date_ini,
                                              date_fim)
            tick_frame = pd.DataFrame(tick_frame)
            date_ini -= timedelta(days=1)
            date_fim -= timedelta(days=1)
            if not tick_frame.empty:
                _.append(tick_frame)
                ctrl += 1
            if ctrl == self.window_days:
                tick_frame = pd.concat(_)
                break
        tick_frame.time = pd.to_datetime(tick_frame.time, unit='s')
        tick_frame.set_index(tick_frame.time, inplace=True)
        tick_frame.sort_index(inplace=True)
        tick_frame.drop_duplicates(inplace=True)
        close_dif = tick_frame.close - tick_frame.open
        self._Y = close_dif.apply(lambda x: 1 if x > 0 else 0).shift(-1).dropna()
        indicator_bb = BollingerBands(close=tick_frame.close, window=10, window_dev=2)
        indicator_rsi = RSIIndicator(close=tick_frame.close, window=2).rsi()
        indicator_stk = stochrsi_k(close=tick_frame.close, window=10)
        indicator_vol = volume_price_trend(close=tick_frame.close, volume=tick_frame.real_volume)
        indicator_vwap = VolumeWeightedAveragePrice(close=tick_frame.close,
                                                    volume=tick_frame.real_volume,
                                                    high=tick_frame.high,
                                                    low=tick_frame.low)
        indicator_uo = ultimate_oscillator(low=tick_frame.low, close=tick_frame.close, high=tick_frame.high)
        indicator_ema_9 = EMAIndicator(close=tick_frame.close, window=9)
        indicator_ema_21 = EMAIndicator(close=tick_frame.close, window=21)
        indicator_STC = STCIndicator(close=tick_frame.close)

        tick_frame['bb_bbm'] = indicator_bb.bollinger_mavg()
        tick_frame['bb_bbh'] = indicator_bb.bollinger_hband()
        tick_frame['bb_bbl'] = indicator_bb.bollinger_lband()
        tick_frame['rsi'] = indicator_rsi
        tick_frame['vol'] = indicator_vol
        tick_frame['k'] = indicator_stk * 100
        tick_frame['vwap'] = indicator_vwap.vwap
        tick_frame['uo'] = indicator_uo
        tick_frame['ema9'] = indicator_ema_9.ema_indicator()
        tick_frame['ema21'] = indicator_ema_21.ema_indicator()
        tick_frame['STC'] = indicator_STC.stc()

        cols = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume']
        cols_1 = ['bb_bbm', 'bb_bbh', 'bb_bbl', 'rsi', 'vol', 'k', 'vwap', 'uo', 'ema9', 'ema21', 'STC']
        tick_var = tick_frame.loc[:, cols].diff(1) / tick_frame.loc[:, cols].shift(1) * 100
        tick_var.dropna(inplace=True)
        tick_var = tick_var.round(2)
        self.tickframe = pd.concat([tick_var, tick_frame.loc[:, cols_1]], axis=1).dropna()
        self.n_features = int(self.tickframe.shape[1])

    def data_final(self, n_in=1, n_out=1):
        tickframe = self.tickframe
        self.n_obs = int(self.n_features * n_in)
        tickframe = self._series_to_supervised(tickframe, n_in, n_out)
        periods = tickframe.index.to_period('d').drop_duplicates().sort_values(ascending=False).values
        train_x = tickframe.loc[(~tickframe.index.to_period('d').isin(periods[0:2])) &
                                tickframe.index.to_period('d').isin(periods), :]
        self.train_y = self._Y[self._Y.index.isin(train_x.index)]
        self.train_x = train_x.values[:, :self.n_obs].reshape(-1, self.n_features, n_in)
        val_x = tickframe.loc[tickframe.index.to_period('d').isin(periods[0:2]), :]
        self.val_y = self._Y[self._Y.index.isin(val_x.index)]
        self.val_x = val_x.values[:, :self.n_obs].reshape(-1, self.n_features, n_in)
        a = 1

    def _series_to_supervised(self, tickframe, n_in, n_out):
        data = self.scaler_x.fit_transform(tickframe)
        pkl.dump(self.scaler_x, open(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\scaler_X.sav',
                                     'wb'))
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.set_index(self.tickframe.index, inplace=True)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg

    def __del__(self):
        mt5.shutdown()


class StockData_robo(object):
    def __init__(self, stock, date, timeframe):
        self.stock = stock
        self.date = date
        self.timeframe = timeframe
        self.scaler_x = pkl.load(open(r'C:\Users\mueld\Documents\Python_Projects\Stock_1\scaler_X.sav', 'rb'))
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self._Y = None
        self.val_y = None
        self._download()

    def _download(self):
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()
        tick_frame = mt5.copy_rates_from(self.stock, self.timeframe, self.date, 80 + 17)
        tick_frame = pd.DataFrame(tick_frame)
        tick_frame.time = pd.to_datetime(tick_frame.time, unit='s')
        tick_frame.set_index(tick_frame.time, inplace=True)
        tick_frame.sort_index(inplace=True)
        tick_frame.drop_duplicates(inplace=True)
        indicator_bb = BollingerBands(close=tick_frame.close, window=10, window_dev=2)
        indicator_rsi = RSIIndicator(close=tick_frame.close, window=2).rsi()
        indicator_stk = stochrsi_k(close=tick_frame.close, window=10)
        indicator_vol = volume_price_trend(close=tick_frame.close, volume=tick_frame.real_volume)
        indicator_vwap = VolumeWeightedAveragePrice(close=tick_frame.close,
                                                    volume=tick_frame.real_volume,
                                                    high=tick_frame.high,
                                                    low=tick_frame.low)
        indicator_uo = ultimate_oscillator(low=tick_frame.low, close=tick_frame.close, high=tick_frame.high)
        indicator_ema_9 = EMAIndicator(close=tick_frame.close, window=9)
        indicator_ema_21 = EMAIndicator(close=tick_frame.close, window=21)
        indicator_STC = STCIndicator(close=tick_frame.close)

        tick_frame['bb_bbm'] = indicator_bb.bollinger_mavg()
        tick_frame['bb_bbh'] = indicator_bb.bollinger_hband()
        tick_frame['bb_bbl'] = indicator_bb.bollinger_lband()
        tick_frame['rsi'] = indicator_rsi
        tick_frame['vol'] = indicator_vol
        tick_frame['k'] = indicator_stk * 100
        tick_frame['vwap'] = indicator_vwap.vwap
        tick_frame['uo'] = indicator_uo
        tick_frame['ema9'] = indicator_ema_9.ema_indicator()
        tick_frame['ema21'] = indicator_ema_21.ema_indicator()
        tick_frame['STC'] = indicator_STC.stc()

        cols = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume']
        cols_1 = ['bb_bbm', 'bb_bbh', 'bb_bbl', 'rsi', 'vol', 'k', 'vwap', 'uo', 'ema9', 'ema21', 'STC']
        tick_var = tick_frame.loc[:, cols].diff(1) / tick_frame.loc[:, cols].shift(1) * 100
        tick_var.dropna(inplace=True)
        tick_var = tick_var.round(2)
        self.tickframe = pd.concat([tick_var, tick_frame.loc[:, cols_1]], axis=1).dropna()
        self.n_features = int(self.tickframe.shape[1])

    def data_final(self, n_in=1, n_out=1):
        tickframe = self.tickframe
        self.n_obs = int(self.n_features * n_in)
        tickframe = self._series_to_supervised(tickframe, n_in, n_out)
        self.train_x = tickframe.values[-1, :self.n_obs].reshape(-1, self.n_features, n_in)

    def _series_to_supervised(self, tickframe, n_in, n_out):
        data = self.scaler_x.transform(tickframe)
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.set_index(self.tickframe.index, inplace=True)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg

    def __del__(self):
        mt5.shutdown()
