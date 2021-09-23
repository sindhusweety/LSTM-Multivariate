import pandas as pd

class Preprocessing(object):
    def _read_sim_utils(self):
        df = pd.read_csv('/home/sindhukumari/PycharmProjects/LSTM Model/sim_utilization-latest.csv')
        df = df[0:10000]
        fRow = df.columns
        print(fRow)
        df.columns = ['timestamp', 'device_id', 'customer', 'pool', 'ctdusage', 'A', 'B', 'C', 'D']
        df = df[['timestamp', 'device_id', 'customer', 'pool', 'ctdusage']]
        df = df.set_index('timestamp')
        #Downsampling of Data from minutes to Days
        '''
        make the data simpler by downsampling them from the frequency of minutes to days.
        tdf = df.set_index('timestamp')
        daily_df = tdf.resample('D', on='timestamp').mean()
        print(daily_df.head())
        '''
        '''
        df['year'] = [int(row) for row in pd.DatetimeIndex(df['timestamp']).year]
        df['month'] = [int(row) for row in pd.DatetimeIndex(df['timestamp']).month]
        df['day'] = [int(row) for row in pd.DatetimeIndex(df['timestamp']).day]
        df['hours'] = [int(row) for row in pd.DatetimeIndex(df['timestamp']).hour]
        df['minutes'] = [int(row) for row in pd.DatetimeIndex(df['timestamp']).minute]
        df['seconds'] = [int(row) for row in pd.DatetimeIndex(df['timestamp']).second]
        df['microseconds'] = [int(row) for row in pd.DatetimeIndex(df['timestamp']).microsecond]
        df = df[['year', 'month', 'day', 'hours', 'minutes', 'seconds', 'microseconds', 'device_id', 'customer', 'pool', 'ctdusage']]'''
        return df

