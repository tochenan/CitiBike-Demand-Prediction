import pandas as pd
import numpy as np
path = 'data/'
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    all_files = glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    return pd.concat(df_from_each_file, ignore_index=True).sort_values(by='started_at')

def preprocess(df):

    df['started_at'] = pd.to_datetime(df['started_at'], format='ISO8601')
    df['ended_at'] = pd.to_datetime(df['ended_at'], format='ISO8601')
    df['month_started'] = df['started_at'].dt.month
    df['day_started'] = df['started_at'].dt.day
    df['weekday_started'] = df['started_at'].dt.weekday
    df['hour_started'] = df['started_at'].dt.hour
    df['minute_started'] = df['started_at'].dt.minute
    df['second_started'] = df['started_at'].dt.second

    return df


def filter_top_station(df, number = 10):
    top_stations = df['start_station_name'].value_counts().head(number).index.tolist()
    return df[df['start_station_name'].isin(top_stations)]

def analyse_data_statistics(df):
    def popular_start_station(df):
        return df.groupby('month_started')['start_station_name'].value_counts().reset_index(name='count')
    
    def popular_end_station(df):
        return df.groupby('month_started')['end_station_name'].value_counts().reset_index(name='count')
      
    def duration(df):
        df['started_at'] = pd.to_datetime(df['started_at'])
        df['ended_at'] = pd.to_datetime(df['ended_at'])
        df['duration'] = df['ended_at'] - df['started_at']
        df['duration'] = df['duration'].dt.total_seconds()

        month_duration = df.groupby('month_started')['duration'].sum().reset_index(name='sum_duration')
        day_duration = df.groupby('weekday_started')['duration'].sum().reset_index(name='sum_duration').sort_values(by='weekday_started')
       
        return month_duration, day_duration
    
    def seasonality_count(df):
        month_count = df.groupby('month_started').size().reset_index(name='count')
        day_count = df.groupby('weekday_started').size().reset_index(name='count').sort_values(by='weekday_started')
        hour_count = df.groupby('hour_started').size().reset_index(name='count')
        return month_count, day_count, hour_count
    
    
    def plotting(df):

        sns.set(style="white")

        # Most popular start station
        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        sns.barplot(x='start_station_name', y='count', hue = 'month_started', data=popular_start_station(df))
        plt.title('Most popular start station')
        plt.xticks(rotation=90)
        sns.despine()
        plt.show()

        # Duration
        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        sns.barplot(x='month_started', y='sum_duration', data=duration(df)[0])
        plt.title('Duration')
        plt.xticks(rotation=90)
        sns.despine()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        sns.barplot(x='weekday_started', y='sum_duration', data=duration(df)[1])
        plt.title('Duration')
        plt.xticks(rotation=90)
        sns.despine()
        plt.show()


        # Seasonality
        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        sns.barplot(x='month_started', y='count', data=seasonality_count(df)[0])
        plt.title('Seasonality')
        plt.xticks(rotation=90)
        sns.despine()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        sns.barplot(x='weekday_started', y='count', data=seasonality_count(df)[1])
        plt.title('Seasonality')
        plt.xticks(rotation=90)
        sns.despine()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        sns.barplot(x='hour_started', y='count', data=seasonality_count(df)[2])
        plt.title('Seasonality')
        plt.xticks(rotation=90)
        sns.despine()
        plt.show()

    plotting(df)
    

if __name__ == '__main__':
    df = load_data(path)
    df = preprocess(df)
    df = filter_top_station(df)
    analyse_data_statistics(df)
    