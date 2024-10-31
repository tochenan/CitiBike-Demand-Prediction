from glob import glob
import pandas as pd
import os
PATH = 'data/'

def load_data(path = PATH):
    # Load all csv files in the folder
    all_files = glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    return pd.concat(df_from_each_file, ignore_index=True).sort_values(by='started_at')

def preprocess(df):
    # Preprocess the data, extract the date and time information

    df['started_at'] = pd.to_datetime(df['started_at'], format='ISO8601')

    # As may 2024 data is not available, we will filter the data before May 1, 2024
    cutoff_date = pd.Timestamp('2024-05-01')
    df = df[df['started_at'] < cutoff_date]

    df['ended_at'] = pd.to_datetime(df['ended_at'], format='ISO8601')
    df['year_started'] = df['started_at'].dt.year
    df['month_started'] = df['started_at'].dt.month
    df['day_started'] = df['started_at'].dt.day
    df['weekday_started'] = df['started_at'].dt.weekday
    df['hour_started'] = df['started_at'].dt.hour
    df['minute_started'] = df['started_at'].dt.minute
    df['second_started'] = df['started_at'].dt.second

    df['year_ended'] = df['ended_at'].dt.year
    df['month_ended'] = df['ended_at'].dt.month
    df['day_ended'] = df['ended_at'].dt.day
    df['weekday_ended'] = df['ended_at'].dt.weekday
    df['hour_ended'] = df['ended_at'].dt.hour
    df['minute_ended'] = df['ended_at'].dt.minute
    df['second_ended'] = df['ended_at'].dt.second


    return df