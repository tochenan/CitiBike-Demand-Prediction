import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, preprocess

viz_path = 'viz/'


def filter_top_station(df, number = 10):
    # Filter the top stations
    top_stations = df['start_station_name'].value_counts().head(number).index.tolist()
    return df[df['start_station_name'].isin(top_stations)]

def analyse_data_statistics(df):
    # Analyse the seasonality related data and plot the gross statistics statistics
    def popular_start_station(df):
        return df.groupby('month_started')['start_station_name'].value_counts().reset_index(name='count')
    
    def popular_end_station(df):
        return df.groupby('month_started')['end_station_name'].value_counts().reset_index(name='count')
      
    def duration(df):
        df['started_at'] = pd.to_datetime(df['started_at'])
        df['ended_at'] = pd.to_datetime(df['ended_at'])
        df['duration'] = df['ended_at'] - df['started_at']
        df['duration'] = df['duration'].dt.total_seconds()

        month_duration = df.groupby(['year_started','month_started'])['duration'].mean().reset_index(name='average_duration').sort_values(by='month_started')
        day_duration = df.groupby(['year_started','weekday_started'])['duration'].mean().reset_index(name='average_duration').sort_values(by='weekday_started')
       
        return month_duration, day_duration
    
    def seasonality_count(df):
        hour_count = df.groupby(['year_started','month_started','weekday_started','hour_started']).size().reset_index(name='count')

        weekday_count = df.groupby(['year_started','month_started','weekday_started']).size().reset_index(name='count')

        month_count = df.groupby(['year_started', 'month_started']).size().reset_index(name='count')

        return month_count, weekday_count, hour_count
    
    
    def plotting(df):

        sns.set(style="white")


        # Duration
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_twin = ax.twinx()
        plt.tight_layout()
        sns.barplot(x='month_started', y='average_duration', data=duration(df)[0])
        sns.pointplot(x='month_started', y='average_duration', hue = 'year_started', data=duration(df)[0], ax=ax_twin)
        plt.title('Duration of rides per month')
        plt.xticks(rotation=90)
        sns.despine()
        plt.savefig(f'{viz_path}Duration of rides per month.png', dpi=300, transparent=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax_twin = ax.twinx()
        sns.barplot(x='weekday_started', y='average_duration', data=duration(df)[1])
        sns.pointplot(x='weekday_started', y='average_duration', hue = 'year_started', data=duration(df)[1], ax=ax_twin)
        plt.title('Duration of rides per weekday')
        plt.xticks(rotation=90)
        sns.despine()
        plt.savefig(f'{viz_path}Duration of rides per weekday.png', dpi=300, transparent=True)


        # Seasonality
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_twin = ax.twinx()
        sns.barplot(x='month_started', y='count', data=seasonality_count(df)[0])
        sns.pointplot(x='month_started', y='count', hue = 'year_started', data=seasonality_count(df)[0], ax=ax_twin)
        plt.title('Seasonality of rides per month')
        plt.xticks(rotation=90)
        sns.despine()
        plt.savefig(f'{viz_path}Seasonality of rides per month.png', dpi=300, transparent=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax_twin = ax.twinx()
        sns.barplot(x='weekday_started', y='count', data=seasonality_count(df)[1])
        sns.pointplot(x='weekday_started', y='count', hue = 'year_started', data=seasonality_count(df)[1], ax=ax_twin)
        plt.title('Seasonality of rides per weekday')
        plt.xticks(rotation=90)
        sns.despine()
        plt.savefig(f'{viz_path}Seasonality of rides per weekday.png', dpi=300, transparent=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax_twin = ax.twinx()
        sns.barplot(x='hour_started', y='count', data=seasonality_count(df)[2])
        sns.pointplot(x='hour_started', y='count', hue = 'year_started', data=seasonality_count(df)[2], ax=ax_twin)
        plt.title('Seasonality of rides per hour')
        plt.xticks(rotation=90)
        sns.despine()
        plt.savefig(f'{viz_path}Seasonality of rides per hour.png', dpi=300, transparent=True)


        plt.figure(figsize=(10, 6))
        plt.tight_layout()
        sns.barplot(x='hour_started', y='count', hue = 'weekday_started', data=seasonality_count(df)[2])
        plt.title('Seasonality of rides per hour and weekday')
        plt.xticks(rotation=90)
        sns.despine()
        plt.savefig(f'{viz_path}Seasonality of rides per hour and weekday.png', dpi=300, transparent=True)

    plotting(df)
    

if __name__ == '__main__':
    df = load_data()
    df = preprocess(df)
    analyse_data_statistics(df)
    