import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins

from utils import load_data, preprocess


viz_path = 'viz/'

def sample_data(df, sample_size):
    return df.sample(sample_size)


def visualize_heatmap(df):
    # Visualize the heatmap
    m = folium.Map(location=[df['start_lat'].mean(), df['start_lng'].mean()], zoom_start=12)
    m.add_child(plugins.HeatMap(data=df[['start_lat', 'start_lng']], radius=15))
    m.save(f'{viz_path}heatmap.html')

def visualize_start_station_distribution(df):
    # visualize year distribution
    plt.figure(figsize=(10, 6))
    sns.jointplot(x=df['start_lat'], y=df['start_lng'], hue = df['year_started'], kind = 'kde', height=10)
    sns.despine()
    plt.savefig(f'{viz_path}year_distribution.png', dpi=300, transparent=True)
    plt.show()  

    # visualize month distribution
    plt.figure(figsize=(10, 6))
    sns.jointplot(x=df['start_lat'], y=df['start_lng'], hue = df['month_started'], kind = 'kde', height=10)
    sns.despine()
    plt.savefig(f'{viz_path}month_distribution.png', dpi=300, transparent=True)
    plt.show

    # visualize weekday distribution
    plt.figure(figsize=(10, 6))
    sns.jointplot(x=df['start_lat'], y=df['start_lng'], hue = df['weekday_started'], kind = 'kde', height=10)
    sns.despine()
    plt.savefig(f'{viz_path}weekday_distribution.png', dpi=300, transparent=True)
    plt.show()

    # visualize hour distribution
    plt.figure(figsize=(10, 6))
    sns.jointplot(x=df['start_lat'], y=df['start_lng'], hue = df['hour_started'], kind = 'kde', height=10, palette='RdGy')
    sns.despine()
    plt.savefig(f'{viz_path}hour_distribution.png', dpi=300, transparent=True)
    plt.show()

    
if __name__ == '__main__':
    df = load_data()
    df = preprocess(df)
    sampled_df = sample_data(df, 100000)
    visualize_heatmap(sampled_df)
    visualize_start_station_distribution(sampled_df)
