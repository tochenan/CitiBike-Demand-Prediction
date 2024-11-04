import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display
from utils import load_data, preprocess

def spatial_demand_extraction(df):
    # Create outgoing bikes data
    outgoing = df.groupby(['start_station_name', 'year_started', 'month_started', 'weekday_started', 'hour_started']).size().reset_index(name='outgoing_bikes')

    # Create incoming bikes data
    incoming = df.groupby(['end_station_name', 'year_ended', 'month_ended', 'weekday_ended', 'hour_ended']).size().reset_index(name='incoming_bikes')
 
    outgoing.rename(columns={'start_station_name': 'station_name', 'year_started': 'year', 'month_started': 'month', 'weekday_started': 'weekday', 'hour_started': 'hour'}, inplace=True)
    incoming.rename(columns={'end_station_name': 'station_name', 'year_ended': 'year', 'month_ended': 'month', 'weekday_ended': 'weekday', 'hour_ended': 'hour'}, inplace=True)

    # Merge the dataframes
    demand = pd.merge(outgoing, incoming, how='outer', on=['station_name', 'year', 'month', 'weekday', 'hour'])
    demand.fillna(0, inplace=True)
    demand['net_bikes'] = demand['incoming_bikes'] - demand['outgoing_bikes']
    
    return demand

def create_station_location_mapping(df):
    # Create DataFrames for start and end stations with unified column names
    start_stations = df[['start_station_name', 'start_lat', 'start_lng']].rename(
        columns={'start_station_name': 'station_name', 'start_lat': 'lat', 'start_lng': 'lng'}
    )

    end_stations = df[['end_station_name', 'end_lat', 'end_lng']].rename(
        columns={'end_station_name': 'station_name', 'end_lat': 'lat', 'end_lng': 'lng'}
    )
    # Concatenate and drop duplicates to get unique stations
    all_stations = pd.concat([start_stations, end_stations]).drop_duplicates(subset='station_name').reset_index(drop=True)
    # Create dictionary for mapping
    station_location_dict = all_stations.set_index('station_name')[['lat', 'lng']].apply(tuple, axis=1).to_dict()
    return station_location_dict


def visualize_timelapsed_network(demand, feature):

    # Define fixed longitude, latitude, and net bike ranges
    timelapsed_averaged = demand.groupby(['station_name', 'latitudes', 'longitudes', feature])['net_bikes'].mean().reset_index()
    lon_min, lon_max = timelapsed_averaged['longitudes'].min(), timelapsed_averaged['longitudes'].max()
    lat_min, lat_max = timelapsed_averaged['latitudes'].min(), timelapsed_averaged['latitudes'].max()
    net_bike_min, net_bike_max = timelapsed_averaged['net_bikes'].min(), timelapsed_averaged['net_bikes'].max()

    # Create a Normalize instance to fix color scaling
    norm = mcolors.Normalize(vmin=-20, vmax=20)

    # Define the function to update the plot based on the selected hour
    def update_plot(time):
        # Clear the current figure
        plt.figure(figsize=(10, 8))
        
        # Filter the data for the selected hour
        data = timelapsed_averaged[timelapsed_averaged[feature] == time]
        
        # Create the scatter plot with consistent size and color scaling
        sns.scatterplot(
            x='longitudes', 
            y='latitudes', 
            size='net_bikes',             # Use net_bikes directly for size
            sizes=(10, 100),              # Fixed min and max point sizes across all hours
            hue='net_bikes',              # Color based on net_bikes for consistent color scaling
            data=data, 
            palette='coolwarm',           # Use a consistent palette
            alpha=0.6, 
            legend='brief',
            hue_norm=norm                 # Apply the Normalize instance for consistent color scaling
        )
        
        # Set fixed axis limits
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)
        
        # Set plot labels and title for each frame
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Net Bike Count by Station Location - {feature}: {time}")
        # plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'), label='Net Bike Count')
        plt.show()

    # Create an interactive slider widget
    slider = widgets.IntSlider(value=0, min=0, max=23, step=1, description=f'{feature.capitalize()}')

    # Use the `interact` function to update the plot when the slider is changed
    widgets.interactive(update_plot,  time=slider)


def main():
    df = load_data()
    df = preprocess(df)
    demand = spatial_demand_extraction(df)
    station_location_dict = create_station_location_mapping(df)
    # Map both lat and lng using the dictionary of tuples
    demand['latitudes'] = demand['station_name'].map(lambda x: station_location_dict.get(x, (None, None))[0])
    demand['longitudes'] = demand['station_name'].map(lambda x: station_location_dict.get(x, (None, None))[1])
    
    visualize_timelapsed_network(demand, 'hour')


if __name__ == '__main__':
    main()