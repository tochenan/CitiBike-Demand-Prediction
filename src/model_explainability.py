import folium
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from xgboost import XGBRegressor

from allocation_forecast import create_station_location_mapping

model_path = 'models/'
data_path = 'data/'
viz_path = 'viz/'
mapping_path = 'data/mapping/station_location_mapping.csv'

features = [
    'latitudes',
    'longitudes',
    'month',
    'weekday',
    'hour',
    'community',
    'betweenness',
    'pagerank',
    'year_2022',
    'year_2023',
    'year_2024',]


# LOAD DATA AND MODELS

def load_station_location_mapping():
    """Load the station location mapping
    Returns:
    station_location_dict (dict): The station location mapping
    """
    station_location = pd.read_csv(mapping_path)
    station_location['location'] = station_location.apply(
        lambda x: tuple(map(float, x['location'].strip('()').split(','))), axis=1)
    station_location_dict = pd.Series(
        station_location['location'].values,
        index=station_location.station_name).to_dict()
    return station_location_dict


def load_best_model(path, target='incoming_bikes'):
    """Load the best model from the specified path
    Args:
    path (str): The path to the model
    target (str): The target variable of the model
    Returns:
    best_model (XGBRegressor): The best model
    """

    best_model = XGBRegressor()
    best_model.load_model(f'{path}best_model_{target}.json')
    return best_model


def load_train_test_data(path):
    """Load the train and test data from the specified path
    Args:
    path (str): The path to the data
    Returns:
    train (pd.DataFrame): The train data
    test (pd.DataFrame): The test data"""
    train = pd.read_csv(f'{path}train/train.csv')
    test = pd.read_csv(f'{path}test/test.csv')
    return train, test


# SHAP VISUALISATION
def shap_visualisation(model, test):
    """Create SHAP visualisations for the model
    Args:
    model (XGBRegressor): The model to explain
    test (pd.DataFrame): The test data

    Returns:
    None
    """

    # 1. Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    X_test = test.drop(['log'], axis=1)

    # 2. Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # 3. Summary Plot: Shows overall feature importance and direction of impact
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(
        f'{viz_path}' +
        'shap_summary_plot.png',
        dpi=300,
        transparent=True)
    plt.close()

    # 4. Detailed Summary Plot: Shows how each feature affects predictions
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(
        f'{viz_path}' +
        'shap_detailed_summary_plot.png',
        dpi=300,
        transparent=True)
    plt.close()

    # 5. Individual SHAP Value Plot: Shows the impact of each feature on a
    # single prediction
    shap.dependence_plot("hour", shap_values, X_test, show=False)

    plt.savefig(
        f'{viz_path}' +
        'shap_individual_plot.png',
        dpi=300,
        transparent=True)
    plt.close()

    # 6. Force Plot: Shows the impact of each feature on a single prediction
    shap.force_plot(explainer.expected_value,
                    shap_values[0, :], X_test.iloc[0, :], show=False)
    plt.savefig(
        f'{viz_path}' +
        'shap_force_plot.png',
        dpi=300,
        transparent=True)
    plt.close()

    # 7. Dependence Plot: Shows the impact of a single feature across the
    # whole dataset
    shap.dependence_plot(
        "hour",
        shap_values,
        X_test,
        interaction_index="hour",
        show=False)
    plt.savefig(
        f'{viz_path}' +
        'shap_dependence_plot.png',
        dpi=300,
        transparent=True)
    plt.close()


def model_explainability():
    """Create SHAP visualisations for the best model
    Returns:
    None
    """
    model = load_best_model(model_path)
    train, test = load_train_test_data(data_path)
    shap_visualisation(model, test)


# PREDICTION AND VISUALISATION
def generate_net_bike_predictions(
        year=2024,
        month=4,
        hour=8,
        weekday=1,
        forecast_month=11):
    """Generate net bike predictions for a specific time period
    Args:
    year (int): The year of the prediction
    month (int): The month of the prediction
    hour (int): The hour of the prediction
    weekday (int): The weekday of the prediction
    forecast_month (int): The month of the forecast
    Returns:
    predictions (pd.DataFrame): The net bike predictions
    """

    incoming_model = load_best_model(model_path, target='incoming_bikes')
    outgoing_model = load_best_model(model_path, target='outgoing_bikes')
    train, test = load_train_test_data(data_path)
    original = pd.concat([train, test], axis=0)

    filtered_df = original[(original[f'year_{year}']) & (original['month'] == month) & (
        original['hour'] == hour) & (original['weekday'] == weekday)]

    filtered_df['month'] = forecast_month

    incomings = incoming_model.predict(filtered_df[features])
    outgoings = outgoing_model.predict(filtered_df[features])

    net_bikes = np.expm1(incomings) - np.expm1(outgoings)
    predictions = pd.DataFrame()
    predictions['latitudes'] = filtered_df['latitudes']
    predictions['longitudes'] = filtered_df['longitudes']
    predictions['net_bikes'] = net_bikes

    def get_station_name(lat, lon):
        """Get the station name from the latitude and longitude
        Args:
        lat (float): The latitude
        lon (float): The longitude
        Returns:
        station (str): The station name"""
        station_location_dict = load_station_location_mapping()
        for station, (station_lat,
                      station_lon) in station_location_dict.items():
            if station_lat == lat and station_lon == lon:
                return station
        return None

    predictions['station_name'] = predictions.apply(
        lambda x: get_station_name(
            x['latitudes'], x['longitudes']), axis=1)
    return predictions


def visualize_netbikes(predictions):
    """Visualize the net bike predictions on a map
    Args:
    predictions (pd.DataFrame): The net bike predictions
    Returns:
    None
    """

    def get_color(net_bike):
        """Get the color for the net bike value
        Args:
        net_bike (float): The net bike value
        Returns:
        color (str): The color code"""
        return matplotlib.colors.to_hex(mapper.to_rgba(net_bike))

    # Function to adjust marker radius
    def get_radius(net_bike):
        """ Get the radius for the net bike value
        Args:
        net_bike (float): The net bike value
        Returns:
        radius (int): The radius of the marker"""
        # Minimum radius is 3, scale down if the original is too large
        return max(3, min(10, 3 + abs(net_bike)))

    # Calculate the center of the map
    map_center_latitude = predictions['latitudes'].mean()
    map_center_longitude = predictions['longitudes'].mean()

    # Create a map
    m = folium.Map(
        location=[
            map_center_latitude,
            map_center_longitude],
        zoom_start=13)

    # Define the colors for the negative, zero, and positive values
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mycmap", ["blue", "white", "red"])

    # Normalize the net bike values
    min_netbike = predictions['net_bikes'].min()
    max_netbike = predictions['net_bikes'].max()
    norm = mcolors.TwoSlopeNorm(vmin=min_netbike, vcenter=0, vmax=max_netbike)

    mapper = matplotlib.cm.ScalarMappable(
        norm=norm, cmap=cmap)  # Create a color mapper

    # Add a marker for each station
    for _, row in predictions.iterrows():
        folium.CircleMarker(
            location=[row['latitudes'], row['longitudes']],
            radius=get_radius(row['net_bikes']),  # Adjusted radius
            color=get_color(row['net_bikes']),
            fill=True,
            fill_color=get_color(row['net_bikes']),
            fill_opacity=0.7,
            tooltip=f"Station: {
                row['station_name']}<br>Net Bike: {
                row['net_bikes']}"
        ).add_to(m)

    # Save the map as an HTML file
    m.save('viz/predictions.html')


if __name__ == '__main__':
    model_explainability()
    predictions = generate_net_bike_predictions()
    visualize_netbikes(predictions)
