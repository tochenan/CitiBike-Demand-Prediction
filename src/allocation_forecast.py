import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from tbats import BATS, TBATS
from xgboost import XGBRegressor

from network import graph_analysis
from utils import load_data, preprocess

features = [
    'latitudes',
    'longitudes',
    'year',
    'month',
    'weekday',
    'hour',
    'community',
    'betweenness',
    'pagerank']


def spatial_demand_extraction(df):
    """Extract the spatial demand from the data
    Args:
        df (pd.DataFrame): The dataframe containing the data
        Returns:
        pd.DataFrame: The dataframe containing the spatial demand"""
    # Create outgoing bikes data
    outgoing = df.groupby(['start_station_name',
                           'year_started',
                           'month_started',
                           'weekday_started',
                           'hour_started']).size().reset_index(name='outgoing_bikes')

    # Create incoming bikes data
    incoming = df.groupby(['end_station_name',
                           'year_ended',
                           'month_ended',
                           'weekday_ended',
                           'hour_ended']).size().reset_index(name='incoming_bikes')

    outgoing.rename(
        columns={
            'start_station_name': 'station_name',
            'year_started': 'year',
            'month_started': 'month',
            'weekday_started': 'weekday',
            'hour_started': 'hour'},
        inplace=True)
    incoming.rename(
        columns={
            'end_station_name': 'station_name',
            'year_ended': 'year',
            'month_ended': 'month',
            'weekday_ended': 'weekday',
            'hour_ended': 'hour'},
        inplace=True)

    # Merge the dataframes
    demand = pd.merge(
        outgoing, incoming, how='outer', on=[
            'station_name', 'year', 'month', 'weekday', 'hour'])
    demand.fillna(0, inplace=True)
    demand['net_bikes'] = demand['incoming_bikes'] - demand['outgoing_bikes']

    return demand


def create_station_location_mapping(df, save = True):
    """Create a mapping of station names to their respective latitudes and longitudes
    Args:
        df (pd.DataFrame): The dataframe containing the data
        Returns:
        dict: The dictionary containing the mapping of station names to their respective latitudes and longitudes"""
    
    # Create DataFrames for start and end stations with unified column names
    start_stations = df[['start_station_name', 'start_lat', 'start_lng']].rename(
        columns={'start_station_name': 'station_name', 'start_lat': 'lat', 'start_lng': 'lng'}
    )

    end_stations = df[['end_station_name', 'end_lat', 'end_lng']].rename(
        columns={'end_station_name': 'station_name', 'end_lat': 'lat', 'end_lng': 'lng'}
    )
    # Concatenate and drop duplicates to get unique stations
    all_stations = pd.concat([start_stations, end_stations]).drop_duplicates(
        subset='station_name').reset_index(drop=True)
    # Create dictionary for mapping
    station_location_dict = all_stations.set_index(
        'station_name')[['lat', 'lng']].apply(tuple, axis=1).to_dict()
    
    if save:
        pd.DataFrame(station_location_dict.items(), columns=['station_name', 'location']).to_csv('data/mapping/station_location_mapping.csv', index=False)
    return station_location_dict


def feature_preparation(df):
    """Prepare the features for the model, combine temporal, spatial, and network features, and create lookback columns
    Args:
        df (pd.DataFrame): The dataframe containing the data
        Returns:
        pd.DataFrame: The dataframe containing the prepared features"""

    # Extract spatial demand and map station names to their respective latitudes and longitudes
    demand = spatial_demand_extraction(df)
    station_location_dict = create_station_location_mapping(df)
    demand['latitudes'] = demand['station_name'].map(
        lambda x: station_location_dict.get(x, (None, None))[0])
    demand['longitudes'] = demand['station_name'].map(
        lambda x: station_location_dict.get(x, (None, None))[1])

    # Extract network features
    network_result = graph_analysis()
    features_long = pd.wide_to_long(
        network_result,
        stubnames=[
            'in_degree',
            'out_degree',
            'community',
            'betweenness',
            'pagerank'],
        i='station_name',
        j='hour',
        sep='_').reset_index()

    # Merge df_hours with the reshaped df_features
    combined_df = pd.merge(
        demand, features_long, on=[
            'station_name', 'hour'], how='left')

    # List of graph features to look back on
    features = [
        'in_degree',
        'out_degree',
        'community',
        'betweenness',
        'pagerank']

    # Add lookback columns for each feature
    for feature in features:
        for i in range(1, 24):  # Create 12 lookback columns
            combined_df[f'{feature}_lag_{i}'] = combined_df.groupby('station_name')[
                feature].shift(i)

    return combined_df


# XGBoost model
def xgboost_optimised(train, test, target='net_bikes'):
    """Optimise the XGBoost model using GridSearchCV
    Args:
        train (pd.DataFrame): The training data
        test (pd.DataFrame): The test data
        target (str): The target column
        Returns:
        XGBRegressor: The best XGBoost model
        np.array: The fitted values
        np.array: The forecasted values"""
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [1, 3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
        'reg_alpha': [0, 0.1, 0.5],   # L1 regularization
        'reg_lambda': [1, 1.5, 2]     # L2 regularization
    }

    # Initialize the XGBoost regressor
    xgb_model = XGBRegressor(random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',  # Negative MSE (minimize MSE)
        cv=3,                              # 3-fold cross-validation
        verbose=1,
        n_jobs=-1                          
    )

    X_train = train.drop([target], axis=1)
    y_train = train[target]
    X_test = test.drop([target], axis=1)
    y_test = test[target]

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_model = grid_search.best_estimator_

    # Use the best model to make predictions
    xgb_fitted = best_model.predict(X_train)
    xgb_forecast = best_model.predict(X_test)

    return best_model, xgb_fitted, xgb_forecast


def xgboost_fit(train, test, target='incoming_bikes'):
    """Fit an XGBoost model and forecast the demand
    Args:
    train (pd.DataFrame): The training data
    test (pd.DataFrame): The test data
    target (str): The target column
    
    Returns:
    XGBRegressor: The XGBoost model
    np.array: The fitted values
    np.array: The forecasted values
    """


    # Initialize the XGBoost regressor
    xgb_model = XGBRegressor(random_state=42)

    X_train = train.drop([target], axis=1)
    y_train = train[target]
    X_test = test.drop([target], axis=1)
    y_test = test[target]

    xgb_model.fit(X_train, y_train)

    # Use the model to make predictions
    xgb_fitted = xgb_model.predict(X_train)
    xgb_forecast = xgb_model.predict(X_test)

    return xgb_model, xgb_fitted, xgb_forecast


def model_evaluation(test, forecast):
    """Evaluate the model performance using MAE, RMSE, and MAPE
    Args:     
    test (np.array): The actual values
    forecast (np.array): The forecasted values
    
    Returns:
    pd.DataFrame: The model performance metrics
    """

    tbats_mae = mean_absolute_error(test, forecast)
    tbats_rmse = np.sqrt(mean_squared_error(test, forecast))
    tbats_mape = np.mean(np.abs((test - forecast) / (forecast + 1e-10))) * 100

    performance = pd.DataFrame(
        {'MAE': [tbats_mae], 'RMSE': [tbats_rmse], 'MAPE': [tbats_mape]})
    return performance


def train_test_split(df, test_size=0.2, save_path='data/', save=False):
    """Split the data into train and test set for forecasting
    Args:
    df (pd.DataFrame): The dataframe containing the data
    test_size (float): The proportion of the data to be used for testing

    Returns:
    pd.DataFrame: The training data
    pd.DataFrame: The test data
    """
    
    train_size = int(len(df) * (1 - test_size))
    train, test = df[:train_size], df[train_size:]

    if save:
        train.to_csv(f'{save_path}train/train.csv', index=False)
        test.to_csv(f'{save_path}test/test.csv', index=False)
    return train, test


def train_and_evaluate_all_models(
        combined_df,
        target='incoming_bikes',
        features=features):
    """Train and evaluate all models for the given target variable
    Args:
    combined_df (pd.DataFrame): The dataframe containing all the features and target variable
    target (str): The target variable
    features (list): The list of features to be used for training

    Returns:
    pd.DataFrame: The model performance metrics for all models
    """


    # Untransformed fit
    combined_df_encoded = pd.get_dummies(
        combined_df[features + [target]], columns=['year'])
    train, test = train_test_split(combined_df_encoded, test_size=0.2)
    _, xgb_fitted, xgb_forecasts = xgboost_fit(train, test, target=target)

    untransformed_performance = model_evaluation(test[target], xgb_forecasts)

    # log-transformed fit

    combined_df['log'] = np.log1p(combined_df[target])
    combined_df_encoded = pd.get_dummies(
        combined_df[features + ['log']], columns=['year'])
    train_log, test_log = train_test_split(combined_df_encoded, test_size=0.2)
    _, xgb_fitted_log, xgb_forecasts_log = xgboost_fit(
        train_log, test_log, target='log')

    log_fitted_performance = model_evaluation(test_log['log'], xgb_forecasts_log)

    # log-transformed fit and filter out low demand stations
    filtered_df = combined_df[combined_df[target] > 9]
    filtered_df['log'] = np.log1p(filtered_df[target])
    combined_df_encoded = pd.get_dummies(
        filtered_df[features + ['log']], columns=['year'])
    train_filtered, test_filtered = train_test_split(combined_df_encoded, test_size=0.2, save=True)
    best_model, xgb_fitted_log_fitlered, xgb_forecasts_log_filtered = xgboost_fit(
        train_filtered, test_filtered, target='log')

    log_fitted_filtered_performance = model_evaluation(
        test_filtered['log'], xgb_forecasts_log_filtered)

    # untransformed fit and filter out low demand stations + lookback features

    lag_features = [i for i in combined_df.columns if 'lag' in i]
    features = features + lag_features
    combined_df_encoded = pd.get_dummies(
        filtered_df[features + ['log']], columns=['year'])
    train_lag, test_lag = train_test_split(combined_df_encoded, test_size=0.2)
    _, lag_xgb_fitted, lag_xgb_forcasts = xgboost_fit(
        train_lag, test_lag, target='log')

    lag_fitted_performance = model_evaluation(test_lag['log'], lag_xgb_forcasts)

    # Optimised the best model using GridSearchCV
    best_model, xgb_fitted_optimised, xgb_forecasts_optimised = xgboost_optimised(
        train_filtered, test_filtered, target='log')
    best_model.save_model(f'models/best_model_{target}.json')

    optimised_performance = model_evaluation(
        test_filtered['log'], xgb_forecasts_optimised)

    # Combine all the results
    summary = pd.concat([untransformed_performance,
                         log_fitted_performance,
                         log_fitted_filtered_performance,
                         lag_fitted_performance,
                         optimised_performance],
                        axis=0)
    summary.index = [
        'untransformed',
        'log_fitted',
        'log_fitted_filtered',
        'lag_fitted',
        'optimised']
    summary.to_csv(f'evaluation/{target}_model_performance.csv')
    return summary


def main():
    df = load_data()
    df = preprocess(df)
    combined_df = feature_preparation(df)
    train_and_evaluate_all_models(combined_df, target='incoming_bikes')
    train_and_evaluate_all_models(combined_df, target='outgoing_bikes')


if __name__ == '__main__':
    main()
