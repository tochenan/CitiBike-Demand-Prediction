import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tbats import TBATS
from xgboost import XGBRegressor

from utils import load_data, preprocess

viz_path = 'viz/'
save_path = 'model_performance/'

# DATA PREPROCESSING and FEATURE ENGINEERING


def data_preprocessing():
    """Preprocess the data

    Returns:
    df: A pandas dataframe"""
    # Preprocess the data
    df = load_data()
    df = preprocess(df)
    demand = temporal_demand_extraction(df)
    train, test = train_test_split(demand)
    return df, demand, train, test


def temporal_demand_extraction(df):
    """Extract the temporal demand
    Args:
    df: A pandas dataframe
    Returns:
    A pandas dataframe"""
    # Extract the temporal demand
    demand = df.groupby(['year_started',
                         'month_started',
                         'weekday_started',
                         'hour_started']).size().reset_index(name='count')
    demand = demand.groupby(['year_started', 'month_started', 'weekday_started']).agg(
        {'count': 'max'}).reset_index()
    demand.rename(
        columns={
            'year_started': 'year',
            'month_started': 'month',
            'weekday_started': 'weekday',
            'hour_started': 'hour'},
        inplace=True)
    return demand


def generate_time_lagged_features(df, lags=8):
    """Generate time-lagged features for forecasting (total rides in the past lag hours)
    Args:
    df: A pandas dataframe
    lags: An integer indicating the number of time lags
    Returns:
    A pandas dataframe"""
    # Generate time-lagged features
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['count'].shift(lag)
    return df


# STATIONARITY AND SEASONALITY

def test_stationarity(timeseries):
    """Test the stationarity of a time series using the Dickey-Fuller test and print the results
    Args:
    timeseries: A pandas series
    Returns:
    None"""
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic',
                                'p-value',
                                '#Lags Used',
                                'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def visualize_seasonality_decomposition(df, period=7):
    """Visualize the seasonality decomposition
    Args:
    df: A pandas dataframe
    period: An integer indicating the seasonal period
    Returns:
    None"""

    result = seasonal_decompose(df['count'], model='additive', period=period)
    result.plot()
    plt.savefig(
        f'{viz_path}seasonality_decomposition.png',
        dpi=300,
        transparent=True)
    plt.show()


# MODELING

def train_test_split(df, test_size=0.2):
    """Split the data into train and test set for forecasting
    Args:
    df: A pandas dataframe
    test_size: A float indicating the proportion of the test set
    Returns:
    train: A pandas dataframe
    test: A pandas dataframe"""

    # Split the data into train and test set for forecasting
    train_size = int(len(df) * (1 - test_size))
    train, test = df[:train_size], df[train_size:]
    return train, test


# TBATS model

def tbats_forecast(train, test, seasonality=[7, 7 * 12]):
    """Forecast the demand using TBATS model
    Args:
    train: A pandas dataframe
    test: A pandas dataframe
    seasonality: A list of integers indicating the seasonal periods
    Returns:
    tbats_fitted: A numpy array
    tbats_forecasts: A numpy array"""

    # Forecast the demand using TBATS model
    train = train['count']
    test = test['count']

    # Create a TBATS estimator specifying the seasonal periods
    estimator = TBATS(seasonal_periods=seasonality)

    # Fit the model
    model = estimator.fit(train)

    # Summarize model effects
    print(model.summary())

    # Forecast future values, e.g., forecast the next 48 hours
    tbats_fitted = model.y_hat
    tbats_forecasts = model.forecast(steps=len(test))

    return tbats_fitted, tbats_forecasts


# XGBoost model
def xgboost_forecast(train, test, visualize=True):
    """Forecast the demand using XGBoost model and return the fitted and forecasted values
    Args:
    train: A pandas dataframe
    test: A pandas dataframe
    visualize: A boolean indicating whether to visualize the forecast
    Returns:
    xgb_fitted: A numpy array
    xgb_forecast: A numpy array"""

    # Forecast the demand using XGBoost model

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
        n_jobs=-1                          # Use all available cores
    )

    X_train = train.drop(['count'], axis=1)
    y_train = train['count']
    X_test = test.drop(['count'], axis=1)
    y_test = test['count']

    # Assuming X_train and y_train are your features and target
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_model = grid_search.best_estimator_

    # use the best model to make predictions
    xgb_fitted = best_model.predict(X_train)
    xgb_forecast = best_model.predict(X_test)

    return xgb_fitted, xgb_forecast


# MODEL EVALUATION

def model_evaluation(test, forecast):
    """Evaluate the model using MAE, RMSE, and MAPE
    Args:
    test: A pandas series, the actual values
    forecast: A numpy array, the forecasted values
    Returns:
    performance: A pandas dataframe"""

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    performance = pd.DataFrame({'MAE': [mae], 'RMSE': [rmse], 'MAPE': [mape]})
    return performance

# MODEL SELECTION


def train_and_evaluate_all_models(demand, train, test):
    """Train and evaluate all models
    Args:
    train: A pandas dataframe
    test: A pandas dataframe
    Returns:
    best_model: A string, the best model
    best_fitted: A numpy array, the fitted values
    best_forecast: A numpy array, the forecasted values"""

    # Fit the TBATS model
    tbat_fitted, tbats_forecasts = tbats_forecast(train, test)

    # Fit the XGBoost model
    xgb_fitted, xgb_forecasts = xgboost_forecast(train, test)

    # Fit the time-lagged features to XGBoost model
    demand = generate_time_lagged_features(demand)
    train_lagged, test_lagged = train_test_split(demand)
    xgb_lagged_fitted, xgb_lagged_forcast = xgboost_forecast(
        train_lagged, test_lagged)

    # Evaluate the TBATS model
    tbats_performance = model_evaluation(test['count'], tbats_forecasts)

    # Evaluate the XGBoost model
    xgb_performance = model_evaluation(test['count'], xgb_forecasts)

    # Evaluate the XGBoost model with time-lagged features
    xgb_lagged_performance = model_evaluation(
        test_lagged['count'], xgb_lagged_forcast)

    # Summarize the performance
    summary = pd.concat([tbats_performance, xgb_performance,
                        xgb_lagged_performance], axis=0)
    summary.index = ['TBATS', 'XGBoost', 'XGBoost with time-lagged features']

    # get the best model
    best_model = summary['MAPE'].idxmin()
    best_fitted = xgb_fitted if best_model == 'XGBoost' else tbat_fitted if best_model == 'TBATS' else xgb_lagged_fitted
    best_forecast = xgb_forecasts if best_model == 'XGBoost' else tbats_forecasts if best_model == 'TBATS' else xgb_lagged_forcast

    summary.to_csv(f'{save_path}demand_model_performance.csv')

    return best_model, best_fitted, best_forecast


# DEMAND PREDICTION

def forecast_demand(model, year, month, weekday):
    """Forecast the demand for a specific year, month, and weekday
    Args:
    model: A string, the model name
    year: An integer, the year
    month: An integer, the month
    weekday: An integer, the weekday
    Returns:
    forecast: A float, the forecasted demand"""

    forecast = model.predict([[year, month, weekday]])
    return forecast


# VISUALIZATION

def visualize_forecast(train, test, fitted, forecast, model_name):
    """Visualize the original, fitted, and forecasted values in the same plot
    Args:
    train: A pandas series, the training set
    test: A pandas series, the test set
    fitted: A numpy array, the fitted values
    forecast: A numpy array, the forecasted values
    model_name: A string, the model name
    Returns:
    Figure"""

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(train, label='Observed', color='Tab:blue')
    plt.plot(fitted, label='Fitted', color='Tab:green')
    plt.plot(test, label='Observed', color='Tab:orange')
    plt.plot(test.index, forecast, label='Forecast', color='Tab:red')
    plt.legend()
    sns.despine()
    return fig


def visualize_best_model(train, test, model, fitted, forecast):
    """Visualize the best model
    Args:
    model: A string, the model name
    fitted: A numpy array, the fitted values
    forecast: A numpy array, the forecasted values
    Returns:
    None"""

    fig = visualize_forecast(
        train['count'],
        test['count'],
        fitted,
        forecast,
        model)
    plt.title(f'{model} Forecast')
    plt.savefig(f'{viz_path}{model}_forecast.png', dpi=300, transparent=True)
    plt.show()


def main():
    # Load and preprocess the data
    df, demand, train, test = data_preprocessing()

    # Test the stationarity of the demand
    test_stationarity(demand['count'])

    # Visualize the seasonality decomposition
    visualize_seasonality_decomposition(demand)

    # Train and evaluate all models
    best_model, best_fitted, best_forecast = train_and_evaluate_all_models(
        demand, train, test)

    # Visualize the best model
    visualize_best_model(train, test, best_model, best_fitted, best_forecast)

    # Forecast the demand for a specific year, month, and weekday
    year = 2024
    month = 11
    weekday = 1
    forecast = forecast_demand(best_model, year, month, weekday)
    print(f'The forecasted demand for {year}-{month}-{weekday} is {forecast}')


if __name__ == '__main__':
    main()
