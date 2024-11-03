import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from utils import load_data, preprocess
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from tbats import TBATS
import seaborn as sns

viz_path = 'viz/'
save_path = 'model_performance/'

### DATA PREPROCESSING and FEATURE ENGINEERING

def temporal_demand_extraction(df):
    # Extract the temporal demand
    demand = df.groupby(['year_started','month_started','weekday_started','hour_started']).size().reset_index(name='count')
    demand = demand.groupby(['year_started','month_started','weekday_started']).agg({'count': 'max'}).reset_index()
    demand.rename(columns={'year_started': 'year', 'month_started': 'month', 'weekday_started': 'weekday', 'hour_started': 'hour'}, inplace=True)
    return demand

def generate_time_lagged_features(df, lags = 8):
    # Generate time-lagged features
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['count'].shift(lag)
    return df



### STATIONARITY TEST

def test_stationarity(timeseries):
    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def visualize_seasonality_decomposition(df, period = 7):
    # Visualize the seasonality decomposition
    result = seasonal_decompose(df['count'], model='additive', period=period)  
    result.plot()
    plt.savefig(f'{viz_path}seasonality_decomposition.png', dpi=300, transparent=True)
    plt.show()


### FORECASTING

def train_test_split(df, test_size = 0.2):
    # Split the data into train and test set for forecasting
    train_size = int(len(df) * (1 - test_size))
    train, test = df[:train_size], df[train_size:]
    return train, test


### TBATS model

def tbats_forecast(train, test, seasonality = [7, 7*12]):
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



### XGBoost model
def xgboost_forecast(train, test, visualize = True):
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


def model_evaluation(test, forecast):
    
    tbats_mae = mean_absolute_error(test, forecast)
    tbats_rmse = np.sqrt(mean_squared_error(test, forecast))
    tbats_mape = np.mean(np.abs((test - forecast) / test)) * 100

    performance = pd.DataFrame({'MAE': [tbats_mae], 'RMSE': [tbats_rmse], 'MAPE': [tbats_mape]})
    return performance


### VISUALIZATION

def visualize_forecast(train, test, fitted, forecast, model_name):
    # Visualize the forecast
    plt.figure(figsize=(6, 4))
    plt.plot(train, label='Observed', color = 'Tab:blue')
    plt.plot(fitted, label='Fitted', color='Tab:green')
    plt.plot(test, label='Observed', color = 'Tab:orange')
    plt.plot(test.index, forecast, label='Forecast', color='Tab:red')
    plt.legend()
    sns.despine()
    plt.savefig(f'{viz_path}{model_name}_forcast.png', dpi=300, transparent=True)
    plt.show()



def main():
    df = load_data()
    df = preprocess(df)
    demand = temporal_demand_extraction(df)
    train, test = train_test_split(demand)

    # visualize the seasonality decomposition
    visualize_seasonality_decomposition(demand)

    # fit the TBATS model
    tbat_fitted, tbats_forecasts = tbats_forecast(train, test)

    # fit the XGBoost model
    xgb_fitted, xgb_forecasts = xgboost_forecast(train, test)
     # visualize the forecast
    visualize_forecast(train['count'], test['count'], xgb_fitted, xgb_forecasts, 'xgb')

    # generate time-lagged features
    demand = generate_time_lagged_features(demand)

    # fit time-lagged features to XGBoost model
    train_lagged, test_lagged = train_test_split(demand)
    xgb_lagged_fitted, xgb_lagged_forcast = xgboost_forecast(train_lagged, test_lagged)



    # evaluate the TBATS model
    tbats_performance = model_evaluation(test['count'], tbats_forecasts)

    # evaluate the XGBoost model
    xgb_performance = model_evaluation(test['count'], xgb_forecasts)

    # evaluate the XGBoost model with time-lagged features
    xgb_lagged_performance = model_evaluation(test_lagged['count'], xgb_lagged_forcast)

    # summarize the performance
    summary = pd.concat([tbats_performance, xgb_performance, xgb_lagged_performance], axis=0)
    summary.index = ['TBATS', 'XGBoost', 'XGBoost with time-lagged features']

    summary.to_csv(f'{save_path}demand_model_performance.csv')



if __name__ == '__main__':
    main()

