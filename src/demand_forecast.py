import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from utils import load_data, preprocess
from tbats import TBATS
import seaborn as sns

viz_path = 'viz/'

def temporal_demand_extraction(df):
    # Extract the temporal demand
    demand = df.groupby(['year_started','month_started','weekday_started','hour_started']).size().reset_index(name='count')
    demand = demand.groupby(['year_started','month_started','weekday_started']).agg({'count': 'max'}).reset_index()
    demand.rename(columns={'year_started': 'year', 'month_started': 'month', 'weekday_started': 'weekday', 'hour_started': 'hour'}, inplace=True)
    return demand


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
    plt.show()



def train_test_split(df, test_size = 0.3):
    # Split the data into train and test set for forecasting
    train_size = int(len(df) * (1 - test_size))
    train, test = df[:train_size], df[train_size:]
    return train, test


def tbats_forecast(train, test, seasonality = [7, 7*12], visualize = True):
    # Forecast the demand using TBATS model

    # Create a TBATS estimator specifying the seasonal periods
    estimator = TBATS(seasonal_periods=seasonality)

    # Fit the model
    model = estimator.fit(train)

    # Summarize model effects
    print(model.summary())

    # Forecast future values, e.g., forecast the next 48 hours
    tbats_fitted = model.y_hat
    tbats_forecasts = model.forecast(steps=len(test))

    if visualize:

        # Plotting the results
        plt.figure(figsize=(6, 4))
        plt.plot(df['count'], label='Observed')

        original_index = range(len(train))
        forecast_index = range(len(train), len(train) + len(test))

        plt.plot(original_index, tbats_fitted, label='Fitted Values', color='green')
        plt.plot(forecast_index, tbats_forecasts, label='Forecasts', color='Tab:red')


        plt.title('TBATS Forecast')
        plt.legend()
        sns.despine()
        plt.savefig(f'{viz_path}TBATS Forecast.png', dpi=300, transparent=True)
        plt.show()

    return test, tbats_forecasts



def xgboost_forecast(train, test, visualize = True):



def model_evaluation(test, forecast):
    
    tbats_mae = mean_absolute_error(test, forecast)
    tbats_rmse = np.sqrt(mean_squared_error(test, forecast))
    tbats_mape = np.mean(np.abs((test - forecast) / test)) * 100

    performance = pd.DataFrame({'MAE': [tbats_mae], 'RMSE': [tbats_rmse], 'MAPE': [tbats_mape]})
    

def main():
    df = load_data()
    df = preprocess(df)
    demand = temporal_demand_extraction(df)
    train, test = train_test_split(demand)
    test, tbats_forecasts = tbats_forecast(train, test)
    model_evaluation(test, tbats_forecasts)

