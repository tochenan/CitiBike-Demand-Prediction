# Citi Bike Demand Prediction and Allocation

## Project Overview
This project leverages exploratory data analysis and machine learning to predict daily and hourly demand for Citi Bike services in New York City. The goal is to optimize bike allocation across various stations to meet fluctuating demands efficiently.

## Objectives
The project is structured around three key objectives:
1. **Exploratory Data Analysis**: To understand trends, patterns, and anomalies in the usage of Citi Bike services.
2. **Time Series Forecasting**: To predict peak daily demand using advanced forecasting models.
3. **Allocation Modeling**: To predict incoming and outgoing bike traffic for every station to ensure optimal distribution and availability.

## Data
The datasets used include historical Citi Bike usage patterns, geographical data of bike stations, and temporal data such as time of day and day of the week.
- **Note**: Data files are not included in this repository. Source data can be found on https://citibikenyc.com/system-data.

## Models Used
- **Time Series Forecasting**: We employ TBATS and XGBoost models, using features such as year, month, and weekday to forecast demand.
- **Machine Learning Allocation Model**: This model uses XGBoost trained on features including year, month, weekday, hour, geographical location, and a graph feature of each station to predict bike allocation needs.

## Folder Structure
- **data/**: Contains datasets used in the project (excluded from tracking).
- **notebooks/**: Jupyter notebooks for exploratory data analysis and model training.
- **src/**: Source code for all project-related functions and models.
  - **main.py**: Main script that runs the forecasting and allocation models.
  - **models.py**: Contains the definitions of the machine learning models.
  - **utils.py**: Utility functions for data handling, like loading and preprocessing.
- **model/**: Stores serialized versions of the best performing models.
- **viz/**: Directory for data visualizations generated during analysis.

## Setup Instructions
1. **Clone the repository**: https://github.com/tochenan/Causallens_Takehome.git
2. **Set up the environment** (ensure Anaconda is installed): dependency requirement can be found in environment.yml
