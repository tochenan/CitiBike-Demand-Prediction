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

- **data/**: This directory contains all datasets utilized in the project. For reasons of privacy and data size, these files are not tracked by version control. Guidance on accessing and using these datasets can be found in accompanying documentation within the directory.

- **notebooks/**: Houses Jupyter notebooks that provide a detailed exploratory data analysis and the steps involved in model training. These notebooks are essential for understanding the data features, the preprocessing steps applied, and the methodologies behind the model development.

- **src/**: Contains the source code for all functions and models related to the project, organized as follows:
  - **seasonality.py**: Implements functions to analyze seasonal trends in the dataset, crucial for understanding temporal patterns that influence model predictions.
  
  - **geo_exploration.py**: Includes scripts for geographic data exploration and analysis, helping to identify spatial patterns and the influence of geographical factors on bike usage.
  
  - **network.py**: Implements functions focused on network feature extraction.
  
  - **network_exploration.ipynb**- A jupyter notebook examining the relationship between network feature of a station, its geographical location and the current hour. Contains time-lapsed interactive visualisation.
  
  - **demand_forecasting.py**: Contains the predictive models and associated utilities for forecasting bike demand, employing time series analysis and machine learning techniques. Produce summary of model performance stored in evaluation/demand_model_performance.csv
  
  - **allocation_forecasting.py**: Features code for developing and refining models that predict bike allocation needs across various stations to ensure optimal availability and service efficiency. To better predict bike allocation, separate models were trained to predict incoming and outgoing bikes for each bike stations. 
  
  -  **model_explainability.py**: This script is focused on providing insights into the decision-making process of machine learning models used within the project. It employs various techniques to explain the predictions made by models. Run it after alloation_forecasting.py. Key functionalities include:
 	 - **Feature Importance**: Calculates the importance scores for each feature in the model, helping to identify what drives the model's decisions.
 	 - **SHAP Values**: Implements SHAP (SHapley Additive exPlanations) to determine how each feature contributes to individual predictions, offering a deep dive into the model's logic.
	 - **Map Visualiation**: Visualize net number of bikes in each station using the best performing models and overlay it on top of the map.
  
  - **utils.py**: Provides utility functions that are commonly used across the project, such as data loading, preprocessing, and data transformation routines, ensuring code reusability and modular architecture.

- **model/**: Stores serialized formats of the best-performing models. This directory may also include configuration files or additional metadata that describes model parameters, training procedures, and performance metrics, facilitating model replication or further development.

- **evaluation/**: Stores summary of model performance result for both demand and allocation forecasting tasks.

- **viz/**: Dedicated to data visualizations created during the analysis phase. This directory includes both exploratory graphics used to derive insights during model building and final visualizations that showcases the performance of the model. It also includes html objects that overlay important station statistics on top of a map.


## Setup Instructions
1. **Clone the repository**: https://github.com/tochenan/Causallens_Takehome.git
2. **Set up the environment** (ensure Anaconda is installed): dependency requirement can be found in environment.yml
