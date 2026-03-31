## Citi Bike Demand and Allocation Forecasting

This repository contains data prep, modeling, and visualization code for Citi Bike demand forecasting and station-level allocation. The code covers temporal demand modeling, station network features, and map-based outputs to help interpret model results.

## What is included
- Time-series demand forecasting with TBATS and XGBoost.
- Station-level incoming/outgoing bike prediction with spatial + network features.
- Graph-based feature extraction from station-to-station trip flows.
- Visualization utilities and map overlays for model outputs.

## Repository layout
- **data/**: Input data and derived splits.
  - **raw/**: Original Citi Bike CSVs.
  - **train/** and **test/**: Train/test splits for station models.
  - **mapping/**: Station name to location mapping.
- **evaluation/**: Model performance summaries (CSV).
- **models/**: Saved TBATS and XGBoost models.
- **src/**: Python modules for preprocessing, modeling, and visualization.
- **viz/**: Output figures and interactive HTML maps.

## Data expectations
Place Citi Bike system data CSVs in **data/raw/**. The loader reads all CSVs in that directory and sorts by `started_at`. Source data: https://citibikenyc.com/system-data

## Environment
Create the environment from **environment.yml** (Conda) or install the listed dependencies manually.

## How to run
Run scripts directly from the project root:

```bash
python src/demand_forecast.py
python src/allocation_forecast.py
python src/model_explainability.py
python src/geo_exploration.py
python src/seasonality.py
python src/network.py
```

## Outputs
- **evaluation/**: Model performance CSVs for demand and allocation.
- **models/**: Saved model artifacts for re-use.
- **viz/**: PNGs and HTML maps (heatmaps, forecasts, net bike overlays).
