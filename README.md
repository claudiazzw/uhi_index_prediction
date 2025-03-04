# 🌍 Predicting Urban Heat Islands with AI

## Overview
Applying machine learning to predict urban heat island (UHI) hotspots in NYC and uncover key contributing factors driving temperature differences. Part of the 2025 EY Open Science AI & Data Challenge, this project aligns with the UN Sustainable Development Goals, aiming to mitigate heat risks and create a more sustainable urban environment. 🚀♻️ 

## Tech Stack
**Programming Language:** Python

**Machine learning frameworks:** Scikit-Learn

**Visualization:** 

## Project Structure
    .
    ├── data                   # Dataset (CSV & TIFF format)
    ├── notebooks              # Jupyter notebooks for EDA & modeling
    ├── src                    # Source code for data processing & ML models
    └── README.md              # Project documentation
    

## Dataset Used
#### Target Dataset: 
UHI Index values for 11229 data points collected on July 24, 2021, in Bronx and Manhattan, NYC

#### Feature Datasets:
European Sentinel-2 Optical Satellite Data

NASA Landsat Optical Satellite Data

#### Additional Datasets:
Building footprints for Bronx & Manhattan

Local weather data for July 24, 2021

## Methodologies
**EDA:** Correlation Analysis, Heatmap

**Feature Engineering:**

We combined ... and ... to derive the NVBI __ value to determine___

**Machine Learning Model:** XGBoost / Random Forest

**Hyperparameter Tuning:** GridSearch 

To optimize the performance of the Random Forest model, multiple hyperparameter configurations were tested. The main parameter adjusted was n_estimators, which determines the number of trees in the forest. The results of these experiments are summarized below:

| Model Used |  Parameters  | Feature Selection | Results |
|:-----------|:-----------:|---------------------:|------:|
| Random Forest  | n_estimators=100, random_state=42 | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly | 0.7429 |
| Random Forest  | n_estimators=500, random_state=42 | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly | 0.7476 |
| Random Forest  | n_estimators=1000, random_state=42 | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly | 0.7489 |



## Evaluation Metrics
The model will be assessed based on:

✔️ Root Mean Squared Error (RMSE)

✔️ Feature Importance Analysis





