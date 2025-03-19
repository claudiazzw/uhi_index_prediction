# 🌍 Predicting Urban Heat Islands with AI

## Overview
Applying machine learning to predict urban heat island (UHI) hotspots in NYC and uncover key contributing factors driving temperature differences. Part of the 2025 EY Open Science AI & Data Challenge, this project aligns with the UN Sustainable Development Goals, aiming to mitigate heat risks and create a more sustainable urban environment. 🚀♻️ 

## Tech Stack 
**Programming Language:** Python

**Data Preprocessing:** Geopandas, rasterio

**Machine learning frameworks:** Scikit-Learn

**Visualization:** Seaborn, Pandas

## Project Structure
    .
    ├── data                   # Dataset (CSV & TIFF format)
    ├── notebooks              # Jupyter notebooks for EDA & modeling
    ├── src                    # Source code for data processing & ML models
    ├── models                 # Stores trained models and weights
    └── README.md              # Project documentation
    |__ Miscellaneous          # Additional template files (ex. Submission_template.csv)
    

## Dataset Used
### Target Dataset: 
UHI Index values for 11229 data points collected on July 24, 2021, in Bronx and Manhattan, NYC

### Feature Datasets:
European Sentinel-2 Optical Satellite Data

NASA Landsat Optical Satellite Data

### Additional Datasets:
Building footprints for Bronx & Manhattan

Local weather data for July 24, 2021


## Methodologies

### 1️⃣ EDA: Correlation Analysis, Heatmap



### 2️⃣ Feature Engineering:

To enhance model performance, several feature engineering techniques were employed.

* **Normalization**: Data preprocessing included applying **StandardScaler** to normalize temperature and index values, ensuring that different features are on comparable scales.

* **Derived Features**: We combined **Sentinel-2 bands (B01, B06, B08)** and **Land Surface Temperature data (Landsat LST)** to compute **Normalized Vegetation Built-up Index (NVBI)**. This index was used to assess the balance between vegetation and built-up areas, which is a critical determinant of UHI intensity.

* **Vegetation and Heat Indicators**: Features such as **Normalized Difference Vegetation Index (NDVI)** and **Land Surface Temperature anomalies (lst_anomaly)** were derived to capture heat absorption and vegetation cover influence on temperature variation.



### 3️⃣ Machine Learning Model:

We experimented with multiple machine learning models to predict UHI index values, with a focus on tree-based ensembles:

#### Random Forest:
* Handles high-dimensional data well and is robust to noise

* Effective at capturing complex, non-linear relationships between temperature and UHI-related features

* Provides feature importance rankings, aiding in understanding key contributors to UHI formation

#### XGBoost:
* More efficient and computationally optimized than traditional Random Forest

* Handles missing data effectively, which is crucial for satellite-derived datasets
   
* Performs well with structured data and has strong regularization capabilities, preventing overfitting in urban climate modeling


### 4️⃣ Hyperparameter Tuning:

To optimize the performance of the Random Forest model, multiple hyperparameter configurations were tested. The main parameter adjusted was n_estimators, which determines the number of trees in the forest. The results of these experiments are summarized below:

| Model Used |  Parameters  | Feature Selection | Results |
|:-----------|:-----------:|---------------------:|------:|
| Random Forest  | n_estimators=100, random_state=42 | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly | 0.7429 |
| Random Forest  | n_estimators=500, random_state=42 | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly | 0.7476 |
| Random Forest  | n_estimators=1000, random_state=42 | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly | 0.7489 |



## Evaluation Metrics
The model will be assessed based on:

✅ Root Mean Squared Error (RMSE)

✅ Feature Importance Analysis


## Data Licensing & Sources

Urban Temperature Data – [Apache 2.0 License](https://github.com/CenterForOpenScience/cos.io/blob/master/LICENSE)

Sentinel-2 Satellite Data – [CC BY-SA 3.0 License](https://creativecommons.org/licenses/by-sa/3.0/igo/)

NYC Building Footprint Data – [Apache 2.0 License](https://github.com/CityOfNewYork/nyc-geo-metadata#Apache-2.0-1-ov-file)

Weather Data – [NYS Mesonet Data Policy](https://nysmesonet.org/documents/NYS_Mesonet_Data_Access_Policy.pdf)




