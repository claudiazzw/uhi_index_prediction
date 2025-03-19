# 🌆 Predicting Urban Heat Islands with AI

## 🌟 Project Overview

Urban Heat Islands (UHIs)—areas significantly warmer than their surroundings—pose severe environmental and public health risks. As part of the **2025 EY Open Science AI & Data Challenge**, this project leverages advanced machine learning techniques to predict UHI hotspots across New York City's Bronx and Manhattan areas and identify the key factors driving these heat disparities.

Our goal aligns with the UN Sustainable Development Goals, specifically aiming to create more sustainable, resilient, and equitable urban environments. 🌍♻️

---

## 🔧 Tech Stack

- **Language:** Python 🐍
- **Data Processing:** Geopandas, Rasterio, SciPy, Shapely, Pyproj
- **Machine Learning:** Scikit-Learn, KDTree, XGBoost, Random Forest
- **Visualization:** Seaborn, Pandas, Matplotlib, GeoPandas plot
- **Interactive Exploration:** Jupyter Notebooks 📓

---

## 📂 Project Structure

```
.
├── data                   # Raw datasets (CSV & TIFF formats)
├── notebooks              # Jupyter notebooks for exploratory analysis & modeling
├── models                 # Saved trained models and weights
├── miscellaneous          # Additional resources (e.g., Submission templates)
├── requirements.txt       # Package Versions
└── README.md              # Project documentation

```

---

## 📊 Datasets Used

### 🎯 **Target Variable:**
- **UHI Index:** Contains 11,229 observations collected on July 24, 2021, in the Bronx and Manhattan, NYC.

### 🛰️ **Feature Data:**
- **European Sentinel-2 Satellite Data:** Multispectral optical data.
- **NASA Landsat Satellite Data:** Land Surface Temperature (LST) measurements.

### 🗺️ **Additional Supporting Data:**
- **Building Footprints:** Detailed building outlines for the Bronx & Manhattan.
- **Local Weather Data:** Hourly weather observations from July 24, 2021.

---

## 🚀 Methodology

### 1️⃣ **Exploratory Data Analysis (EDA)**
- **Correlation Analysis & Spatial Visualization**: Heatmaps to detect feature relationships and spatial distributions of UHI intensity.
![image](https://github.com/user-attachments/assets/11401b12-cdcf-4702-986d-015c2093c8e9)

### 2️⃣ **Feature Engineering**
- **Normalization (StandardScaler):** Ensured consistent scales across all temperature and spectral data.
- **Derived Indices:**
  - **NVBI (Normalized Vegetation Built-up Index):** Combined Sentinel-2 bands (B01, B06, B08) and Landsat LST to quantify vegetation versus built-up area impact.
  - **NDVI (Vegetation Cover Index):** Indicates vegetation density and cooling potential.
  - **LST Anomalies:** Identified regions experiencing unusual heat relative to baseline.
  - **Building Density:** Calculates the number of buildings of each geographical point within a predefined radius
  - **Temperature Gradent:** Measures the intensity of heat retention at a given location with the difference between the land surface temperature and the air temperature at the surface.

### 3️⃣ **Machine Learning Models**
- **Random Forest (final model)** 🌳:
  - Robust to data noise and non-linear interactions.
  - Provided clear feature importance insights.

- **XGBoost** ⚡:
  - Optimized tree-based method.
  - Efficient handling of sparse data and robust regularization to prevent overfitting.

- **Multi-Layer Perceptrons (MLPs)**
  - Implemented deep feedforward networks with varying depths and widths.
  - Despite extensive hyperparameter tuning (e.g., layer sizes, dropout rates, learning rates), MLPs underperformed.
  - The limited dataset size and lack of sufficient training examples made high-capacity networks ineffective.

- **Convolutional Neural Networks (CNNs)** 🛰️:
  - Treated hexagonal grids as structured data for spatial feature extraction.
  - The irregular geometry of hex grids and dataset sparsity introduced challenges.
  - Standard CNNs struggled without extensive preprocessing or graph-based modeling techniques.

- **Long Short-Term Memory (LSTM) Networks and Temporal Models** ⏳:
  - Explored for potential sequential dependencies in spatial grids ordered by proximity.
  - Did not yield substantial improvements due to the dataset's lack of temporal dynamics (single-day snapshot).

- **Vision Transformers (ViT)** 🎭 :
  - Conducted preliminary experiments using ViTs with grid structures as image-like inputs.
  - The small dataset size and high sparsity led to poor generalization and unstable training.
    Required significantly more data and specialized augmentation techniques to be effective.

    
### 4️⃣ **Hyperparameter Tuning & Model Selection**
We rigorously tested multiple hyperparameter settings using cross-validation:

| Model           | Parameters                                 | Features Selected                                         | Pearson Correlation |
|-----------------|--------------------------------------------|-----------------------------------------------------------|---------------------|
| Random Forest 🌳 | n_estimators=100, random_state=42          | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly             | 0.7429              |
| Random Forest 🌳 | n_estimators=500, random_state=42          | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly             | 0.7476              |
| Random Forest 🌳 | n_estimators=1000, random_state=42   | B01, B06, B08, NDVI, Landsat_LST, lst_anomaly       | 0.7489       |
| Random Forest 🌳 | n_estimators=500, random_state=42   | B01, B04, B08, NDVI, Landsat_LST, lst_anomaly, building_density      | 0.7816      |
| Random Forest 🌳 | n_estimators=500, random_state=42   | B01, B04, B08, NDVI, Landsat_LST, lst_anomaly, building_density, temp_gradient       | 0.8037       |

---

## 📈 Evaluation Metrics
Our final model evaluation included:

- **R-squared**
- **Pearson Correlation Coefficient**

---

## 🔗 Quick Start & Usage

🛠️ Installation


```bash
pip install -r requirements.txt
```

🚦 Running Notebooks

Navigate to the notebooks directory and launch Jupyter:

```bash
cd notebooks
jupyter notebook
```

Take a look at the notebooks section for example usage of the dataset and models, and try swapping the models and and datas listed in the corresponding section to see additional results. 

Explore our preprocessing, feature engineering, and modeling notebooks directly!

---

## 🤝 Connect With Us
We're excited to collaborate and exchange ideas:
- **Yixin Huang**: [LinkedIn](https://www.linkedin.com/in/yixin-huang-91b7781aa/)
- **Zhenzhen Wu**: [LinkedIn](https://www.linkedin.com/in/zhenzhen-wu-48925922b/)

Together, let's leverage AI to tackle urban heat islands and build a sustainable future! 🌍🌳✨

---

## 📜 Data Sources & Licensing
- **Urban Temperature Data** ([Apache 2.0 License](https://github.com/CenterForOpenScience/cos.io/blob/master/LICENSE))
- **Sentinel-2 Satellite Data** ([CC BY-SA 3.0 License](https://creativecommons.org/licenses/by-sa/3.0/igo/))
- **NYC Building Footprint Data** ([Apache 2.0 License](https://github.com/CityOfNewYork/nyc-geo-metadata#Apache-2.0-1-ov-file))
- **Weather Data** ([NYS Mesonet Data Policy](https://nysmesonet.org/documents/NYS_Mesonet_Data_Access_Policy.pdf))

---


