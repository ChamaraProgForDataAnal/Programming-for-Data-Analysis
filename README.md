# Beijing Multi-Site Air Quality Analysis

This project involves the analysis and prediction of air quality using the Beijing Multi-Site Air Quality dataset. The dataset, sourced from the UCI Machine Learning Repository, includes pollutant concentrations and meteorological data recorded over several years. The project explores the impact of various factors on air quality, builds machine learning models, and provides an interactive application for exploration and prediction.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Analysis Workflow](#analysis-workflow)
  - [Task 1: Data Handling](#task-1-data-handling)
  - [Task 2: Exploratory Data Analysis (EDA)](#task-2-exploratory-data-analysis-eda)
  - [Task 3: Model Building](#task-3-model-building)
  - [Task 4: Application Development](#task-4-application-development)
- [Challenges and Learnings](#challenges-and-learnings)
- [Technologies Used](#technologies-used)
- [How to Run the Application](#how-to-run-the-application)
- [References](#references)

---

## Dataset Overview

The dataset originates from the Beijing Municipal Environmental Monitoring Center and contains hourly readings from 12 monitoring stations over four years. Key features include:

- **Pollutants:** PM2.5, PM10, SO2, NO2, CO, O3
- **Meteorological Parameters:** Temperature (TEMP), Pressure (PRES), Dew Point (DEWP), Wind Speed (WSPM), Wind Direction (WD), Rainfall (RAIN)

---

## Analysis Workflow

### Task 1: Data Handling

- **Steps:**
  - Downloaded and imported data using Python libraries.
  - Merged pollutant and meteorological datasets by aligning timestamps and stations.
  - Imputed missing values using station-wise and month-wise averages.
  - Addressed mismatched timestamps and optimized processing for large datasets.

---

### Task 2: Exploratory Data Analysis (EDA)

- **Steps:**
  - Preprocessed data by imputing missing values, removing duplicates, and engineering features (e.g., AQI calculation).
  - Conducted univariate, bivariate, and multivariate analyses.
  - Visualizations included histograms, scatter plots, and correlation heatmaps.
  - Principal Component Analysis (PCA) reduced dimensionality while retaining critical information.

---

### Task 3: Model Building

- **Steps:**
  - Preprocessed the data using StandardScaler and LabelEncoder.
  - Built three models:
    - k-Nearest Neighbors (k-NN)
    - Naive Bayes (GaussianNB)
    - Logistic Regression
  - Hyperparameters were tuned using GridSearchCV.
  - Logistic Regression achieved the highest accuracy (99.99%) across all AQI categories.

---

### Task 4: Application Development

- **Features:**
  - **Data Overview:** Summarized dataset and pollutant distributions.
  - **EDA:** Provided interactive visualizations like histograms, scatter plots, and heatmaps.
  - **Modeling and Prediction:** Allowed users to predict AQI categories based on input conditions.
- **Technology:** Developed with Python and Streamlit, with GitHub for version control.

---

## Challenges and Learnings

### Challenges

- Managing large datasets and handling missing data.
- Addressing class imbalance in AQI categories.
- Integrating machine learning models into an interactive GUI.

### Learnings

- Advanced skills in data cleaning, feature engineering, and hyperparameter tuning.
- Experience in creating user-centric applications for data exploration and prediction.

---

## Technologies Used

- **Programming Languages:** Python
- **Libraries and Tools:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Streamlit, Joblib
- **Version Control:** GitHub

---

## How to Run the Application

1. Clone the repository:
   ```bash
   [https://github.com/ChamaraProgForDataAnal/Programming-for-Data-Analysis.git]
2. Navigate to the project directory:
  cd beijing-air-quality-analysis
3. Install dependencies:
  pip install -r requirements.txt
4. Run the Streamlit application:
  streamlit run app.py
