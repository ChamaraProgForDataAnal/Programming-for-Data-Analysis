import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    # Load the data from individual files
    file_list = [
        "PRSA_Data_Guanyuan_20130301-20170228.csv",
        "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
        "PRSA_Data_Wanliu_20130301-20170228.csv",
        "PRSA_Data_Tiantan_20130301-20170228.csv",
        "PRSA_Data_Wanshouxigong_20130301-20170228.csv",
        "PRSA_Data_Nongzhanguan_20130301-20170228.csv",
        "PRSA_Data_Shunyi_20130301-20170228.csv",
        "PRSA_Data_Changping_20130301-20170228.csv",
        "PRSA_Data_Dingling_20130301-20170228.csv",
        "PRSA_Data_Huairou_20130301-20170228.csv",
        "PRSA_Data_Gucheng_20130301-20170228.csv",
        "PRSA_Data_Dongsi_20130301-20170228.csv"
    ]

    dataframes = []
    for file in file_list:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            st.warning(f"Error loading file {file}: {e}")

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Handling missing values
    columns_with_nulls = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for column in columns_with_nulls:
        combined_df[column] = combined_df.groupby(['year', 'month', 'station'])[column].transform(lambda x: x.fillna(x.mean()))

    if 'wd' in combined_df.columns:
        combined_df['wd'] = combined_df.groupby(['year', 'month', 'station'])['wd'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))

    # Calculate AQI using PM2.5 as an example
    def calculate_aqi_pm25(pm25):
        if pm25 <= 35:
            return pm25 * (50 / 35)
        elif pm25 <= 75:
            return ((pm25 - 35) * (50 / 40)) + 50
        elif pm25 <= 115:
            return ((pm25 - 75) * (50 / 40)) + 100
        elif pm25 <= 150:
            return ((pm25 - 115) * (50 / 35)) + 150
        elif pm25 <= 250:
            return ((pm25 - 150) * (100 / 100)) + 200
        elif pm25 <= 350:
            return ((pm25 - 250) * (100 / 100)) + 300
        else:
            return ((pm25 - 350) * (100 / 150)) + 400

    combined_df['AQI'] = combined_df['PM2.5'].apply(calculate_aqi_pm25)

    # Assign AQI status
    def assign_aqi_status(aqi):
        if aqi < 51:
            return 'Good'
        elif aqi < 101:
            return 'Satisfactory'
        elif aqi < 201:
            return 'Moderate'
        elif aqi < 301:
            return 'Poor'
        elif aqi < 401:
            return 'Very Poor'
        else:
            return 'Severe'

    combined_df['AQI_Status'] = combined_df['AQI'].apply(assign_aqi_status)

    return combined_df

# Preprocessing function
def preprocess_data(data):
    # Label encoding for categorical columns
    le = LabelEncoder()
    data['station'] = le.fit_transform(data['station'])
    data['wd'] = le.fit_transform(data['wd'])

    # Define features and target
    X = data.drop(columns=['AQI_Status', 'AQI'])
    y = data['AQI_Status']

    # Handle remaining NaN values in X
    for column in X.columns:
        if X[column].isnull().sum() > 0:
            if X[column].dtype == 'object':
                X[column].fillna(X[column].mode()[0], inplace=True)
            else:
                X[column].fillna(X[column].mean(), inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
# Set page configuration
st.set_page_config(page_title="Beijing Air Quality Analysis", layout="wide")

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    options = [
        "Home",
        "Data Overview",
        "Exploratory Data Analysis",
        "Modeling and Prediction",
        "About the Project"
    ]
    choice = st.sidebar.radio("Go to", options)

    # Load and preprocess data
    raw_data = load_and_preprocess_data()

    if choice == "Home":
        st.title("Welcome to Beijing Air Quality Analysis")
        st.markdown("---")
        st.write(
            "This application allows you to explore, analyze, and model air quality data from Beijing. "
            "Use the navigation panel on the left to switch between sections."
        )
        st.image("https://images.unsplash.com/photo-1643902433280-8f8807b7f743?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fGFpciUyMHBvbGx1dGlvbnxlbnwwfHwwfHx8MA%3D%3D", use_container_width =True)

    elif choice == "Data Overview":
        st.title("Data Overview")
        st.markdown("---")
        st.write("### Dataset Summary")
        st.write(raw_data.describe())

        st.write("### Sample Data")
        st.dataframe(raw_data.head())

        st.write("### Missing Values")
        st.bar_chart(raw_data.isnull().sum())

    elif choice == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis")
        st.markdown("---")

        numeric_data = raw_data.select_dtypes(include=[np.number])

        st.write("### Correlation Heatmap")
        corr = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("### PM2.5 Distribution by Station")
        if 'PM2.5' in raw_data.columns and 'station' in raw_data.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(x="station", y="PM2.5", data=raw_data, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("Required columns 'PM2.5' or 'station' not found in the dataset.")

    elif choice == "Modeling and Prediction":
            st.title("Modeling and Prediction")
            st.markdown("---")

            X_scaled, y = preprocess_data(raw_data)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            models = {
                "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                "Naive Bayes": GaussianNB(),
                "Logistic Regression": LogisticRegression(max_iter=1000)
            }

            results = {}
            reports = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = accuracy
                reports[name] = classification_report(y_test, y_pred, output_dict=True)

                st.write(f"### {name}")
                st.write(f"Accuracy: {accuracy:.2f}")

            st.write("### Model Comparison")
            fig, ax = plt.subplots()
            ax.bar(results.keys(), results.values(), color="skyblue")
            ax.set_title("Accuracy of Different Models")
            ax.set_ylabel("Accuracy")
            st.pyplot(fig)

            st.write("### Classification Reports")
            for name, report in reports.items():
                st.write(f"#### {name}")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

    elif choice == "About the Project":
        st.title("About the Project")
        st.markdown("---")
        st.write(
            "This project leverages the Beijing Multi-Site Air Quality dataset to analyze pollution trends, "
            "build machine learning models for prediction, and provide an interactive platform for exploration."
        )
        st.image("https://images.unsplash.com/photo-1611273426858-450d8e3c9fce?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YWlyJTIwcG9sbHV0aW9ufGVufDB8fDB8fHww", use_container_width =True)

if __name__ == "__main__":
    main()