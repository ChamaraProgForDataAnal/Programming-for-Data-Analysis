import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Custom CSS for styling
def set_custom_style():
    st.markdown(
        """
        <style>
        /* General background styling */
        .stApp {
            background-color:rgb(251, 232, 232);
            color:rgb(44, 42, 42);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color:rgb(160, 69, 69);
        }
        [data-testid="stSidebar"] .css-1v3fvcr {
            color: white;
        }
        [data-testid="stSidebar"] .css-1n76uvr {
            color: white;
        }

        /* Header styling */
        [data-testid="stHeader"] {
            background-color:rgb(160, 69, 69);;
            color: white;
        }

        /* Dataframe styling */
        .stDataFrame {
            background-color:rgb(243, 222, 222);
            color:  #1E1E1E;
        }

        /* Buttons styling */
        .stButton > button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 5px;
            padding: 5px 10px;
        }

        /* Titles and headings */
        h1, h2, h3, h4, h5, h6, p {
            color:rgb(86, 7, 7);
        }

        /* Tables */
        table {
            background-color: #1E1E1E;
            color: white;
        }

        /* Hover effects for interactivity */
        button:hover {
            background-color: #ff7878 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


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

# PCA for Dimensionality Reduction
def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)
    return pca_result, pca.explained_variance_ratio_

# Visualize PCA results
def visualize_pca(pca_result, y):
    st.markdown("### PCA Visualization")
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['AQI_Status'] = y.values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='AQI_Status', palette='Set1', alpha=0.7)
    plt.title("PCA Results")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)

# Set page configuration
st.set_page_config(page_title="Beijing Air Quality Analysis", layout="wide")

def main():
    # Sidebar navigation

        # Set custom styles
    set_custom_style()

    st.sidebar.title("Navigation")
    options = [
        "Home",
        "Data Overview",
        "Exploratory Data Analysis",
        "Modeling and Prediction",
        "Advanced Analysis",
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
        st.image("https://images.unsplash.com/photo-1643902433280-8f8807b7f743?w=600&auto=format&fit=crop&q=60", use_container_width=True)

    elif choice == "Data Overview":
        st.title("Data Overview")
        st.markdown("---")

        # Dataset Summary
        st.subheader("📋 Dataset Summary")
        st.write("A quick overview of the dataset:")
        summary = raw_data.describe(include="all").transpose()
        st.dataframe(summary.style.background_gradient(cmap="Blues"))

        # Sample Data
        st.subheader("🔍 Sample Data")
        st.write("Here is a preview of the dataset:")
        st.dataframe(raw_data.head(), use_container_width=True)

        # Missing Values
        st.subheader("❓ Missing Values Analysis")
        st.write("Visualizing missing values in the dataset:")
        missing_values = raw_data.isnull().sum()
        st.bar_chart(missing_values)
        
        # Percentage of missing values
        missing_percent = (missing_values / len(raw_data)) * 100
        st.write("Percentage of Missing Values:")
        # Convert Series to DataFrame
        missing_percent_df = missing_percent.to_frame(name="Missing Percentage")

        # Apply background gradient and display
        st.dataframe(missing_percent_df.style.background_gradient(cmap="Reds"))


        # Column-wise Value Counts
        st.subheader("🔢 Column-Wise Value Counts")
        selected_column = st.selectbox("Select a column to view value counts:", raw_data.columns)
        if raw_data[selected_column].dtype == "object" or len(raw_data[selected_column].unique()) < 20:
            value_counts = raw_data[selected_column].value_counts()
            st.write(value_counts)
            fig, ax = plt.subplots()
            value_counts.plot(kind="bar", ax=ax, color="orange")
            ax.set_title(f"Value Counts for {selected_column}")
            st.pyplot(fig)
        else:
            st.warning("Selected column has too many unique values to display value counts.")

        # Summary of Categorical and Numerical Features
        st.subheader("📊 Feature Type Summary")
        num_cols = raw_data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = raw_data.select_dtypes(include=["object"]).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numerical Columns:**")
            st.write(num_cols)

        with col2:
            st.write("**Categorical Columns:**")
            st.write(cat_cols)

        # Correlation Matrix
        st.subheader("🔗 Correlation Matrix (Numerical Features)")
        corr_matrix = raw_data[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

        # Dataset Download
        st.subheader("💾 Download Processed Dataset")
        csv = raw_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_dataset.csv",
            mime="text/csv"
        )

    elif choice == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis")
        st.markdown("---")

        # Numeric Data Extraction
        numeric_data = raw_data.select_dtypes(include=[np.number])

        # Correlation Heatmap
        st.subheader("🔗 Correlation Heatmap")
        st.write("Understanding relationships between numerical features:")
        corr = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f", linewidths=0.5)
        ax.set_title("Correlation Heatmap", fontsize=16)
        st.pyplot(fig)

        # PM2.5 Distribution by Station
        st.subheader("📊 PM2.5 Distribution by Station")
        st.write("Visualizing PM2.5 levels across different stations:")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x="station", y="PM2.5", data=raw_data, ax=ax, palette="viridis")
        ax.set_title("PM2.5 Levels by Station", fontsize=16)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Time-Series Analysis for PM2.5
        st.subheader("📈 Time-Series Analysis of PM2.5")
        st.write("Analyzing PM2.5 trends over time:")
        if 'datetime' not in raw_data.columns:
            raw_data['datetime'] = pd.to_datetime(raw_data[['year', 'month', 'day', 'hour']])
        raw_data.set_index('datetime', inplace=True)
        daily_avg_pm25 = raw_data['PM2.5'].resample('D').mean()

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(daily_avg_pm25, color="blue", alpha=0.7)
        ax.set_title("Daily PM2.5 Concentration Over Time", fontsize=16)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=14)
        st.pyplot(fig)

        # Monthly PM2.5 Trends
        st.subheader("📅 Monthly PM2.5 Trends")
        st.write("Aggregating PM2.5 levels by month:")
        raw_data['month'] = raw_data.index.month
        monthly_avg_pm25 = raw_data.groupby('month')['PM2.5'].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=monthly_avg_pm25.index, y=monthly_avg_pm25.values, ax=ax, palette="Blues_d")
        ax.set_title("Average PM2.5 Levels by Month", fontsize=16)
        ax.set_xlabel("Month", fontsize=14)
        ax.set_ylabel("Average PM2.5 (µg/m³)", fontsize=14)
        st.pyplot(fig)

        # PM2.5 Distribution Histogram
        st.subheader("📊 PM2.5 Distribution Histogram")
        st.write("Understanding the distribution of PM2.5 levels:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(raw_data['PM2.5'], bins=50, kde=True, color="orange", ax=ax)
        ax.set_title("PM2.5 Concentration Distribution", fontsize=16)
        ax.set_xlabel("PM2.5 (µg/m³)", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        st.pyplot(fig)

        # Scatter Plot: PM2.5 vs Temperature
        st.subheader("🌡️ PM2.5 vs Temperature")
        st.write("Exploring the relationship between PM2.5 levels and temperature:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=raw_data['TEMP'], y=raw_data['PM2.5'], alpha=0.7, color="green", ax=ax)
        ax.set_title("Scatter Plot: PM2.5 vs Temperature", fontsize=16)
        ax.set_xlabel("Temperature (°C)", fontsize=14)
        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=14)
        st.pyplot(fig)

        # Scatter Plot: PM2.5 vs Wind Speed
        st.subheader("🌬️ PM2.5 vs Wind Speed")
        st.write("Analyzing the impact of wind speed on PM2.5 levels:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=raw_data['WSPM'], y=raw_data['PM2.5'], alpha=0.7, color="purple", ax=ax)
        ax.set_title("Scatter Plot: PM2.5 vs Wind Speed", fontsize=16)
        ax.set_xlabel("Wind Speed (m/s)", fontsize=14)
        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=14)
        st.pyplot(fig)

        # Station-Wise PM2.5 Analysis
        st.subheader("🏭 Station-Wise PM2.5 Analysis")
        selected_station = st.selectbox("Select a station to view its PM2.5 trend:", raw_data['station'].unique())
        station_data = raw_data[raw_data['station'] == selected_station]
        station_daily_avg = station_data['PM2.5'].resample('D').mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(station_daily_avg, color="red", alpha=0.7)
        ax.set_title(f"Daily PM2.5 Levels at Station {selected_station}", fontsize=16)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=14)
        st.pyplot(fig)

        st.markdown(
            "These visualizations provide insights into the patterns and relationships between PM2.5 levels and various environmental factors, helping to better understand air quality trends."
        )

    elif choice == "Modeling and Prediction":
        st.title("Modeling and Prediction")
        st.markdown("---")

        # Preprocessing the data
        X_scaled, y = preprocess_data(raw_data)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Available models
        st.subheader("🔍 Select Models to Train")
        model_options = st.multiselect(
            "Choose models to train:",
            ["k-Nearest Neighbors", "Naive Bayes", "Logistic Regression", "Random Forest"],
            default=["k-Nearest Neighbors", "Logistic Regression"]
        )

        # Define models
        models = {
            "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=st.sidebar.slider("k-Nearest Neighbors (n_neighbors)", 1, 15, 5)),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=st.sidebar.slider("Logistic Regression (max_iter)", 100, 1000, 500)),
            "Random Forest": RandomForestClassifier(
                n_estimators=st.sidebar.slider("Random Forest (n_estimators)", 50, 300, 100),
                random_state=42
            )
        }

        # Train selected models
        results = {}
        reports = {}
        confusion_matrices = {}

        if model_options:
            for name in model_options:
                st.write(f"### Training {name}")
                model = models[name]
                model.fit(X_train, y_train)

                # Predictions and evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = accuracy
                reports[name] = classification_report(y_test, y_pred, output_dict=True)
                confusion_matrices[name] = confusion_matrix(y_test, y_pred)

                # Display results
                st.write(f"**Accuracy**: {accuracy:.2f}")
                st.write("#### Classification Report")
                st.dataframe(pd.DataFrame(reports[name]).transpose())

                # Confusion Matrix
                st.write("#### Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrices[name], annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique(), ax=ax)
                ax.set_title(f"Confusion Matrix: {name}", fontsize=14)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

        # Compare model performances
        st.subheader("📊 Model Performance Comparison")
        if results:
            # Accuracy comparison
            accuracy_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Accuracy", y="Model", data=accuracy_df, palette="viridis", ax=ax)
            ax.set_title("Model Accuracy Comparison", fontsize=16)
            st.pyplot(fig)

            # Additional insights on performance
            st.write("### Insights")
            best_model = max(results, key=results.get)
            st.write(f"**Best Performing Model**: {best_model} with an accuracy of {results[best_model]:.2f}")
        else:
            st.warning("No models selected for training.")

    elif choice == "Advanced Analysis":
        st.title("Advanced Analysis")
        st.markdown("---")

        st.subheader("🔍 Principal Component Analysis (PCA)")
        st.write("Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset while retaining the most critical information.")

        # Preprocess data for PCA
        X_scaled, y = preprocess_data(raw_data)

        # User input for PCA
        st.markdown("### Configure PCA")
        n_components = st.slider("Select the number of components for PCA", min_value=2, max_value=min(X_scaled.shape[1], 10), value=2, step=1)

        # Apply PCA
        pca_result, explained_variance = apply_pca(X_scaled, n_components=n_components)

        # Display PCA results
        st.markdown("### PCA Visualization")
        visualize_pca(pca_result, pd.Series(y))

        st.write("### Explained Variance Ratio")
        for i, variance in enumerate(explained_variance):
            st.write(f"Principal Component {i + 1}: {variance:.2%}")

        # Cumulative variance
        cumulative_variance = np.cumsum(explained_variance)
        st.write("### Cumulative Explained Variance")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="--", color="b")
        ax.set_title("Cumulative Explained Variance by Principal Components", fontsize=16)
        ax.set_xlabel("Number of Components", fontsize=14)
        ax.set_ylabel("Cumulative Variance Explained", fontsize=14)
        ax.grid()
        st.pyplot(fig)

        # Visualizing the feature contributions to PCs
        st.write("### Feature Contributions to Principal Components")
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        feature_contributions = pd.DataFrame(pca.components_, columns=raw_data.drop(columns=['AQI_Status', 'AQI']).columns)
        st.dataframe(feature_contributions.T)

        # Advanced Clustering Analysis (optional)
        st.markdown("### Clustering with PCA")
        if st.checkbox("Enable KMeans Clustering"):
            from sklearn.cluster import KMeans

            # User input for number of clusters
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3, step=1)

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(pca_result)

            # Visualize clustering results
            st.markdown("#### PCA with Clustering")
            pca_df = pd.DataFrame(pca_result, columns=[f"PC{i + 1}" for i in range(n_components)])
            pca_df["Cluster"] = cluster_labels
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", alpha=0.7)
            plt.title("PCA with Clustering", fontsize=16)
            plt.xlabel("Principal Component 1", fontsize=14)
            plt.ylabel("Principal Component 2", fontsize=14)
            plt.legend(title="Cluster", loc="upper right")
            st.pyplot(plt)

            # Display cluster centers
            st.markdown("#### Cluster Centers (Centroids)")
            st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=[f"PC{i + 1}" for i in range(n_components)]))

    elif choice == "About the Project":
        st.title("About the Project")
        st.markdown("---")

        # Project Introduction
        st.subheader("📄 Project Overview")
        st.write(
            """
            This project leverages the Beijing Multi-Site Air Quality dataset to analyze air pollution trends, 
            build machine learning models for prediction, and provide an interactive platform for data exploration 
            and visualization. The primary focus is on understanding **PM2.5 concentrations** and their relationship 
            with other environmental factors.
            """
        )

        # Dataset Details
        st.subheader("📊 Dataset Details")
        st.markdown(
            """
            - **Source**: Beijing Environmental Monitoring Center  
            - **Time Period**: March 1, 2013, to February 28, 2017  
            - **Data Includes**:
                - Air pollutant levels: PM2.5, PM10, SO2, NO2, CO, O3  
                - Meteorological data: Temperature, Pressure, Wind Speed, Rainfall  
            - **Stations**: Data collected from **12 monitoring stations** across Beijing  
            """
        )

        # Features of the Application
        st.subheader("✨ Key Features of the Application")
        st.write(
            """
            - **Data Overview**: Comprehensive summary of the dataset, including missing value analysis.
            - **Exploratory Data Analysis (EDA)**: Visualizations like correlation heatmaps, scatter plots, and time-series trends.
            - **Modeling & Prediction**: Build and evaluate various machine learning models to predict air quality.
            - **Advanced Analysis**: Dimensionality reduction using PCA and clustering analysis.
            """
        )

        # Use Cases Section
        st.subheader("🚀 Use Cases")
        st.write(
            """
            - **Environmental Policy Making**: Insights for air quality improvement measures.
            - **Public Awareness**: Educating citizens about air quality trends and their health impact.
            - **Data Science Projects**: Showcase for advanced machine learning and data visualization techniques.
            """
        )

        # Interactive Fun Fact Section
        st.subheader("🌍 Fun Fact")
        if st.button("Why is PM2.5 important?"):
            st.write(
                """
                PM2.5 particles are fine particulate matter with a diameter of less than 2.5 micrometers.  
                - **Health Impact**: These particles can penetrate deep into the lungs and even enter the bloodstream, 
                causing serious respiratory and cardiovascular health issues.
                - **Environmental Impact**: PM2.5 significantly reduces visibility and contributes to haze.
                """
            )

        # Highlighted Image
        st.image(
            "https://images.unsplash.com/photo-1611273426858-450d8e3c9fce?w=600&auto=format&fit=crop&q=60",
            caption="Smog in Beijing. Source: Unsplash",
            use_container_width=True,
        )

        # Project Goals Section
        st.subheader("🎯 Project Goals")
        st.markdown(
            """
            1. Analyze pollution trends across Beijing from 2013 to 2017.
            2. Build and evaluate predictive models for air quality forecasting.
            3. Provide an interactive, user-friendly platform for data exploration and model analysis.
            """
        )

        # Acknowledgements Section
        st.subheader("🤝 Acknowledgements")
        st.write(
            """
            - **Dataset Source**: Beijing Environmental Monitoring Center  
            - **Libraries Used**: Streamlit, Pandas, Scikit-learn, Seaborn, Matplotlib  
            """
        )

        # User Feedback Section
        st.subheader("📝 Your Feedback")
        st.text_area("We value your feedback! Share your thoughts or suggestions below:")

        # Social Sharing Section
        st.subheader("📢 Spread the Word")
        st.markdown(
            """
            If you enjoyed exploring this project, share it with your network!  
            [GitHub Repository](https://github.com/) | [Twitter](https://twitter.com) | [LinkedIn](https://linkedin.com)
            """
        )

if __name__ == "__main__":
    main()
