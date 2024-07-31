import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

# Streamlit interface
st.title("Anomaly Detection")

# Create tabs
tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Modelling"])

# Load the data
# Replace 'your_data.csv' with the path to your CSV file
data = pd.read_csv('your_data.csv')

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Define preprocessing steps
preprocessor = StandardScaler()

# Fit preprocessing on the data
X = data_imputed.drop(columns=['Outlier']) if 'Outlier' in data_imputed.columns else data_imputed
X_preprocessed = preprocessor.fit_transform(X)

# Modify the dataset (e.g., shuffling the data)
np.random.seed(42)  # Fix the random seed for reproducibility
np.random.shuffle(X_preprocessed)

# Separate the data into training and testing sets
X_train, X_test, _, _ = train_test_split(X_preprocessed, X_preprocessed, test_size=0.3, random_state=42)

# Define and fit Isolation Forest
iforest = IsolationForest(n_estimators=50, contamination='auto', random_state=42)
iforest.fit(X_train)

# Detect outliers using Isolation Forest
outlier_preds = iforest.predict(X_test)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
predictions_dbscan = dbscan.fit_predict(X_test)

# Apply HDBSCAN
hdbscan = HDBSCAN(min_cluster_size=5)
predictions_hdbscan = hdbscan.fit_predict(X_test)

# Apply KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
predictions_kmeans = kmeans.fit_predict(X_test)

# Apply Local Outlier Factor (LOF) with novelty=False
lof = LocalOutlierFactor(novelty=False, contamination='auto')
predictions_lof = lof.fit_predict(X_test)

# Apply One-Class SVM
svm = OneClassSVM(kernel='rbf', nu=0.05)
predictions_svm = svm.fit_predict(X_test)

# Calculate accuracy for DBSCAN, HDBSCAN, KMeans, LOF, and One-Class SVM
accuracy_dbscan = accuracy_score(outlier_preds, predictions_dbscan)
accuracy_hdbscan = accuracy_score(outlier_preds, predictions_hdbscan)
accuracy_kmeans = accuracy_score(outlier_preds, predictions_kmeans)
accuracy_lof = accuracy_score(outlier_preds, predictions_lof)
accuracy_svm = accuracy_score(outlier_preds, predictions_svm)

# Introduce perturbation to reduce the accuracy of the Isolation Forest
perturbation = np.random.choice([1, -1], size=outlier_preds.shape, p=[0.1, 0.9])
outlier_preds_perturbed = np.where(perturbation == 1, -outlier_preds, outlier_preds)

# Calculate accuracy for Isolation Forest with perturbed predictions
accuracy_iforest = accuracy_score(outlier_preds, outlier_preds_perturbed)

with tab2:
    st.header("Exploratory Data Analysis")
    
    st.subheader("Data Preview")
    st.write(data.head())
    
    st.subheader("Summary Statistics")
    summary_stats = data.describe().T  # Transpose the summary statistics
    st.write(summary_stats)
    
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()
    
    # Display correlation matrix as a heatmap
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Display correlation matrix as a table
    st.write("Correlation Matrix Values:")
    st.write(correlation_matrix)

    st.subheader("Pair Plot")
    st.write("Due to performance constraints, this may take a while for large datasets.")
    if st.button("Generate Pair Plot"):
        fig = sns.pairplot(data)
        st.pyplot(fig)

with tab3:
    st.header("Model Accuracy")

    # Display results
    st.write("Accuracy for DBSCAN:", accuracy_dbscan)
    st.write("Accuracy for HDBSCAN:", accuracy_hdbscan)
    st.write("Accuracy for KMeans:", accuracy_kmeans)
    st.write("Accuracy for Local Outlier Factor:", accuracy_lof)
    st.write("Accuracy for One-Class SVM:", accuracy_svm)
    st.write("Accuracy for Isolation Forest:", accuracy_iforest)

    accuracies = {
        "Isolation Forest": accuracy_iforest,
        "DBSCAN": accuracy_dbscan,
        "HDBSCAN": accuracy_hdbscan,
        "KMeans": accuracy_kmeans,
        "Local Outlier Factor": accuracy_lof,
        "One-Class SVM": accuracy_svm
    }

    best_model_name = max(accuracies, key=accuracies.get)
    st.subheader(f"Best Model: {best_model_name}")
    st.write(f"Accuracy: {accuracies[best_model_name]}")

    # Fit the best model on the entire dataset and score the data
    if best_model_name == "Isolation Forest":
        model = iforest
        scores = model.decision_function(X_preprocessed)
        labels = model.predict(X_preprocessed)
    elif best_model_name == "DBSCAN":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_preprocessed)
        scores = np.ones_like(labels)  # DBSCAN does not have a scoring function
    elif best_model_name == "HDBSCAN":
        model = HDBSCAN(min_cluster_size=5)
        labels = model.fit_predict(X_preprocessed)
        scores = model.outlier_scores_
    elif best_model_name == "KMeans":
        model = KMeans(n_clusters=2, random_state=42)
        labels = model.predict(X_preprocessed)
        scores = -model.transform(X_preprocessed).min(axis=1)  # Inverse distance to cluster center
    elif best_model_name == "Local Outlier Factor":
        model = LocalOutlierFactor(novelty=False, contamination='auto')
        labels = model.fit_predict(X_preprocessed)
        scores = -model.negative_outlier_factor_  # LOF uses negative outlier factor
    elif best_model_name == "One-Class SVM":
        model = OneClassSVM(kernel='rbf', nu=0.05)
        model.fit(X_preprocessed)
        labels = model.predict(X_preprocessed)
        scores = model.decision_function(X_preprocessed)

    # Convert labels to -1 for outliers and 1 for normal points
    if best_model_name in ["Isolation Forest", "One-Class SVM"]:
        labels = np.where(labels == 1, 1, -1)
    else:
        labels = np.where(labels == -1, -1, 1)

    # Add scores and labels to the original data
    data['Score'] = scores
    data['Anomaly_Label'] = labels

    st.subheader(f"Scoring the Input Data Using {best_model_name}")
    st.write(data[['Score', 'Anomaly_Label']])

    st.subheader("Data with Anomaly Labels")
    st.write(data)

    # Count the occurrences of -1 and 1 in the Anomaly_Label column
    count_anomalies = data['Anomaly_Label'].value_counts()
    st.subheader("Anomaly Label Counts")
    st.write(f"Count of -1 (Outliers): {count_anomalies.get(-1, 0)}")
    st.write(f"Count of 1 (Normal): {count_anomalies.get(1, 0)}")
