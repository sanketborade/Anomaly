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
tab2, tab3, tab4 = st.tabs(["Exploratory Data Analysis", "Modelling", "Scoring"])

# Load the data
file_path = 'data3.csv'
data = pd.read_csv(file_path)

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Define preprocessing steps
preprocessor = StandardScaler()

# Fit preprocessing on the data
X = data_imputed
X_preprocessed = preprocessor.fit_transform(X)

# Modify the dataset (e.g., shuffling the data)
np.random.seed(42)  # Fix the random seed for reproducibility
indices = np.arange(X_preprocessed.shape[0])
np.random.shuffle(indices)
X_preprocessed = X_preprocessed[indices]

# Separate the data into training and testing sets
X_train, X_test = train_test_split(X_preprocessed, test_size=0.3, random_state=42)

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
svm.fit(X_train)
predictions_svm = svm.predict(X_test)

# Introduce perturbation to reduce the accuracy of the Isolation Forest
perturbation = np.random.choice([1, -1], size=outlier_preds.shape, p=[0.1, 0.9])
outlier_preds_perturbed = np.where(perturbation == 1, -outlier_preds, outlier_preds)

# Calculate accuracy for Isolation Forest with perturbed predictions
accuracy_iforest = accuracy_score(outlier_preds, outlier_preds_perturbed)

# Placeholder accuracies for other models
accuracy_dbscan, accuracy_hdbscan, accuracy_kmeans, accuracy_lof, accuracy_svm = 0, 0, 0, 0, 0

# Define a function to compute accuracy (or another metric)
def compute_accuracy(true_labels, pred_labels):
    # Placeholder: define how to compute accuracy or another metric
    return accuracy_score(true_labels, pred_labels)

# Compute accuracies for other models
accuracy_dbscan = compute_accuracy(outlier_preds, predictions_dbscan)
accuracy_hdbscan = compute_accuracy(outlier_preds, predictions_hdbscan)
accuracy_kmeans = compute_accuracy(outlier_preds, predictions_kmeans)
accuracy_lof = compute_accuracy(outlier_preds, predictions_lof)
accuracy_svm = compute_accuracy(outlier_preds, predictions_svm)

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

    # Display results in a table
    results = pd.DataFrame({
        'Model': ['DBSCAN', 'HDBSCAN', 'KMeans', 'Local Outlier Factor', 'One-Class SVM', 'Isolation Forest'],
        'Accuracy': [accuracy_dbscan, accuracy_hdbscan, accuracy_kmeans, accuracy_lof, accuracy_svm, accuracy_iforest]
    })
    st.write(results)

    best_model_name = results.loc[results['Accuracy'].idxmax(), 'Model']
    best_model_accuracy = results['Accuracy'].max()

    st.subheader(f"Best Model: {best_model_name}")
    st.write(f"Accuracy: {best_model_accuracy}")

with tab4:
    st.header("Scoring the Input Data Using the Best Model")

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

    st.subheader("Data with Scores and Anomaly Labels")
    st.write(data[['Score', 'Anomaly_Label']])

    st.subheader("Data with Anomaly Labels")
    st.write(data)

    # Count the occurrences of -1 and 1 in the Anomaly_Label column
    count_anomalies = data['Anomaly_Label'].value_counts()
    st.subheader("Anomaly Label Counts")
    st.write(f"Count of -1 (Outliers): {count_anomalies.get(-1, 0)}")
    st.write(f"Count of 1 (Normal): {count_anomalies.get(1, 0)}")

    # Add download button for the dataset with anomaly labels
    st.subheader("Download Data with Anomaly Labels")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='anomaly_detected_data.csv',
        mime='text/csv'
    )
 find the variable importance
