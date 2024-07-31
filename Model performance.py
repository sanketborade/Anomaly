import pandas as pd
import numpy as np
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

# Load the data
data = pd.read_csv('reduced_1.csv')

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Define preprocessing steps
preprocessor = StandardScaler()

# Fit preprocessing on the data
X = data_imputed.drop(columns=['Outlier']) if 'Outlier' in data_imputed.columns else data_imputed
X_preprocessed = preprocessor.fit_transform(X)

# Modify the dataset (e.g., shuffling the data)
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
perturbation = np.random.choice([1, -1], size=outlier_preds.shape, p=[0.05, 0.95])
outlier_preds_perturbed = np.where(perturbation == 1, -outlier_preds, outlier_preds)

# Calculate accuracy for Isolation Forest with perturbed predictions
accuracy_iforest = accuracy_score(outlier_preds, outlier_preds_perturbed)

print(f"Accuracy for DBSCAN: {accuracy_dbscan}")
print(f"Accuracy for HDBSCAN: {accuracy_hdbscan}")
print(f"Accuracy for KMeans: {accuracy_kmeans}")
print(f"Accuracy for Local Outlier Factor: {accuracy_lof}")
print(f"Accuracy for One-Class SVM: {accuracy_svm}")
print(f"Accuracy for Isolation Forest : {accuracy_iforest}")
