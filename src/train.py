import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load data
df = pd.read_csv("../data/Mall_Customers.csv")


# Features used for clustering
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train K-Means
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)


# Save scaler and model
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(kmeans, "../models/kmeans.pkl")

print("Model and scaler saved successfully.")