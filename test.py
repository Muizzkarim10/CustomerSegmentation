import joblib
import pandas as pd

scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")

new_customer = pd.DataFrame({
    "Age": [30],
    "Annual Income (k$)": [85],
    "Spending Score (1-100)": [80]
})

scaled_customer = scaler.transform(new_customer)

cluster = kmeans.predict(scaled_customer)

print("Predicted Cluster:", cluster[0])