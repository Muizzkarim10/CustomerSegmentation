# Customer Segmentation using K-Means

An unsupervised machine learning project that uses **K-Means clustering** to segment customers based on their age, annual income, and spending score.

## 📊 Dataset

The dataset contains **200 customers** with no missing values.

Features used:

- Age
- Annual Income (k$)
- Spending Score (1-100)

## 🔧 Approach

- Exploratory Data Analysis
- Feature scaling with `StandardScaler`
- K-Means clustering
- Elbow Method
- Silhouette Score
- PCA for visualization

The Elbow Method suggested **K=4**, while the highest Silhouette Score was achieved by K=6. **K=4 was selected** because it produced simpler and more interpretable customer segments.

## 👥 Results

The four clusters identified were:

- **Cluster 0:** Older customers with moderate income and spending
- **Cluster 1:** Young, high-income, high-spending customers
- **Cluster 2:** Young customers with lower income but relatively high spending
- **Cluster 3:** High-income, low-spending customers

PCA reduced the three features to two components while retaining approximately **77.6% of the variance**.

## 📁 Project Structure

```text
CustomerSegmentation/
├── data/
├── models/
├── notebooks/
├── src/
├── .gitignore
├── ReadMe.md
└── requirements.txt
```

## 🛠️ Technologies

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter

## ▶️ Run

```bash
pip install -r requirements.txt
python src/train.py
```