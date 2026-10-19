
# Mini Project 3: Iris Flower Clustering Project

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
true_labels = iris.target

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means with 3 clusters
model = KMeans(n_clusters=3, random_state=42, n_init=10)
predicted_clusters = model.fit_predict(X_scaled)

# Step 4: Compare predicted clusters with actual species
results = pd.DataFrame({
    "Actual Species": [
        iris.target_names[i] for i in true_labels
    ],
    "Predicted Cluster": predicted_clusters
})

print("IRIS FLOWER CLUSTERING PROJECT")
print("\nFirst 10 results:")
print(results.head(10).to_string(index=False))

print("\nComparison: Actual Species vs Predicted Cluster")
print(pd.crosstab(
    results["Actual Species"],
    results["Predicted Cluster"]
))

# Step 5: Calculate clustering agreement
score = adjusted_rand_score(true_labels, predicted_clusters)
print("\nAdjusted Rand Index:", round(score, 3))

# Step 6: Plot the predicted clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    X[:, 2], X[:, 3],
    c=predicted_clusters,
    cmap="viridis",
    s=50
)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Predicted Iris Clusters (K-Means)")
plt.colorbar(label="Predicted Cluster")
plt.tight_layout()
plt.savefig("mini_project_clusters.png", dpi=150)
plt.show()

# Step 7: Plot the actual Iris species
plt.figure(figsize=(8, 5))
plt.scatter(
    X[:, 2], X[:, 3],
    c=true_labels,
    cmap="viridis",
    s=50
)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Actual Iris Species")
plt.colorbar(label="Actual Species (0, 1, 2)")
plt.tight_layout()
plt.savefig("actual_iris_species.png", dpi=150)
plt.show()

# Step 8: Save the comparison table
results.to_csv("cluster_comparison.csv", index=False)

print("\nProject completed successfully!")
print("Saved: mini_project_clusters.png")
print("Saved: actual_iris_species.png")
print("Saved: cluster_comparison.csv")
