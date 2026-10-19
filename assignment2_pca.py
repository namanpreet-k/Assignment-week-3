
# Assignment 2: PCA on the Iris Dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
actual_species = iris.target

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA to reduce 4 features to 2
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Step 4: Display the results
print("PCA Results")
print("Original dataset shape:", X.shape)
print("Reduced dataset shape:", X_reduced.shape)

variance = pca.explained_variance_ratio_.sum() * 100
print("Variance explained by 2 components:",
      round(variance, 2), "%")

# Step 5: Visualize the reduced data
plt.figure(figsize=(8, 5))
plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=actual_species,
    cmap="viridis",
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Dataset After PCA")
plt.colorbar(label="Actual Species (0, 1, 2)")
plt.tight_layout()

# Step 6: Save and display the graph
plt.savefig("pca_iris.png", dpi=150)
plt.show()

print("Graph saved as pca_iris.png")
