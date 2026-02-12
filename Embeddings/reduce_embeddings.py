import pandas as pd
from sklearn.decomposition import PCA

# Load 64d file
data = pd.read_csv("machine_learning_features_node2vec64_features.csv")

# Select embedding columns
embedding_cols = [col for col in data.columns if col.startswith("node2vec_")]

# Apply PCA
pca = PCA(n_components=5, random_state=42)
reduced_embeddings = pca.fit_transform(data[embedding_cols])

# Create new dataframe
reduced_df = pd.DataFrame(reduced_embeddings,
                          columns=[f"node2vec_{i}" for i in range(5)])

# Add rating + sentiments if needed
reduced_df["rating"] = data["rating"]

if "avg_sentiment" in data.columns:
    reduced_df["avg_sentiment"] = data["avg_sentiment"]
if "min_sentiment" in data.columns:
    reduced_df["min_sentiment"] = data["min_sentiment"]
if "max_sentiment" in data.columns:
    reduced_df["max_sentiment"] = data["max_sentiment"]

# Save new file
reduced_df.to_csv("machine_learning_features_node2vec5_from64.csv", index=False)

print("Reduced 64d → 5d successfully.")

