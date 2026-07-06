"""
Task 15 — K-Means Clustering
Pipeline: load -> choose k -> fit -> evaluate -> profile -> name -> stability -> recommend

Dataset: sklearn's built-in Wine dataset (real chemical analysis data, 178 samples,
13 features). We deliberately ignore the true labels during clustering (unsupervised),
but keep them around only to sanity-check stability at the end.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
OUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# 0. Load & prepare data (train/val/test discipline)
# ---------------------------------------------------------------------------
data = load_wine(as_frame=True)
df = data.frame.drop(columns=["target"])   # drop label -- unsupervised task
true_labels = data.frame["target"]         # kept only for a stability sanity-check later

# Split off a held-out test set. For clustering there's no "y" to fit against,
# but holding out data still lets you check that cluster structure generalizes.
X_train, X_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

# ---------------------------------------------------------------------------
# 1 & 2. Run K-Means across candidate k values; evaluate with elbow + silhouette
# ---------------------------------------------------------------------------
k_range = range(2, 9)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_train_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_train_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow Method (Inertia)")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_range), sil_scores, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/k_selection.png", dpi=150)
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"Silhouette scores by k: {dict(zip(k_range, [round(s,3) for s in sil_scores]))}")
print(f"Chosen k (highest silhouette): {best_k}")

# ---------------------------------------------------------------------------
# 3. Fit final model at chosen k
# ---------------------------------------------------------------------------
final_km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
train_clusters = final_km.fit_predict(X_train_scaled)
test_clusters = final_km.predict(X_test_scaled)

final_sil = silhouette_score(X_train_scaled, train_clusters)
final_inertia = final_km.inertia_
print(f"\nFinal model -> k={best_k}, silhouette={final_sil:.3f}, inertia={final_inertia:.1f}")

# ---------------------------------------------------------------------------
# 4. Profile each cluster's defining characteristics
# ---------------------------------------------------------------------------
profile_df = X_train.copy()
profile_df["cluster"] = train_clusters

cluster_profile = profile_df.groupby("cluster").mean()
overall_mean = X_train.mean()

# z-score each cluster's feature means against the overall mean so you can see
# what's distinctively HIGH or LOW per cluster, not just raw averages
cluster_profile_z = (cluster_profile - overall_mean) / X_train.std()
print("\nCluster profile (z-scores vs overall mean):")
print(cluster_profile_z.round(2))

plt.figure(figsize=(10, 5))
sns.heatmap(cluster_profile_z, annot=True, fmt=".1f", cmap="coolwarm", center=0)
plt.title("Cluster Profiles (standardized feature means)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cluster_profile_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Name clusters in business terms
# ---------------------------------------------------------------------------
# Interpreted from outputs/cluster_profile_heatmap.png:
# - Cluster 0: high alcohol, high proline, high flavanoids/phenols, high OD280/OD315
#   -> richer, higher-quality, higher-alcohol wines
# - Cluster 1: high malic acid, high color intensity, but low hue and low OD280/OD315
#   -> tart, deeply colored, but low on the quality-dilution ratio
# - Cluster 2: low alcohol, low color intensity, low proline
#   -> lighter-bodied, more delicate wines
cluster_names = {
    0: "Premium / Full-Bodied",
    1: "High Acid / Low Clarity",
    2: "Light / Delicate",
}
cluster_actions = {
    0: "Position for premium pricing tier; recommend to customers who buy high-end reds.",
    1: "Route toward blending or value-tier positioning rather than premium shelf placement.",
    2: "Market toward lighter-wine drinkers; pair with lighter foods (fish, salads).",
}

print("\nCluster names (interpreted from profile):")
for c, name in cluster_names.items():
    print(f"  Cluster {c}: {name}  ->  {cluster_actions[c]}")

# ---------------------------------------------------------------------------
# 6. Check stability across seeds
# ---------------------------------------------------------------------------
seeds = [0, 1, 2, 3, 4]
seed_labelings = []
for s in seeds:
    km_s = KMeans(n_clusters=best_k, random_state=s, n_init=10)
    seed_labelings.append(km_s.fit_predict(X_train_scaled))

ari_scores = []
for i in range(len(seed_labelings)):
    for j in range(i + 1, len(seed_labelings)):
        ari_scores.append(adjusted_rand_score(seed_labelings[i], seed_labelings[j]))

print(f"\nStability check -- pairwise Adjusted Rand Index across {len(seeds)} seeds:")
print(f"  mean ARI = {np.mean(ari_scores):.3f}  (closer to 1.0 = more stable)")

# Sanity check against true wine cultivars (NOT used in training, just a sanity signal)
ari_vs_truth = adjusted_rand_score(true_labels.loc[X_train.index], train_clusters)
print(f"  ARI vs true wine cultivar labels (sanity check only): {ari_vs_truth:.3f}")

# ---------------------------------------------------------------------------
# Save results table
# ---------------------------------------------------------------------------
results = pd.DataFrame({
    "cluster": list(cluster_names.keys()),
    "name": list(cluster_names.values()),
    "recommended_action": list(cluster_actions.values()),
    "size": profile_df["cluster"].value_counts().sort_index().values,
})
results.to_csv(f"{OUT_DIR}/cluster_summary.csv", index=False)
print(f"\nSaved: {OUT_DIR}/k_selection.png, {OUT_DIR}/cluster_profile_heatmap.png, {OUT_DIR}/cluster_summary.csv")
