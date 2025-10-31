from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

captions = ["Three women sit in a village", "Three woman sit in a village", "Three women are sitting outdoors in a small town", "A group of women sit in a village"]
caption = "Guy in blue shirt dances with girl in white shirt"

# Ajoutons la phrase supplémentaire à la liste pour calculer toutes les distances
all_captions = captions + [caption]

model = SentenceTransformer('sentence-transformers/roberta-large-nli-stsb-mean-tokens')
embeddings = model.encode(all_captions)

# Calcul des distances euclidiennes entre toutes les paires de phrases
distances_euclidean = pdist(embeddings, metric='euclidean')
distance_matrix_euclidean = squareform(distances_euclidean)

# Calcul des distances cosinus entre toutes les paires de phrases
distances_cosine = pdist(embeddings, metric='cosine')
distance_matrix_cosine = squareform(distances_cosine)

# Affichage des résultats
print("=== DISTANCES ENTRE LES REPRÉSENTATIONS DANS L'ESPACE LATENT ===\n")

print("Phrases analysées:")
for i, caption in enumerate(all_captions):
    print(f"{i}: {caption}")

print("\n=== DISTANCES EUCLIDIENNES ===")
print("Matrice des distances euclidiennes:")
print(np.round(distance_matrix_euclidean, 4))

print("\n=== DISTANCES COSINUS ===")
print("Matrice des distances cosinus:")
print(np.round(distance_matrix_cosine, 4))

print("\n=== ANALYSE DÉTAILLÉE DES DISTANCES ===")
n = len(all_captions)
for i in range(n):
    for j in range(i+1, n):
        print(f"Distance entre phrase {i} et phrase {j}:")
        print(f"  Euclidienne: {distance_matrix_euclidean[i,j]:.4f}")
        print(f"  Cosinus: {distance_matrix_cosine[i,j]:.4f}")
        print(f"  Phrase {i}: {all_captions[i]}")
        print(f"  Phrase {j}: {all_captions[j]}")
        print()

# Visualisation des distances
plt.figure(figsize=(15, 6))

# Heatmap des distances euclidiennes
plt.subplot(1, 2, 1)
sns.heatmap(distance_matrix_euclidean, annot=True, fmt='.3f', 
            xticklabels=[f"P{i}" for i in range(n)], 
            yticklabels=[f"P{i}" for i in range(n)],
            cmap='viridis')
plt.title('Distances Euclidiennes')

# Heatmap des distances cosinus
plt.subplot(1, 2, 2)
sns.heatmap(distance_matrix_cosine, annot=True, fmt='.3f',
            xticklabels=[f"P{i}" for i in range(n)], 
            yticklabels=[f"P{i}" for i in range(n)],
            cmap='viridis')
plt.title('Distances Cosinus')

plt.tight_layout()
plt.show()

# Statistiques sur les distances
print("=== STATISTIQUES ===")
print(f"Distance euclidienne moyenne: {np.mean(distances_euclidean):.4f}")
print(f"Distance euclidienne médiane: {np.median(distances_euclidean):.4f}")
print(f"Distance euclidienne min: {np.min(distances_euclidean):.4f}")
print(f"Distance euclidienne max: {np.max(distances_euclidean):.4f}")

print(f"Distance cosinus moyenne: {np.mean(distances_cosine):.4f}")
print(f"Distance cosinus médiane: {np.median(distances_cosine):.4f}")
print(f"Distance cosinus min: {np.min(distances_cosine):.4f}")
print(f"Distance cosinus max: {np.max(distances_cosine):.4f}")

# Trouver les paires les plus similaires et les plus différentes
min_euclidean_idx = np.unravel_index(np.argmin(distance_matrix_euclidean + np.eye(n) * 1000), distance_matrix_euclidean.shape)
max_euclidean_idx = np.unravel_index(np.argmax(distance_matrix_euclidean), distance_matrix_euclidean.shape)

print(f"\nPaire la plus similaire (distance euclidienne):")
print(f"  Phrases {min_euclidean_idx[0]} et {min_euclidean_idx[1]} (distance: {distance_matrix_euclidean[min_euclidean_idx]:.4f})")
print(f"  '{all_captions[min_euclidean_idx[0]]}'")
print(f"  '{all_captions[min_euclidean_idx[1]]}'")

print(f"\nPaire la plus différente (distance euclidienne):")
print(f"  Phrases {max_euclidean_idx[0]} et {max_euclidean_idx[1]} (distance: {distance_matrix_euclidean[max_euclidean_idx]:.4f})")
print(f"  '{all_captions[max_euclidean_idx[0]]}'")
print(f"  '{all_captions[max_euclidean_idx[1]]}'")

min_cosine_idx = np.unravel_index(np.argmin(distance_matrix_cosine + np.eye(n) * 1000), distance_matrix_cosine.shape)
max_cosine_idx = np.unravel_index(np.argmax(distance_matrix_cosine), distance_matrix_cosine.shape)

print(f"\nPaire la plus similaire (distance cosinus):")
print(f"  Phrases {min_cosine_idx[0]} et {min_cosine_idx[1]} (distance: {distance_matrix_cosine[min_cosine_idx]:.4f})")
print(f"  '{all_captions[min_cosine_idx[0]]}'")
print(f"  '{all_captions[min_cosine_idx[1]]}'")

print(f"\nPaire la plus différente (distance cosinus):")
print(f"  Phrases {max_cosine_idx[0]} et {max_cosine_idx[1]} (distance: {distance_matrix_cosine[max_cosine_idx]:.4f})")
print(f"  '{all_captions[max_cosine_idx[0]]}'")
print(f"  '{all_captions[max_cosine_idx[1]]}'")