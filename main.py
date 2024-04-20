# main.py

from clust import read_csv, initialize_centroids, assign_points_to_clusters
from drawing import draw

# Chemin vers le fichier CSV
file_path = '3d_data.csv'

# Nombre de clusters
k = 10

# Lire les données à partir du fichier CSV
data = read_csv(file_path)

# Initialiser les centroïdes
centroids = initialize_centroids(data, k)

# Assigner les points aux clusters
cluster_assignments = assign_points_to_clusters(data, centroids)

# Afficher les données et leurs assignations de cluster
draw(data, windowSize=800)


