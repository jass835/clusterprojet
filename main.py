from clust import read_csv, initialize_centroids, assign_points_to_clusters, save_results_to_csv, calculate_quality_statistic, plot_average_distance, animate_kmeans
from drawing import draw
import numpy as np
import matplotlib.pyplot as plt

# Chemin vers le fichier CSV
file_path = 'mock_2d_data.csv'

# Nombre de clusters
k = 10

# Lire les données à partir du fichier CSV
data = read_csv(file_path)

# Initialiser les centroïdes
centroids = initialize_centroids(data, k)

# Assigner les points aux clusters
cluster_assignments = assign_points_to_clusters(data, centroids)

# Calculer la statistique de qualité de k-means
quality_statistic = calculate_quality_statistic(data, centroids, cluster_assignments)
print("La statistique de qualité de k-means est :", quality_statistic)

# Afficher les données et leurs assignations de cluster
draw(data, windowSize=800)

# Sauvegarder les résultats de k-means dans un fichier CSV
output_file = 'results.csv'
save_results_to_csv(data, centroids, cluster_assignments, output_file)
print("Les résultats de k-means ont été sauvegardés dans", output_file)

# Bonus 1: Tracer le graphe de la distance moyenne en fonction du nombre de clusters
plot_average_distance(data, max_k=10)

# Bonus 2: Animation montrant le découpage des données en clusters
frames = []
max_iterations = 10
for i in range(max_iterations):
    cluster_assignments = assign_points_to_clusters(data, centroids)
    frames.append({'centroids': centroids, 'cluster_assignments': cluster_assignments})
    new_centroids = np.array([np.mean(data[cluster_assignments == j], axis=0) for j in range(k)])
    centroids = new_centroids

animate_kmeans(data, frames)

