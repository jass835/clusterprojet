import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_csv(file_path):
    """
    Fonction pour lire les données à partir d'un fichier CSV (en sautant la première ligne si elle contient des en-têtes).
    
    Args:
    file_path (str): Chemin vers le fichier CSV.
    
    Returns:
    numpy.ndarray: Tableau numpy contenant les points de données.
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    return data

def initialize_centroids(data, k):
    """
    Fonction pour initialiser les centroïdes pour l'algorithme k-means.
    
    Args:
    data (numpy.ndarray): Tableau numpy contenant les points de données.
    k (int): Nombre de clusters.
    
    Returns:
    numpy.ndarray: Tableau numpy contenant les centroïdes initialisés.
    """
    centroids = data[random.sample(range(len(data)), k)]
    return centroids

def assign_points_to_clusters(data, centroids):
    """
    Fonction pour assigner les points aux clusters en fonction des centroïdes les plus proches.
    
    Args:
    data (numpy.ndarray): Tableau numpy contenant les points de données.
    centroids (numpy.ndarray): Tableau numpy contenant les centroïdes.
    
    Returns:
    numpy.ndarray: Tableau numpy contenant les affectations de cluster pour chaque point.
    """
    distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments

def calculate_quality_statistic(data, centroids, cluster_assignments):
    """
    Calculates the quality statistic for k-means clustering.

    Args:
    data (numpy.ndarray): Tableau numpy contenant les points de données.
    centroids (numpy.ndarray): Tableau numpy contenant les centroïdes.
    cluster_assignments (numpy.ndarray): Tableau numpy contenant les affectations de cluster pour chaque point.

    Returns:
    float: La moyenne des distances entre chaque point de données et son centroïde assigné.
    """
    total_distance = np.sum(np.sum((data[i] - centroids[cluster_assignments[i]]) ** 2) for i in range(len(data)))
    quality_statistic = total_distance / len(data)
    return quality_statistic

def save_results_to_csv(data, centroids, cluster_assignments, output_file):
    """
    Sauvegarde les résultats de k-means dans un fichier CSV avec les coordonnées des centroides.
    
    Args:
    data (numpy.ndarray): Tableau numpy contenant les points de données.
    centroids (numpy.ndarray): Tableau numpy contenant les centroïdes.
    cluster_assignments (numpy.ndarray): Tableau numpy contenant les affectations de cluster pour chaque point.
    output_file (str): Chemin vers le fichier CSV de sortie.
    """
    data_with_centroids = np.hstack((data, centroids[cluster_assignments]))
    np.savetxt(output_file, data_with_centroids, delimiter=',', header='x,y,centroid_x,centroid_y', comments='')

# Bonus 1: Tracer le graphe de la distance moyenne en fonction du nombre de clusters
def plot_average_distance(data, max_k):
    distances = []
    for k in range(1, max_k + 1):
        centroids = initialize_centroids(data, k)
        cluster_assignments = assign_points_to_clusters(data, centroids)
        distance = calculate_quality_statistic(data, centroids, cluster_assignments)
        distances.append(distance)
    plt.plot(range(1, max_k + 1), distances)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average Distance')
    plt.title('Average Distance vs Number of Clusters')
    plt.show()

# Bonus 2: Animation montrant le découpage des données en clusters
def animate_kmeans(data, frames):
    fig = plt.figure()
    ax = plt.axes(xlim=(np.min(data[:, 0]), np.max(data[:, 0])), ylim=(np.min(data[:, 1]), np.max(data[:, 1])))
    scatter = ax.scatter(data[:, 0], data[:, 1], s=10)
    centroids = frames[0]['centroids']
    centroids_plot = ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')

    def update(frame):
        cluster_assignments = frame['cluster_assignments']
        centroids = frame['centroids']
        scatter.set_array(cluster_assignments)
        centroids_plot.set_offsets(centroids)
        return scatter, centroids_plot

    anim = FuncAnimation(fig, update, frames=frames, interval=1000, blit=True)
    plt.show()
