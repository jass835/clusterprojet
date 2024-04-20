import csv
import random

def read_csv(file_path):
    """
    Fonction pour lire les données à partir d'un fichier CSV (en sautant la première ligne si elle contient des en-têtes).
    
    Args:
    file_path (str): Chemin vers le fichier CSV.
    
    Returns:
    list de listes: Liste des points de données.
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Sauter la première ligne (en-têtes)
        for row in reader:
            data.append([float(x) for x in row])
    return data



def initialize_centroids(data, k):
    """
    Fonction pour initialiser les centroïdes pour l'algorithme k-means.
    
    Args:
    data (list de listes): Liste des points de données.
    k (int): Nombre de clusters.
    
    Returns:
    list de listes: Liste des centroïdes initialisés.
    """
    centroids = random.sample(data, k)
    return centroids

def assign_points_to_clusters(data, centroids):
    """
    Fonction pour assigner les points aux clusters en fonction des centroïdes les plus proches.
    
    Args:
    data (list de listes): Liste des points de données.
    centroids (list de listes): Liste des centroïdes.
    
    Returns:
    list: Liste des affectations de cluster pour chaque point.
    """
    clusters = []
    for point in data:
        distances = [sum((point[i] - centroid[i]) ** 2 for i in range(len(point))) for centroid in centroids]
        cluster = distances.index(min(distances))
        clusters.append(cluster)
    return clusters

