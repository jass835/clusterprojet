import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def lire_csv(chemin_fichier):
    """
    Fonction pour lire les données à partir d'un fichier CSV (en sautant la première ligne si elle contient des en-têtes).
    
    Args:
    chemin_fichier (str): Chemin vers le fichier CSV.
    
    Returns:
    numpy.ndarray: Tableau numpy contenant les points de données.
    """
    donnees = np.loadtxt(chemin_fichier, delimiter=',', skiprows=1)
    return donnees

def initialiser_centroides(donnees, k):
    """
    Fonction pour initialiser les centroids pour l'algorithme k-means.
    
    Args:
    donnees (numpy.ndarray): Tableau numpy contenant les points de données.
    k (int): Nombre de clusters.
    
    Returns:
    numpy.ndarray: Tableau numpy contenant les centroids initialisés.
    """
    indices = random.sample(range(len(donnees)), k)
    centroides = donnees[indices]
    return centroides

def assigner_points_aux_clusters(donnees, centroides):
    """
    Fonction pour assigner les points aux clusters en fonction des centroids les plus proches.
    
    Args:
    donnees (numpy.ndarray): Tableau numpy contenant les points de données.
    centroides (numpy.ndarray): Tableau numpy contenant les centroids.
    
    Returns:
    numpy.ndarray: Tableau numpy contenant les affectations de cluster pour chaque point.
    """
    distances = np.sqrt(((donnees[:, np.newaxis] - centroides) ** 2).sum(axis=2))
    affectations_clusters = np.argmin(distances, axis=1)
    return affectations_clusters

def calculer_statistique_qualite(donnees, centroides, affectations_clusters):
    """
    Calcule la statistique de qualité pour le clustering k-means.
    
    Args:
    donnees (numpy.ndarray): Tableau numpy contenant les points de données.
    centroides (numpy.ndarray): Tableau numpy contenant les centroids.
    affectations_clusters (numpy.ndarray): Tableau numpy contenant les affectations de cluster pour chaque point.
    
    Returns:
    float: La moyenne des distances entre chaque point de données et son centroid assigné.
    """
    distance_totale = np.sum(np.sum((donnees[i] - centroides[affectations_clusters[i]]) ** 2) for i in range(len(donnees)))
    statistique_qualite = distance_totale / len(donnees)
    return statistique_qualite

def sauvegarder_resultats_csv(donnees, centroides, affectations_clusters, fichier_sortie):
    """
    Sauvegarde les résultats de k-means dans un fichier CSV avec les coordonnées des centroids.
    
    Args:
    donnees (numpy.ndarray): Tableau numpy contenant les points de données.
    centroides (numpy.ndarray): Tableau numpy contenant les centroids.
    affectations_clusters (numpy.ndarray): Tableau numpy contenant les affectations de cluster pour chaque point.
    fichier_sortie (str): Chemin vers le fichier CSV de sortie.
    """
    centroides_assignes = centroides[affectations_clusters]
    if donnees.shape[1] == 2:
        header = 'x,y,centroid_x,centroid_y'
    elif donnees.shape[1] == 3:
        header = 'x,y,z,centroid_x,centroid_y,centroid_z'
    donnees_avec_centroides = np.hstack((donnees, centroides_assignes))
    np.savetxt(fichier_sortie, donnees_avec_centroides, delimiter=',', header=header, comments='')

def tracer_distance_moyenne(donnees, k_max):
    """
    Tracer le graphe de la distance moyenne en fonction du nombre de clusters.
    
    Args:
    donnees (numpy.ndarray): Tableau numpy contenant les points de données.
    k_max (int): Nombre maximum de clusters à tester.
    """
    distances = []
    for k in range(1, k_max + 1):
        centroides = initialiser_centroides(donnees, k)
        affectations_clusters = assigner_points_aux_clusters(donnees, centroides)
        distance = calculer_statistique_qualite(donnees, centroides, affectations_clusters)
        distances.append(distance)
    plt.plot(range(1, k_max + 1), distances)
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Distance Moyenne')
    plt.title('Distance Moyenne vs Nombre de Clusters')
    plt.show()

def animer_kmeans(donnees, frames):
    """
    Animation montrant le découpage des données en clusters au fil des itérations.
    
    Args:
    donnees (numpy.ndarray): Tableau numpy contenant les points de données.
    frames (list): Liste de dictionnaires contenant les centroides et les affectations de clusters pour chaque itération.
    """
    fig = plt.figure()
    if donnees.shape[1] == 2:
        ax = plt.axes(xlim=(np.min(donnees[:, 0]), np.max(donnees[:, 0])), ylim=(np.min(donnees[:, 1]), np.max(donnees[:, 1])))
        scatter = ax.scatter(donnees[:, 0], donnees[:, 1], s=10)
        centroides = frames[0]['centroides']
        centroides_plot = ax.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x')

        def mettre_a_jour(frame):
            affectations_clusters = frame['affectations_clusters']
            centroides = frame['centroides']
            scatter.set_array(affectations_clusters)
            centroides_plot.set_offsets(centroides)
            return scatter, centroides_plot

    elif donnees.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(np.min(donnees[:, 0]), np.max(donnees[:, 0]))
        ax.set_ylim(np.min(donnees[:, 1]), np.max(donnees[:, 1]))
        ax.set_zlim(np.min(donnees[:, 2]), np.max(donnees[:, 2]))
        scatter = ax.scatter(donnees[:, 0], donnees[:, 1], donnees[:, 2], s=10)
        centroides = frames[0]['centroides']
        centroides_plot = ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], c='red', marker='x')

        def mettre_a_jour(frame):
            ax.collections.clear()
            affectations_clusters = frame['affectations_clusters']
            centroides = frame['centroides']
            scatter = ax.scatter(donnees[:, 0], donnees[:, 1], donnees[:, 2], c=affectations_clusters, s=10)
            centroides_plot = ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], c='red', marker='x')
            return scatter, centroides_plot

    anim = FuncAnimation(fig, mettre_a_jour, frames=frames, interval=1000)
    plt.show()
