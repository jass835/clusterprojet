from clust import lire_csv, initialiser_centroides, assigner_points_aux_clusters, sauvegarder_resultats_csv, calculer_statistique_qualite, tracer_distance_moyenne, animer_kmeans
from drawing import draw
import numpy as np

# Chemin vers le fichier CSV contenant les données
chemin_fichier = '2d_data.csv'

# Nombre de clusters souhaités
nombre_clusters = 10

# Lire les données à partir du fichier CSV
donnees = lire_csv(chemin_fichier)

# Initialiser les centroides pour les clusters
centroides = initialiser_centroides(donnees, nombre_clusters)

# Assigner les points de données aux clusters correspondants
affectations_clusters = assigner_points_aux_clusters(donnees, centroides)

# Calculer la statistique de qualité du clustering k-means
statistique_qualite = calculer_statistique_qualite(donnees, centroides, affectations_clusters)
print("La statistique de qualité du k-means est :", statistique_qualite)

# Afficher les données et leurs assignations de cluster
draw(donnees, windowSize=800)

# Sauvegarder les résultats de k-means dans un fichier CSV
fichier_sortie = 'resultats.csv'
sauvegarder_resultats_csv(donnees, centroides, affectations_clusters, fichier_sortie)
print("Les résultats de k-means ont été sauvegardés dans", fichier_sortie)

# Bonus 1: Tracer le graphe de la distance moyenne en fonction du nombre de clusters
tracer_distance_moyenne(donnees, k_max=10)

# Bonus 2: Animation montrant le découpage des données en clusters au fil des itérations
frames = []
max_iterations = 10
for i in range(max_iterations):
    affectations_clusters = assigner_points_aux_clusters(donnees, centroides)
    frames.append({'centroides': centroides, 'affectations_clusters': affectations_clusters})
    nouveaux_centroides = np.array([np.mean(donnees[affectations_clusters == j], axis=0) for j in range(nombre_clusters)])
    centroides = nouveaux_centroides

animer_kmeans(donnees, frames)
