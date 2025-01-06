import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Fonction pour afficher les correspondances
def afficher_correspondances(photo, scan_image, keypoints_test, keypoints_scan, matches, nom_scan):
    correspondances_image = cv2.drawMatches(photo, keypoints_test, scan_image, keypoints_scan, matches, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                            matchesThickness=1)
    plt.figure(figsize=(12, 8))
    plt.imshow(correspondances_image)
    plt.title(f"Correspondances avec : {nom_scan}")
    plt.axis("off")
    plt.show()

# Fonction pour calculer le score de correspondance
def calculer_score_correspondance(matches, top_n=50):
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    top_matches = sorted_matches[:top_n]
    score = np.mean([match.distance for match in top_matches])
    return score

# Fonction pour extraire la ROI
def extraire_roi(image, roi_size=0.5):
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    x1, y1 = int(center_x - roi_size * w // 2) - 150, int(center_y - roi_size * h // 2) - 105
    x2, y2 = int(center_x + roi_size * w // 2) + 150, int(center_y + roi_size * h // 2) - 185
    return image[y1:y2, x1:x2]

# Initialiser ORB
orb = cv2.ORB_create()

# Chargement de la photo
photo_nom_carte = "charizard_photo.jpg"
photo = cv2.imread(photo_nom_carte, cv2.IMREAD_GRAYSCALE)
photo = cv2.resize(photo, (600, 825)) # Comme les scans

# Extraction de la partie de la photo qui nous interesse le plus
photo_roi = extraire_roi(photo)

# Détecter les points clés et descripteurs sur la ROI de la photo
kp_test, des_test = orb.detectAndCompute(photo_roi, None)

# Dossier contenant les scans
dataset_folder = "dataset"

# Parcourir tous les fichiers dans le dataset
resultats = []
for scan_nom in os.listdir(dataset_folder):
    scan_path = os.path.join(dataset_folder, scan_nom)

    # Charger l'image de scan
    scan_image = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)
    scan_image = cv2.resize(scan_image, (600, 825))
    scan_image_roi = extraire_roi(scan_image)

    # Détecter les points clés et descripteurs sur la ROI de l'image de scan
    kp_scan, des_scan = orb.detectAndCompute(scan_image_roi, None)

    # Matcher les descripteurs
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_test, des_scan)

    # Calculer le score
    score = calculer_score_correspondance(matches)
    resultats.append((scan_nom, score, kp_scan, scan_image_roi, matches))

# Trier les résultats par score croissant (meilleur correspondance = score bas)
resultats = sorted(resultats, key=lambda x: x[1])

# Afficher les correspondances pour les 3 meilleures correspondances
print("Meilleures correspondances :")
for scan_nom, score, kp_scan, scan_image_roi, matches in resultats[:3]:
    print(f"{scan_nom} - Score : {score:.2f}")
    afficher_correspondances(photo_roi, scan_image_roi, kp_test, kp_scan, matches[:200], scan_nom)
