import numpy as np
import matplotlib
# Erzwinge das nicht-interaktive Backend für Docker
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from sklearn.cluster import KMeans

# ==========================================
# PFADE DEFINIEREN
# ==========================================
IMAGE_DIR = "./jpg"
OUTPUT_DIR = "./Fotos"

# Das erste Bild aus deinem Datensatz holen
search_path = os.path.join(IMAGE_DIR, "*.jpg")
all_image_paths = sorted(glob.glob(search_path))
target_image_path = all_image_paths[0]

# 1. Bild laden und vorbereiten
img = Image.open(target_image_path).convert('RGB')
img = img.resize((224, 224))
img_np = np.array(img)

# 2. Bild für das Farbraum-Clustering flachklopfen (50176 Pixel, 3 Farbkanäle)
pixels = img_np.reshape(-1, 3)

print("[INFO] Berechne Farbraum-Cluster (K-Means)...")
# 3. K-Means ausführen (k=3 für Blüte, Blätter, Hintergrund)
kmeans = KMeans(n_clusters=35, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pixels)
cluster_map = cluster_labels.reshape(224, 224)

print("[INFO] Erstelle das 3-Panel-Diagramm...")
# 4. Erstelle ein Layout mit 1 Reihe und 3 Spalten
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- LINKES BILD: Original ---
axes[0].imshow(img_np)
axes[0].set_title("1. Originalbild", fontsize=14, fontweight='bold')
axes[0].axis('off')

# --- MITTLERES BILD: 3D Punktwolke ---
# Da die mittlere Spalte 3D sein muss, löschen wir die 2D-Achse und ersetzen sie durch eine 3D-Achse
axes[1].remove()
ax_3d = fig.add_subplot(1, 3, 2, projection='3d')

# Subsampling (z.B. jeden 15. Pixel), damit der 3D-Plot flüssig generiert wird
subsample = 15
sub_pixels = pixels[::subsample]
sub_labels = cluster_labels[::subsample]

# Wir plotten die Punkte im RGB-Raum und färben sie exakt nach ihren Cluster-IDs
# Das Farbschema 'jet' sorgt dafür, dass die Clusterfarben perfekt zur rechten Map passen
scatter = ax_3d.scatter(sub_pixels[:, 0], sub_pixels[:, 1], sub_pixels[:, 2], 
                         c=sub_labels, cmap='jet', marker='.', alpha=0.5, s=20)

ax_3d.set_xlabel('Rot-Kanal')
ax_3d.set_ylabel('Grün-Kanal')
ax_3d.set_zlabel('Blau-Kanal')
ax_3d.set_xlim([0, 255])
ax_3d.set_ylim([0, 255])
ax_3d.set_zlim([0, 255])
ax_3d.set_title("2. 3D-Farbraum (Clustered)", fontsize=14, fontweight='bold')
# Blickwinkel leicht anpassen für eine optimale Sicht auf die Trennung
ax_3d.view_init(elev=20, azim=45)

# --- RECHTES BILD: Cluster-Map ---
axes[2].imshow(cluster_map, cmap='jet')
axes[2].set_title("3. 2D-Cluster-Map", fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()

# 5. Ergebnis im Ordner 'Fotos' speichern
output_filepath = os.path.join(OUTPUT_DIR, "color_clustering_3panels.png")
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
print(f"\n[ERFOLG] Die 3-Panel-Visualisierung wurde gespeichert unter: '{output_filepath}'")