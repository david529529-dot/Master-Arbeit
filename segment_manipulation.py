import numpy as np
import matplotlib
# Erzwinge das nicht-interaktive Backend für Docker
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from skimage.segmentation import slic

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

# 2. SLIC Superpixel berechnen (wichtig: dasselbe Setup wie eben)
segments_slic = slic(img_np, n_segments=100, compactness=10, start_label=1)
unique_segments = np.unique(segments_slic)

# 3. Eine Kopie des Bildes für die Manipulation erstellen
manipulated_img = img_np.copy()

# 4. Zufällig 15 Segmente auswählen, die wir verändern wollen
np.random.seed(42) # Seed für Reproduzierbarkeit
chosen_segments = np.random.choice(unique_segments, size=15, replace=False)
print(f"[INFO] Ausgewählte Superpixel-IDs für die Manipulation: {chosen_segments}")

# 5. Die Maskierung und Manipulation durchführen
for segment_id in chosen_segments:
    # Erstelle eine boolesche Maske: True überall dort, wo die aktuelle Segment-ID liegt
    mask = (segments_slic == segment_id)
    
    # Manipuliere NUR die Pixel, die von der Maske getroffen werden
    # Hier färben wir das Segment komplett rot ein: R=255, G=0, B=0
    manipulated_img[mask] = [255, 0, 0]

# 6. Visualisierung: Original und manipuliertes Bild nebeneinander
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_np)
axes[0].set_title("Originalbild", fontsize=14)
axes[0].axis('off')

axes[1].imshow(manipulated_img)
axes[1].set_title("Lokal manipulierte Superpixel (Rot)", fontsize=14)
axes[1].axis('off')

plt.tight_layout()

# Ergebnis im Ordner 'Fotos' speichern
output_filepath = os.path.join(OUTPUT_DIR, "superpixel_manipulation.png")
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
print(f"\n[ERFOLG] Das manipulierte Bild wurde gespeichert unter: '{output_filepath}'")