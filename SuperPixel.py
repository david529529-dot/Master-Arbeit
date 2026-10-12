import numpy as np
import matplotlib
# Erzwinge das nicht-interaktive Backend für Docker
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from skimage.segmentation import slic, mark_boundaries

# ==========================================
# PFADE DEFINIEREN
# ==========================================
IMAGE_DIR = "./jpg"
OUTPUT_DIR = "./Fotos"

# Sicherstellen, dass der Ausgabeordner existiert
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Das allererste Bild aus deinem Datensatz holen
search_path = os.path.join(IMAGE_DIR, "*.jpg")
all_image_paths = sorted(glob.glob(search_path))

if len(all_image_paths) == 0:
    raise FileNotFoundError(f"Keine Bilder in '{IMAGE_DIR}' gefunden.")

target_image_path = all_image_paths[20]
print(f"[INFO] Nutze Bild für Superpixel: {target_image_path}")

# 1. Bild laden und in ein Numpy-Array konvertieren
img = Image.open(target_image_path).convert('RGB')
# Wir verkleinern es leicht auf die ResNet-Größe (224x224), damit man die Struktur gut sieht
#img = img.resize((224, 224))
img_np = np.array(img)

# 2. SLIC Superpixel berechnen
# n_segments: Ungefähre Anzahl an Superpixel-Waben, die wir generieren wollen
# compactness: Höhere Werte machen die Waben quadratischer/regelmäßiger
segments_slic = slic(img_np, n_segments=200, compactness=10, start_label=1)
print(f"[INFO] Anzahl eindeutiger Superpixel-Regionen: {len(np.unique(segments_slic))}")

# 3. Grenzen der Superpixel im Bild einzeichnen
# mark_boundaries legt ein farbiges Gitter über das Originalbild
img_with_boundaries = mark_boundaries(img_np, segments_slic, color=(1, 0, 0)) # Rote Grenzen

# 4. Visualisierung: Original und Superpixel nebeneinander (1 Reihe, 2 Spalten)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Linkes Bild: Original
axes[0].imshow(img_np)
axes[0].set_title("Originalbild (224x224)", fontsize=14)
axes[0].axis('off')

# Rechtes Bild: Superpixel
axes[1].imshow(img_with_boundaries)
axes[1].set_title("Bild mit SLIC Superpixeln", fontsize=14)
axes[1].axis('off')

plt.tight_layout()

# Ergebnis im Ordner 'Fotos' speichern
output_filepath = os.path.join(OUTPUT_DIR, "superpixel_vergleich.png")
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
print(f"\n[ERFOLG] Das Superpixel-Bild wurde gespeichert unter: '{output_filepath}'")