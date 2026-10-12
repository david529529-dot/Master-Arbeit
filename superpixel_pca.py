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

# 1. Bild laden und vorbereiten
search_path = os.path.join(IMAGE_DIR, "*.jpg")
all_image_paths = sorted(glob.glob(search_path))
target_image_path = all_image_paths[0]

img = Image.open(target_image_path).convert('RGB')
img = img.resize((224, 224))
img_np = np.array(img) / 255.0  # WICHTIG: Auf [0, 1] normalisieren für die PCA!

# 2. SLIC Superpixel berechnen
segments_slic = slic(img_np, n_segments=100, compactness=10, start_label=1)
unique_segments = np.unique(segments_slic)

# 3. DEINE gelernten PCA-Ergebnisse aus den 30 Bildern eintragen
# (Ersetze diese Werte falls deine Terminalausgabe leicht abwich)
eigenvectors = np.array([
    [0.551, 0.584, 0.595],  # PC 1 (Helligkeit)
    [-0.816, 0.222, 0.534], # PC 2 (Farbachse Blüte/Blatt)
    [0.174, -0.781, 0.599]  # PC 3 (Feinabstimmung, Beispielwerte)
])
eigenvalues = np.array([0.15, 0.04, 0.01])  # Beispiel-Varianzen (lambdas)

# 4. Kopie für das augmentierte Bild erstellen
augmented_img = img_np.copy()

# 5. Zufällig 25 Segmente für die Transformation auswählen
np.random.seed(24)
chosen_segments = np.random.choice(unique_segments, size=25, replace=False)
print(f"[INFO] Wende Fancy PCA auf folgende Segmente an: {chosen_segments}")

# 6. Lokale Fancy PCA auf den ausgewählten Waben anwenden
for segment_id in chosen_segments:
    mask = (segments_slic == segment_id)
    
    # Für JEDES Segment würfeln wir ein neues Alpha (Zufallsstärke) aus einer Normalverteilung
    # Wir nutzen eine etwas höhere Standardabweichung, damit der Effekt gut sichtbar ist
    alphas = np.random.normal(0, 1.5, size=3) 
    
    # Die Formel aus dem Paper: Verschiebung 'a' berechnen
    # a = alpha_1 * lambda_1 * p_1 + alpha_2 * lambda_2 * p_2 + alpha_3 * lambda_3 * p_3
    a = np.zeros(3)
    for i in range(3):
        a += alphas[i] * eigenvalues[i] * eigenvectors[i]
    
    # WICHTIG: Wir überschreiben nicht, sondern ADDIEREN die Störung 'a' auf die Pixel der Maske
    augmented_img[mask] = augmented_img[mask] + a

# Pixelwerte wieder sauber auf den Bereich [0, 1] clippen, falls sie Grenzen überschreiten
augmented_img = np.clip(augmented_img, 0, 1)

# 7. Visualisierung und Abspeichern im Ordner 'Fotos'
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_np)
axes[0].set_title("Originalbild (Normalisiert)", fontsize=14)
axes[0].axis('off')

axes[1].imshow(augmented_img)
axes[1].set_title("Lokale Fancy PCA auf Superpixeln", fontsize=14)
axes[1].axis('off')

plt.tight_layout()

output_filepath = os.path.join(OUTPUT_DIR, "superpixel_fancy_pca.png")
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
print(f"\n[ERFOLG] Das Kombi-Bild wurde gespeichert unter: '{output_filepath}'")