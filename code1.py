import numpy as np
import matplotlib
# Erzwinge ein nicht-interaktives Backend für Docker ohne Display
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import glob
import os

# ==========================================
# PFADE DEFINIEREN (Für Docker / Linux)
# ==========================================
IMAGE_DIR = "./jpg"       # Quellordner deiner Blumenbilder
OUTPUT_DIR = "./Fotos"    # Dein neuer Zielordner für die Visualisierungen

NUM_IMAGES = 2
IMAGE_SIZE = (224, 224)

# Sicherstellen, dass der Zielordner existiert
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[INFO] Ordner '{OUTPUT_DIR}' wurde automatisch erstellt.")

# Suche nach Bildern im Quellordner
search_path = os.path.join(IMAGE_DIR, "*.jpg")
all_image_paths = sorted(glob.glob(search_path))

if len(all_image_paths) == 0:
    raise FileNotFoundError(f"Keine Bilder im Ordner '{IMAGE_DIR}' gefunden. Aktuelles Arbeitsverzeichnis ist: {os.getcwd()}")

print(f"Gefunden: {len(all_image_paths)} Bilder im Container.")
print(f"Verarbeite die ersten {NUM_IMAGES} Bilder...")

all_pixels = []
for fp in all_image_paths[:NUM_IMAGES]:
    img = Image.open(fp).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_np = np.array(img) / 255.0
    all_pixels.append(img_np.reshape(-1, 3))

all_pixels = np.vstack(all_pixels)
print(f"Datenmatrix Form für PCA: {all_pixels.shape}")

# PCA berechnen
pca = PCA(n_components=3)
pca.fit(all_pixels)

print("\n" + "="*40)
print("          PCA-ERGEBNISSE")
print("="*40)
for i in range(3):
    print(f"PC {i+1} Eigenvektor: {pca.components_[i]}")
    print(f"Varianz \u03bb_{i+1}:       {pca.explained_variance_[i]:.4f}")
    print(f"Erklärt:              {pca.explained_variance_ratio_[i]*100:.2f}%")
    print("-" * 40)

# Visualisierung erstellen
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sub_pixels = all_pixels[::1000]
ax.scatter(sub_pixels[:, 0], sub_pixels[:, 1], sub_pixels[:, 2], c=sub_pixels, marker='.', alpha=0.6, s=15)

mean = pca.mean_
colors = ['red', 'green', 'blue']
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    v = comp * np.sqrt(var) * 2.5
    ax.quiver(mean[0], mean[1], mean[2], v[0], v[1], v[2], color=colors[i], linewidth=3, label=f'PC {i+1}')

ax.set_xlabel('Rot')
ax.set_ylabel('Grün')
ax.set_zlabel('Blau')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_title(f'PCA Hauptachsen - {NUM_IMAGES} Bilder (Docker)')
ax.legend()

# WICHTIG: Pfad geändert, speichert jetzt im Unterordner 'Fotos'
output_filepath = os.path.join(OUTPUT_DIR, "pca_result.png")
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
print(f"\n[ERFOLG] Der Plot wurde im Ordner gespeichert unter: '{output_filepath}'")