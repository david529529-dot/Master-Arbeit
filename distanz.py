import numpy as np
import matplotlib
# Erzwinge das nicht-interaktive Backend für Docker
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import glob

# ==========================================
# PFADE DEFINIEREN
# ==========================================
IMAGE_DIR = "./jpg"
OUTPUT_DIR = "./Fotos"

if __name__ == "__main__":
    # 1. Zielbild laden (Wir nehmen Bild Nr. 11 aus deinem Datensatz)
    all_image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    target_image_path = all_image_paths[10]
    
    img = Image.open(target_image_path).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img) / 255.0
    h, w, _ = img_np.shape
    
    # Pixel für die PCA flachklopfen
    pixels = img_np.reshape(-1, 3)
    
    # 2. PCA für dieses EINZELNE Bild berechnen
    mean = np.mean(pixels, axis=0)
    centered_pixels = pixels - mean
    cov_matrix = np.cov(centered_pixels, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sortieren von groß nach klein
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T
    
    # Finden, welcher originale RGB-Kanal am wenigsten zur wichtigsten Komponente (PC1) beiträgt
    # Der Kanal mit dem kleinsten Absolutwert im stärksten Eigenvektor wird vernachlässigt
    weakest_channel_idx = np.argmin(np.abs(eigenvectors[0]))
    channels = ['Rot', 'Grün', 'Blau']
    keep_channels = [i for i in range(3) if i != weakest_channel_idx]
    
    print(f"[INFO] PCA-Analyse für das Bild abgeschlossen.")
    print(f"  -> Schwächster Farbkanal in PC1 (wird vernachlässigt): {channels[weakest_channel_idx]}")
    print(f"  -> Genutzte Farbkanäle für den Plot: {channels[keep_channels[0]]} und {channels[keep_channels[1]]}")
    
    # 3. Berechne für jeden Pixel den Abstand zur linken unteren Ecke (0, Helligkeit/Höhe h)
    # In Bildern ist die linke untere Ecke bei Koordinate (x=0, y=h)
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Euklidischer Abstand zur linken unteren Ecke (0, h)
    distances = np.sqrt((x_coords - 0)**2 + (y_coords - h)**2)
    # Normalisieren auf [0, 1] damit es zu den Farben passt
    distances_normalized = distances / np.max(distances)
    distances_flat = distances_normalized.reshape(-1)
    
    # 4. Daten für das Plotten vorbereiten (Subsampling, damit der Plot nicht überlädt)
    # Wir nehmen jeden 10. Pixel für eine saubere Punktwolke
    sub_sample = slice(None, None, 10)
    plot_pixels = pixels[sub_sample]
    plot_distances = distances_flat[sub_sample]
    
    # Achsen-Zuweisung
    axis_x_data = plot_pixels[:, keep_channels[0]]
    axis_y_data = plot_pixels[:, keep_channels[1]]
    axis_z_data = plot_plot_distances = plot_distances
    
    # Die echten Pixelfarben als Farbe für die Punkte im Plot nutzen
    point_colors = plot_pixels 
    
    # 5. Visualisierung: Wir erstellen zwei verschiedene Blickwinkel
    # Blickwinkel 1 (Standard-Schrägansicht)
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(axis_x_data, axis_y_data, axis_z_data, c=point_colors, marker='.', s=15, alpha=0.6)
    
    ax1.set_xlabel(f'{channels[keep_channels[0]]}-Intensität')
    ax1.set_ylabel(f'{channels[keep_channels[1]]}-Intensität')
    ax1.set_zlabel('Abstand zur linken unteren Ecke (Norm.)')
    ax1.set_title("3D-Objekt-Farbraum (Ansicht 1)", fontsize=14, fontweight='bold')
    ax1.view_init(elev=20, azim=45) # Kamera-Winkel 1
    
    output_path1 = os.path.join(OUTPUT_DIR, "3d_space_view1.png")
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Blickwinkel 2 (Von oben/Andere Seite, um die räumliche Trennung besser zu sehen)
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(axis_x_data, axis_y_data, axis_z_data, c=point_colors, marker='.', s=15, alpha=0.6)
    
    ax2.set_xlabel(f'{channels[keep_channels[0]]}-Intensität')
    ax2.set_ylabel(f'{channels[keep_channels[1]]}-Intensität')
    ax2.set_zlabel('Abstand zur linken unteren Ecke (Norm.)')
    ax2.set_title("3D-Objekt-Farbraum (Ansicht 2 - Gedreht)", fontsize=14, fontweight='bold')
    ax2.view_init(elev=40, azim=135) # Kamera-Winkel 2 (Gedreht!)
    
    output_path2 = os.path.join(OUTPUT_DIR, "3d_space_view2.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"\n[ERFOLG] Beide 3D-Blickwinkel wurden im Ordner '{OUTPUT_DIR}' gespeichert!")
    print(f"  -> Ansicht 1: '3d_space_view1.png'")
    print(f"  -> Ansicht 2: '3d_space_view2.png'")