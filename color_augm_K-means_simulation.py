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
# WICHTIG: HIER IMPORTIEREN WIR DEIN NEUES MODUL
# ==========================================
from pca_utils import calculate_dataset_pca

# Pfade definieren
IMAGE_DIR = "./jpg"
OUTPUT_DIR = "./Fotos"

if __name__ == "__main__":
    
    # 1. Dynamische PCA über dein ausgelagertes Modul berechnen
    ANZAHL_BILDER = 5000
    eigenvalues, eigenvectors = calculate_dataset_pca(IMAGE_DIR, num_images_to_analyze=ANZAHL_BILDER)
    
    # 2. Zielbild laden
    all_image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    target_image_path = all_image_paths[10]
    
    img = Image.open(target_image_path).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img) / 255.0
    
    # 3. Farbraum-Clustering (K-Means Simulation für GNG)
    pixels = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pixels)
    cluster_map = cluster_labels.reshape(224, 224)
    
    # 4. Automatisch das lila Blüten-Cluster finden
    cluster_scores = []
    for i in range(3):
        cluster_pixels = pixels[cluster_labels == i]
        score = np.mean(cluster_pixels[:, 0] + cluster_pixels[:, 2]) - np.mean(cluster_pixels[:, 1])
        cluster_scores.append(score)
    flower_cluster_id = np.argmax(cluster_scores)
    
    # 5. Lokale Fancy PCA auf das Cluster anwenden
    augmented_img = img_np.copy()
    flower_mask = (cluster_map == flower_cluster_id)
    print(eigenvectors)
    print(eigenvalues)
    # Subtile Helligkeits- und starke Farbachsen-Verschiebung (PC 2)
    alpha_1 = 0.5 # np.random.normal(0, 0.1)
    alpha_2 = 6.3 
    alpha_3 = 0.7 #np.random.normal(0, 0.4)
    
    a = (alpha_1 * eigenvalues[0] * eigenvectors[0] + 
         alpha_2 * eigenvalues[1] * eigenvectors[1] + 
         alpha_3 * eigenvalues[2] * eigenvectors[2])
    
    augmented_img[flower_mask] = augmented_img[flower_mask] + a
    augmented_img = np.clip(augmented_img, 0, 1)
    
    # ==========================================
    # 6. ERWEITERTES SPEICHERN: JETZT MIT MASKE
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Auf 3 Panels vergrößert
    
    # Panel 1: Originalbild
    axes[0].imshow(img_np)
    axes[0].set_title("Originalbild", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Die extrahierte Maske (Weiß = True, Schwarz = False)
    axes[1].imshow(flower_mask, cmap='gray')
    axes[1].set_title("Extrahierte Blüten-Maske", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Das Endergebnis
    axes[2].imshow(augmented_img)
    axes[2].set_title("Augmentiertes Ergebnis", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_filepath = os.path.join(OUTPUT_DIR, "colorAugment_K-means.png")
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"\n[ERFOLG] Bild inklusive Maske erfolgreich generiert und gespeichert unter: '{output_filepath}'")