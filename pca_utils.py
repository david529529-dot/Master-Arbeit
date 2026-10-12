import numpy as np
from PIL import Image
import os
import glob

def calculate_dataset_pca(image_dir, num_images_to_analyze=30):
    """
    Berechnet die PCA über eine dynamische Anzahl von Bildern.
    Abgesichert für sehr kleine Bildanzahlen.
    """
    search_path = os.path.join(image_dir, "*.jpg")
    all_image_paths = sorted(glob.glob(search_path))
    
    if len(all_image_paths) == 0:
        raise FileNotFoundError(f"Keine Bilder in '{image_dir}' gefunden.")
        
    selected_paths = all_image_paths[:num_images_to_analyze]
    
    pixel_list = []
    for path in selected_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize((80, 80)) # Etwas kleiner für schnellere Berechnung in der Schleife
        img_np = np.array(img) / 255.0
        pixel_list.append(img_np.reshape(-1, 3))
        
    all_pixels = np.vstack(pixel_list)
    
    mean = np.mean(all_pixels, axis=0)
    centered_pixels = all_pixels - mean
    
    # rowvar=False bedeutet, dass die Spalten die Kanäle (R,G,B) sind
    cov_matrix = np.cov(centered_pixels, rowvar=False)
    
    # Falls nur 1 Bild geladen wird, kann die Kovarianzmatrix numerisch instabil sein.
    # Wir fügen eine winzige Prise Rauschen hinzu, um Abstürze zu verhindern.
    cov_matrix += np.eye(3) * 1e-8
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx].T 
    
    return eigenvalues, eigenvectors