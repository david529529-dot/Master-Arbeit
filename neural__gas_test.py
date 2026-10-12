import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

# ==========================================
# PFADE DEFINIEREN
# ==========================================
IMAGE_DIR = "./jpg"
OUTPUT_DIR = "./Fotos"

# Zielbild laden (Erste Blume)
all_image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
target_image_path = all_image_paths[50]

img = Image.open(target_image_path).convert('RGB')
img = img.resize((224, 224))
img_np = np.array(img)
pixels = img_np.reshape(-1, 3) / 255.0  # Normalisiert auf [0, 1]

# ==========================================
# SCHLANKE IMPLEMENTIERUNG: GROWING NEURAL GAS (GNG)
# ==========================================
print("[GNG-START] Trainiere echtes Growing Neural Gas auf den Bildpixeln...")

# Parameter für das GNG
MAX_NODES = 5          # Maximale Anzahl an Neuronen (Prototypen)
LAMBDA_GROW = 200       # Alle 200 Pixel-Schritte ein neues Neuron einfügen
AGE_MAX = 50            # Maximale Lebensdauer einer Kante
EPSILON_W = 0.1         # Lernrate für das Gewinner-Neuron
EPSILON_N = 0.005       # Lernrate für die Nachbarn-Neuronen

# 1. Initialisierung mit 2 zufälligen Knoten aus den Pixeln
np.random.seed(42)
node_indices = np.random.choice(len(pixels), size=2, replace=False)
nodes = pixels[node_indices].copy()
errors = np.zeros(2)
adj_matrix = np.zeros((2, 2))  # Adjazenzmatrix für Kanten
age_matrix = np.zeros((2, 2))  # Matrix für das Alter der Kanten

# Zufälliges Subsampling der Bildpixel für schnelles Training im Container
sample_pixels = pixels[np.random.choice(len(pixels), size=10000, replace=False)]

for step, x in enumerate(sample_pixels):
    # Schritt A: Finde die zwei am nächsten liegenden Knoten (Winner & Second)
    dists = np.linalg.norm(nodes - x, axis=1)
    nearest_indices = np.argsort(dists)
    s1, s2 = nearest_indices[0], nearest_indices[1]
    
    # Schritt B: Fehler des Gewinner-Knotens aktualisieren
    errors[s1] += dists[s1]**2
    
    # Schritt C: Gewinner und seine Nachbarn im Graphen anpassen
    nodes[s1] += EPSILON_W * (x - nodes[s1])
    neighbors = np.where(adj_matrix[s1] > 0)[0]
    for n in neighbors:
        nodes[n] += EPSILON_N * (x - nodes[n])
        age_matrix[s1, n] += 1
        age_matrix[n, s1] += 1
        
    # Schritt D: Kante zwischen s1 und s2 erzeugen/erneuern
    adj_matrix[s1, s2] = 1
    adj_matrix[s2, s1] = 1
    age_matrix[s1, s2] = 0
    age_matrix[s2, s1] = 0
    
    # Schritt E: Kanten-Alterung und Löschen toter Verbindungen
    # Kanten löschen, die zu alt sind
    too_old = (age_matrix > AGE_MAX)
    adj_matrix[too_old] = 0
    age_matrix[too_old] = 0
    
    # Knoten ohne Kanten finden und eliminieren
    has_edges = (np.sum(adj_matrix, axis=1) > 0)
    if not np.all(has_edges):
        nodes = nodes[has_edges]
        errors = errors[has_edges]
        adj_matrix = adj_matrix[has_edges][:, has_edges]
        age_matrix = age_matrix[has_edges][:, has_edges]

    # Schritt F: DYNAMISCHES WACHSTUM (Neues Neuron einfügen)
    if step > 0 and step % LAMBDA_GROW == 0 and len(nodes) < MAX_NODES:
        # Finde Knoten mit maximalem Fehler
        q = np.argmax(errors)
        # Finde dessen Nachbarn mit dem größten Fehler
        neighbors_q = np.where(adj_matrix[q] > 0)[0]
        if len(neighbors_q) > 0:
            f = neighbors_q[np.argmax(errors[neighbors_q])]
            
            # Neues Neuron genau in der Mitte einfügen
            new_node = 0.5 * (nodes[q] + nodes[f])
            nodes = np.vstack([nodes, new_node])
            
            # Kantenstruktur anpassen
            new_idx = len(nodes) - 1
            adj_matrix = np.pad(adj_matrix, ((0, 1), (0, 1)), mode='constant')
            age_matrix = np.pad(age_matrix, ((0, 1), (0, 1)), mode='constant')
            
            adj_matrix[q, f] = 0
            adj_matrix[f, q] = 0
            adj_matrix[q, new_idx] = 1
            adj_matrix[new_idx, q] = 1
            adj_matrix[f, new_idx] = 1
            adj_matrix[new_idx, f] = 1
            
            # Fehler dämpfen
            errors[q] *= 0.5
            errors[f] *= 0.5
            errors = np.append(errors, errors[q])

print(f"[GNG-ERFOLG] GNG hat vollautomatisch {len(nodes)} Prototyp-Knoten gelernt!")

# ==========================================
# EXTRAKTION DER FINALEN CLUSTER-LABELS
# ==========================================
# Jedes Pixel wird nun dem nächstgelegenen GNG-Knoten zugewiesen
final_dists = np.linalg.norm(pixels[:, None, :] - nodes[None, :, :], axis=2)
gng_labels = np.argmin(final_dists, axis=1)
cluster_map_gng = gng_labels.reshape(224, 224)

# ==========================================
# VISUALISIERUNG (3 Panels mit echtem GNG)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Original
axes[0].imshow(img_np)
axes[0].set_title("1. Originalbild", fontsize=14, fontweight='bold')
axes[0].axis('off')

# 2. 3D Farbraum mit GNG-Knoten
axes[1].remove()
ax_3d = fig.add_subplot(1, 3, 2, projection='3d')
sub_pixels = pixels[::25]
sub_labels = gng_labels[::25]

# Pixelwolke plotten
ax_3d.scatter(sub_pixels[:, 0]*255, sub_pixels[:, 1]*255, sub_pixels[:, 2]*255, 
              c=sub_labels, cmap='rainbow', marker='.', alpha=0.3, s=10)

# ECHTE GNG-KNOTEN ALS GROSSE PUNKTE EINZEICHNEN
ax_3d.scatter(nodes[:, 0]*255, nodes[:, 1]*255, nodes[:, 2]*255, 
              color='black', marker='o', s=80, edgecolors='white', label='GNG Neuronen')

ax_3d.set_xlabel('Rot')
ax_3d.set_ylabel('Grün')
ax_3d.set_zlabel('Blau')
ax_3d.set_xlim([0, 255])
ax_3d.set_ylim([0, 255])
ax_3d.set_zlim([0, 255])
ax_3d.set_title("2. RGB-Raum + GNG-Neuronen", fontsize=14, fontweight='bold')
ax_3d.legend()
ax_3d.view_init(elev=20, azim=45)

# 3. GNG-basierte Cluster-Map
axes[2].imshow(cluster_map_gng, cmap='rainbow')
axes[2].set_title("3. Echte GNG Cluster-Map", fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
output_filepath = os.path.join(OUTPUT_DIR, "gng_3panels.png")
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
print(f"Die GNG-Visualisierung liegt unter: '{output_filepath}'")