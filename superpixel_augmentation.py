# Superpixel-basierte Farbaugmentation für das Training.
#
# Idee: Statt das ganze Bild global umzufärben (ColorJitter), bekommt jeder Superpixel eine
# eigene Farbveränderung (Helligkeit, Sättigung, Farbton).
#   modus="unabhaengig": jeder Superpixel zieht seine Veränderung unabhängig
#   modus="graph"      : die Zufallswerte werden über den CHL-Graphen geglättet, d. h. stark
#                        verbundene Superpixel (gleicher Objektteil) werden ähnlich umgefärbt
#
# Die Segmentierung ist deterministisch und wird EINMAL vorab berechnet und gecacht.
# Zufällig ist im Training nur die Umfärbung -> praktisch keine Zusatzkosten pro Epoche.

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse
from skimage.color import rgb2hsv, hsv2rgb

from slic_ng import slic

try:
    import torch                      # Zufall über torch -> in DataLoader-Workern sauber geseedet

    def _rand(*shape):
        return torch.rand(*shape).numpy()
except ImportError:                   # nur für Tests ohne torch
    def _rand(*shape):
        return np.random.rand(*shape)


# ========== Vorab-Berechnung der Superpixel ==========
def _cache_datei(cache_dir, bild_pfad):
    return Path(cache_dir) / f"{Path(bild_pfad).stem}.npz"


def _berechne_eins(args):
    bild_pfad, cache_dir, method, n_segments, m, groesse = args
    ziel = _cache_datei(cache_dir, bild_pfad)
    if ziel.exists():
        return
    img = np.asarray(Image.open(bild_pfad).convert("RGB").resize((groesse, groesse), Image.BILINEAR))
    labels, info = slic(img, n_segments=n_segments, m=m, method=method, return_info=True)
    np.savez_compressed(ziel, labels=labels.astype(np.uint16),
                        edges=info["chl_edges"].astype(np.int32),
                        counts=info["chl_counts"].astype(np.int32))


def berechne_cache(bild_pfade, cache_dir, method, n_segments, m, groesse=224, num_workers=8):
    os.makedirs(cache_dir, exist_ok=True)
    fehlend = [p for p in bild_pfade if not _cache_datei(cache_dir, p).exists()]
    print(f"[SLIC-Cache] {cache_dir}: {len(bild_pfade) - len(fehlend)} vorhanden, {len(fehlend)} werden berechnet")
    if fehlend:
        args = [(p, cache_dir, method, n_segments, m, groesse) for p in fehlend]
        with Pool(num_workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_berechne_eins, args), 1):
                if i % 100 == 0 or i == len(args):
                    print(f"[SLIC-Cache] {i}/{len(args)}")


# ========== Umfärbung ==========
class SuperpixelFarbAug:
    def __init__(self, cache_dir, brightness=0.5, saturation=0.1, hue=0.05,
                 modus="unabhaengig", glaett_schritte=3):
        if modus not in ("unabhaengig", "graph"):
            raise ValueError(f"Unbekannter Modus: {modus}")
        self.cache_dir = Path(cache_dir)
        self.brightness, self.saturation, self.hue = brightness, saturation, hue
        self.modus, self.glaett_schritte = modus, glaett_schritte

    def _zufallswerte(self, n_sp, edges, counts):
        x = _rand(n_sp, 3) * 2 - 1                       # pro Superpixel: (Helligkeit, Sättigung, Farbton) in [-1, 1]
        if self.modus == "graph" and len(edges) > 0:
            w = sparse.coo_matrix((counts, (edges[:, 0], edges[:, 1])), shape=(n_sp, n_sp)).tocsr()
            w = w + w.T
            grad = np.asarray(w.sum(1)).ravel()
            hat_nachbarn = grad > 0
            w_norm = sparse.diags(1 / np.maximum(grad, 1e-12)) @ w
            for _ in range(self.glaett_schritte):        # Mittelung mit den Nachbarn, gewichtet nach Kantenstärke
                x_neu = 0.5 * x + 0.5 * (w_norm @ x)
                x[hat_nachbarn] = x_neu[hat_nachbarn]
            # Glätten verkleinert die Streuung -> auf die Streuung der Gleichverteilung zurückskalieren,
            # damit beide Modi gleich starke Veränderungen machen
            x = x / (x.std(0, keepdims=True) + 1e-8) * (1 / np.sqrt(3))
            x = np.clip(x, -1, 1)
        return x

    def __call__(self, pil_img, stem):
        daten = np.load(self.cache_dir / f"{stem}.npz")
        labels = daten["labels"].astype(np.int64)
        x = self._zufallswerte(labels.max() + 1, daten["edges"], daten["counts"])

        hsv = rgb2hsv(np.asarray(pil_img))
        fb = (1 + self.brightness * x[:, 0])[labels]      # wie ColorJitter: Faktor in [1-b, 1+b]
        fs = (1 + self.saturation * x[:, 1])[labels]
        dh = (self.hue * x[:, 2])[labels]
        hsv[..., 0] = (hsv[..., 0] + dh) % 1.0
        hsv[..., 1] = np.clip(hsv[..., 1] * fs, 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2] * fb, 0, 1)
        return Image.fromarray((hsv2rgb(hsv) * 255).round().astype(np.uint8))


def speichere_beispiele(bild_pfade, aug, ziel, groesse=224, anzahl_bilder=4, anzahl_varianten=4):
    # Zeile pro Bild: Original | mehrere zufällige Augmentierungen -> visuell prüfen vor dem Training!
    zeilen = []
    for p in bild_pfade[:anzahl_bilder]:
        img = Image.open(p).convert("RGB").resize((groesse, groesse), Image.BILINEAR)
        varianten = [np.asarray(img)] + [np.asarray(aug(img, Path(p).stem)) for _ in range(anzahl_varianten)]
        zeilen.append(np.hstack(varianten))
    Image.fromarray(np.vstack(zeilen)).save(ziel)
    print(f"Beispiel-Augmentierungen gespeichert: {ziel}")
