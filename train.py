import torch
from torchvision.models import resnet50
from torchvision.datasets import Flowers102 as _Flowers102
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import csv
import os
import glob
import random
from collections import deque
from pathlib import Path

from superpixel_augmentation import berechne_cache, SuperpixelFarbAug, speichere_beispiele

letzte_n_epochen = 100  # Mittelwert-Auswertung am Ende über die letzten K Epochen

# ========== AUGMENTATION WÄHLEN ==========
# "none"        : keine Augmentierung
# "colorjitter" : klassisch, globales ColorJitter auf das ganze Bild (Baseline)
# "slic_kmeans" : Superpixel-Umfärbung, Superpixel aus SLIC mit k-means
# "slic_ng"     : Superpixel-Umfärbung, Superpixel aus SLIC mit Neural Gas
augmentation = "colorjitter"

# Nur für "slic_*":
farb_modus = "unabhaengig"   # "unabhaengig" oder "graph" (Glättung über den CHL-Graphen)
n_segments = 150             # Anzahl Superpixel
slic_m = 10.0                # Kompaktheit
glaett_schritte = 3          # nur für farb_modus="graph"

# === Seed setzen ===
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# === Hyperparameter ===
lr = 1e-4
batch_size = 50
epochen = 400
weight_decay = 7.5e-7
num_classes = 102
num_workers = 18
bildgroesse = 224

# === Farb-Hyperparameter (für ColorJitter UND Superpixel-Umfärbung gleich -> fairer Vergleich) ===
brightness = 0.5
contrast   = 0.3   # bei slic_*: global auf das ganze Bild, wie bei ColorJitter
saturation = 0.1
hue        = 0.05

# === Fester runs-Ordner (eigener Name, überschreibt nicht) ===
# Für "colorjitter" exakt das alte Namensschema -> alte Läufe/Eval-Skript passen weiterhin
basis = (f"lr{lr}_bs{batch_size}_ep{epochen}_wd{weight_decay}_nc{num_classes}_seed{seed}")
if augmentation == "colorjitter":
    exp_name = (f"flowers102_colorjitter_{basis}"
                f"_bj{brightness}_cj{contrast}_sj{saturation}_hj{hue}")
elif augmentation == "none":
    exp_name = f"flowers102_none_{basis}"
elif augmentation in ("slic_kmeans", "slic_ng"):
    exp_name = (f"flowers102_{augmentation}_{farb_modus}_K{n_segments}_m{slic_m}"
                + (f"_gs{glaett_schritte}" if farb_modus == "graph" else "")
                + f"_{basis}_bj{brightness}_cj{contrast}_sj{saturation}_hj{hue}")
else:
    raise ValueError(f"Unbekannte Augmentation: {augmentation}")

run_dir = os.path.join("runs", exp_name)
os.makedirs(run_dir, exist_ok=True)
print(f"=== Lauf: {exp_name} ===")

# Pfade für Checkpoints/Monitoring/Grafik
last_path = os.path.join(run_dir, "last.pth")           # zum Resumen (überschreibt sich)
csv_path  = os.path.join(run_dir, "monitoring.csv")
plot_path = os.path.join(run_dir, "loss_accuracy.png")

# Ordner für die letzten letzte_n_epochen Epoch-Checkpoints
epoch_ckpt_dir = os.path.join(run_dir, "epoch_checkpoints")
os.makedirs(epoch_ckpt_dir, exist_ok=True)

# === Modell ===
class myModel(torch.nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.model = resnet50()  # keine Pretrained-Weights
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myModel().to(device)

# === Transforms ===
normalize_tail = [
    transforms.ToTensor(),
    lambda x: x.expand(3, -1, -1),  # no-op bei 3-Kanal, behält Kompatibilität
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
]

if augmentation == "colorjitter":
    transform_train = transforms.Compose([
        transforms.Resize((bildgroesse, bildgroesse)),
        transforms.ColorJitter(brightness=brightness, contrast=contrast,
                               saturation=saturation, hue=hue),
        *normalize_tail,
    ])
elif augmentation == "none":
    transform_train = transforms.Compose([transforms.Resize((bildgroesse, bildgroesse)), *normalize_tail])
else:
    # Bei slic_*: Resize + Superpixel-Umfärbung passieren im Dataset (braucht den Dateinamen),
    # danach nur noch globaler Kontrast + Normalisierung
    transform_train = transforms.Compose([
        transforms.ColorJitter(contrast=contrast),
        *normalize_tail,
    ])

# VAL: deterministisch (keine Augmentierung), identisch zur Baseline
transform_val = transforms.Compose([transforms.Resize((bildgroesse, bildgroesse)), *normalize_tail])

# === Daten laden ===
class Flowers102Thor(_Flowers102):
    _download_url_prefix = "https://thor.robots.ox.ac.uk/flowers/102/"


class Flowers102Superpixel(Flowers102Thor):
    # Wie Flowers102, aber vor dem Transform wird jeder Superpixel umgefärbt.
    # Der Dateiname wird gebraucht, um die vorab berechnete Segmentierung zu laden.
    def __init__(self, *args, sp_aug=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sp_aug = sp_aug

    def __getitem__(self, idx):
        pfad, label = self._image_files[idx], self._labels[idx]
        img = Image.open(pfad).convert("RGB").resize((bildgroesse, bildgroesse), Image.BILINEAR)
        img = self.sp_aug(img, Path(pfad).stem)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


projekt_dir = Path(__file__).resolve().parent
root = str(projekt_dir / "data" / "flowers102")
download = False  # Datensatz liegt vollständig vor

if augmentation in ("slic_kmeans", "slic_ng"):
    methode = "kmeans" if augmentation == "slic_kmeans" else "ng"
    cache_dir = projekt_dir / "data" / "flowers102" / "slic_cache" / f"{methode}_K{n_segments}_m{slic_m}_s{bildgroesse}"
    ds_train = Flowers102Superpixel(root=root, split="train", transform=transform_train, download=download)
    # Segmentierung einmal vorab berechnen (bereits vorhandene werden übersprungen)
    berechne_cache([str(p) for p in ds_train._image_files], cache_dir, methode,
                   n_segments, slic_m, groesse=bildgroesse, num_workers=num_workers)
    ds_train.sp_aug = SuperpixelFarbAug(cache_dir, brightness=brightness, saturation=saturation,
                                        hue=hue, modus=farb_modus, glaett_schritte=glaett_schritte)
    # Vor dem Training ansehen: runs/<exp_name>/beispiele_augmentation.png
    speichere_beispiele([str(p) for p in ds_train._image_files], ds_train.sp_aug,
                        os.path.join(run_dir, "beispiele_augmentation.png"), groesse=bildgroesse)
else:
    ds_train = Flowers102Thor(root=root, split="train", transform=transform_train, download=download)

ds_val = Flowers102Thor(root=root, split="val", transform=transform_val, download=download)

train_data = ds_train
val_data   = ds_val

# Robuste Klassenanzahl-Ermittlung, falls .classes nicht existiert
def infer_num_classes(ds, fallback=num_classes):
    if hasattr(ds, 'classes') and ds.classes is not None:
        return len(ds.classes)
    if hasattr(ds, 'class_to_idx') and ds.class_to_idx is not None:
        return len(ds.class_to_idx)
    if hasattr(ds, 'labels') and ds.labels is not None:
        try:
            return len(set(ds.labels))
        except Exception:
            pass
    if hasattr(ds, 'targets') and ds.targets is not None:
        try:
            return len(set(ds.targets))
        except Exception:
            pass
    return fallback

print(f"Anzahl Bilder im Datensatz: {len(train_data)} (Train), {len(val_data)} (Val)")
print(f"Anzahl Klassen im Trainingsdatensatz: {infer_num_classes(train_data)}, "
      f"Anzahl Klassen im Val-Datensatz: {infer_num_classes(val_data)}")

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
valloader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ======= DIAGNOSE =======
print("=== DIAGNOSE ===")
base = Path(root) / "flowers-102"
print("jpg/ vorhanden:", (base / "jpg").is_dir())
print("imagelabels.mat vorhanden:", (base / "imagelabels.mat").is_file())
print("setid.mat vorhanden:", (base / "setid.mat").is_file())
print("Augmentation:", augmentation, f"(farb_modus={farb_modus})" if augmentation.startswith("slic") else "")
print("Transform (train):", getattr(train_data, "transform", None))
print("Transform (val)  :", getattr(val_data, "transform", None))

# Gleiches Sample zweimal laden – mit Augmentierung meist NICHT identisch (random)
img0, y0 = train_data[0]
img1, y1 = train_data[0]
print("Random-Aug aktiv? Pixelgleich =", torch.equal(img0, img1))  # bei "none" True, sonst i.d.R. False

# === Optimierer und Loss ===
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_func = torch.nn.CrossEntropyLoss()

# === CSV-Monitoring vorbereiten ===
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Avg_CE_Loss', 'Top1_Acc', 'Top5_Acc'])

# === Initialisierung ===
start_epoch = 0
loss_list = []                 # Train-Loss (pro Batch)
accuracy_list = []             # Train-Acc (pro Batch)
val_accuracy_per_epoch = []    # Top-1 Val pro Epoche
val_loss_per_epoch = []        # CE-Loss Val pro Epoche

# === Resume: ggf. letzten Checkpoint laden ===
if os.path.isfile(last_path):
    checkpoint = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    for g in optimizer.param_groups:
        g['lr'] = lr  # aktuellen Code-Wert setzen

    start_epoch = checkpoint['epoch'] + 1
    loss_list = checkpoint.get('loss_list', [])
    accuracy_list = checkpoint.get('accuracy_list', [])
    # Kompatibel zu alten und neuen Keys
    val_accuracy_per_epoch = checkpoint.get('test_accuracy_per_epoch', checkpoint.get('val_accuracy_per_epoch', []))
    val_loss_per_epoch = checkpoint.get('val_loss_per_epoch', [])
    print(f"Checkpoint geladen. Fortsetzung bei Epoche {start_epoch + 1}.")
else:
    print("Kein 'last' Checkpoint gefunden. Starte bei Epoche 1.")

# === Last-letzte_n_epochen-Checkpoints: vorhandene einsammeln und ggf. aufräumen ===
existing = sorted(glob.glob(os.path.join(epoch_ckpt_dir, "epoch_*.pth")))
if len(existing) > letzte_n_epochen:
    for p in existing[:-letzte_n_epochen]:
        try:
            os.remove(p)
        except OSError as e:
            print(f"[WARN] Konnte alten Checkpoint nicht löschen: {p} ({e})")
    existing = existing[-letzte_n_epochen:]
last_letzte_n_epochen_files = deque(existing, maxlen=letzte_n_epochen)

# === Trainings- und Validierungsloop ===
for epoche in tqdm(range(start_epoch, epochen), desc="Training Epochs"):
    model.train()
    for batch in tqdm(trainloader, desc=f"Epoche {epoche+1}", leave=False):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(dim=1) == labels).float().mean()
        loss_list.append(loss.detach().item())
        accuracy_list.append(accuracy.detach().item())

    # === Validierung ===
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    total_val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            total_val_loss += loss.item() * labels.size(0)

            _, top5_preds = outputs.topk(5, dim=1)
            correct_top1 += (top5_preds[:, 0] == labels).sum().item()
            correct_top5 += (top5_preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    avg_val_loss = total_val_loss / total

    val_accuracy_per_epoch.append(top1_acc)
    val_loss_per_epoch.append(avg_val_loss)

    print(f"[Epoche {epoche+1}] Avg_CE_Loss: {avg_val_loss:.4f}, Top1_Acc: {top1_acc:.4f}, Top5_Acc: {top5_acc:.4f}")

    # === Monitoring-CSV pro Epoche anhängen ===
    with open(csv_path, mode='a', newline='') as f:
        csv.writer(f).writerow([epoche + 1, avg_val_loss, top1_acc, top5_acc])

    # === 'Last'-Checkpoint speichern (überschreibt sich) ===
    to_save = {
        'epoch': epoche,  # 0-basiert gespeichert
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss_list': loss_list,
        'accuracy_list': accuracy_list,
        # absichtlich alter Key-Name für Rückwärtskompatibilität
        'test_accuracy_per_epoch': val_accuracy_per_epoch,
        'val_loss_per_epoch': val_loss_per_epoch,
    }
    torch.save(to_save, last_path)
    print(f"Last-Checkpoint gespeichert (Epoche {epoche+1}) -> {last_path}")

    # === Per-Epoch-Checkpoint ablegen und nur die letzten letzte_n_epochen behalten ===
    epoch_path = os.path.join(epoch_ckpt_dir, f"epoch_{epoche+1:04d}.pth")
    torch.save(to_save, epoch_path)
    if len(last_letzte_n_epochen_files) == last_letzte_n_epochen_files.maxlen:
        oldest = last_letzte_n_epochen_files.popleft()
        try:
            os.remove(oldest)
        except OSError as e:
            print(f"[WARN] Konnte alten Checkpoint nicht löschen: {oldest} ({e})")
    last_letzte_n_epochen_files.append(epoch_path)

# === Am Ende: Mittelwert der Val-Top1 der letzten k Epochen in CSV ===
k = min(letzte_n_epochen, len(val_accuracy_per_epoch))
if k > 0:
    mean_last_k = float(np.mean(val_accuracy_per_epoch[-k:]))
    with open(csv_path, mode='a', newline='') as f:
        w = csv.writer(f)
        w.writerow([])  # Leerzeile als Trenner
        w.writerow(["Keine echte Epoche", "Platzhalter",
                    f"Mittelwert der Val-Top-1-Accuracy über die letzten {k} Epochen",
                    "Platzhalter"])
        w.writerow([-1, -1.0, mean_last_k, -1.0])
    print(f"[SUMMARY] Mittelwert Top1-Accuracy (letzte {k} Val-Epochen): {mean_last_k:.4f}")
else:
    print(f"[SUMMARY] Nicht genug Epochen für Mittelwert der letzten {letzte_n_epochen}.")

# === Visualisierung: 3 Subplots (Train-Loss, Val-Loss, Accuracy) ===
fig, axs = plt.subplots(3, 1, figsize=(10, 16))

# Batches pro Epoche (gegen 0 abgesichert)
bpe = len(trainloader) if len(trainloader) > 0 else 1

# Subplot 1: Train-Loss (Batches -> fraktionale Epoche)
x_train_loss = np.arange(1, len(loss_list) + 1) / bpe
axs[0].plot(x_train_loss, loss_list, label='Train Loss', color='blue')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_yscale('log')
axs[0].set_title('Train Loss über Batches')
axs[0].legend()

# Subplot 2: Val-Loss (Epochen)
axs[1].plot(range(1, 1 + len(val_loss_per_epoch)),
            val_loss_per_epoch, label='Val Avg CE-Loss', color='red', linestyle='--')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_yscale('log')
axs[1].set_title('Avg CE-Loss (Val) über Epochen')
axs[1].legend()

# Subplot 3: Accuracy (Train-Batches + Val-Epochen)
x_train_acc = np.arange(1, len(accuracy_list) + 1) / bpe
axs[2].plot(x_train_acc, accuracy_list, label='Train Accuracy', color='green')
axs[2].plot(range(1, 1 + len(val_accuracy_per_epoch)),
            val_accuracy_per_epoch, label='Top-1 Val Accuracy', color='orange', linestyle='-')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Accuracy')
axs[2].set_title('Train and Val Accuracy über Batches/Epochen')
axs[2].legend()

plt.tight_layout()
plt.savefig(plot_path)
print(f"Plot gespeichert unter: {plot_path}")
plt.show()
