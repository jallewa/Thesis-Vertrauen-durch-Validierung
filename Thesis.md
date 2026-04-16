# Thesis - Vertrauen durch Validierung

**Eine quantitative Evaluation der Robustheit und Erklärungstreue von XAI-Methoden bei der dermatologischen Bildklassifikation auf dem HAM10000-Datensatz.**

Dieses Notebook demonstriert die gesamte experimentelle Pipeline der Bachelorarbeit: Vom Laden und Aufbereiten der dermatologischen Bilddaten über das Training der Convolutional Neural Networks (ResNet-101 und MobileNetV3) bis hin zur abschließenden quantitativen Evaluation der Explainable AI (XAI) Methoden LIME und Grad-CAM mithilfe des Quantus-Frameworks.

## 0. Installation
In diesem Schritt werden alle benötigten Bibliotheken und Abhängigkeiten installiert. Neben den Standard-Bibliotheken für Deep Learning (PyTorch) werden hier insbesondere captum für die Erklärungsgenerierung sowie quantus für die quantitative Validierung der XAI-Methoden benötigt


```python
# Führe diese Zelle nur aus, wenn die Pakete noch nicht installiert sind.
# Das Flag -q sorgt dafür, dass der Output nicht das ganze Notebook überflutet.

# Tipp: Eine eigene Umgebung dafür erstellen. Im Anaconda Prompt folgende Befehle ausführen:
#  - conda create --name thesis_env python=3.11 -y
#  - conda activate ml_env
#  - conda install ipykernel -y
#  - python -m ipykernel install --user --name thesis_env --display-name "cb thesis env"

# 1. Alle großen "schweren" Pakete über Conda installieren. 
# Das ! erlaubt den Terminal-Befehl im Notebook und das -y bestätigt die Installation automatisch.
#!conda install -y -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.4 pandas "numpy<2.0" seaborn matplotlib ipywidgets scikit-learn scikit-image imbalanced-learn captum

# 2. Nur spezielle Pakete (wie quantus), die es bei Conda oft nicht gibt, über pip installieren:
#%pip install quantus
```


```python
# 1. Python Standardbibliotheken
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
from collections import Counter

# 2. Drittanbieter-Bibliotheken (Datenverarbeitung & Machine Learning)
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from PIL import Image
from skimage.segmentation import slic
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 3. PyTorch (Deep Learning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights

# 4. Explainable AI (XAI)
import quantus
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import LayerGradCam, Lime, visualization as viz
```


```python
# Globale Konstanten & Pfade

# Den HAM10000-Datensatz unter https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T runterladen, entpacken
# und an die entsprechenden stellen kopieren und evntl. die folgenden Pfade anpassen.
METADATA_PATH = "./HAM10000_metadata.csv"
IMAGE_DIR = "./images/"

# Modell-Pfade
MODEL_RESNET101_FILE_PATH = "./ham10000_model_resnet101.pth"
MODEL_MOBILE_NET_V3_FILE_PATH = "./ham10000_model_mobilenet_v3.pth" 

# Device Konfiguration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Aktuell verwendetes Gerät: {device}")
```

    Aktuell verwendetes Gerät: cuda
    

## 1. HAM10000 Datensatz
Als standardisierte Datengrundlage für diese Arbeit dient der HAM10000-Datensatz (Human Against Machine with 10000 training images). Er umfasst über 10.000 dermatoskopische Bilder der sieben häufigsten pigmentierten Hautläsionen und dient als etablierter Benchmark für KI-Diagnosesysteme in der Dermatologie.

### Bilddaten laden
Zunächst werden die Bilddaten sowie die dazugehörigen Metadaten (Diagnosen) aus dem Dateisystem in einen strukturierten Pandas-DataFrame geladen.


```python
 # Wir laden die Metadaten-Datei, die Labels und Bild-IDs enthält.
df = pd.read_csv(METADATA_PATH)

# Erstellen des vollen Pfades zu den Bildern, damit wir sie später laden können
# Wir fügen '.jpg' an die image_id an.
df['path'] = df['image_id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lesion_id</th>
      <th>image_id</th>
      <th>dx</th>
      <th>dx_type</th>
      <th>age</th>
      <th>sex</th>
      <th>localization</th>
      <th>dataset</th>
      <th>path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HAM_0000118</td>
      <td>ISIC_0027419</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>./images/ISIC_0027419.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HAM_0000118</td>
      <td>ISIC_0025030</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>./images/ISIC_0025030.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HAM_0002730</td>
      <td>ISIC_0026769</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>./images/ISIC_0026769.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HAM_0002730</td>
      <td>ISIC_0025661</td>
      <td>bkl</td>
      <td>histo</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
      <td>vidir_modern</td>
      <td>./images/ISIC_0025661.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HAM_0001466</td>
      <td>ISIC_0031633</td>
      <td>bkl</td>
      <td>histo</td>
      <td>75.0</td>
      <td>male</td>
      <td>ear</td>
      <td>vidir_modern</td>
      <td>./images/ISIC_0031633.jpg</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Encoding Labels
label_encoder = LabelEncoder()

df['label'] = label_encoder.fit_transform(df['dx'])

# Display the mapping between original labels and encoded labels
label_mapping = {klasse: int(wert) for klasse, wert in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
print("Label Encoding Mapping:")
print(label_mapping)
```

    Label Encoding Mapping:
    {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
    

### Datensplit
Die Daten werden strikt in Trainings-, Validierungs- und Testdatensätze unterteilt. Das Modell lernt die Merkmale auf den Trainingsdaten und optimiert seine Hyperparameter auf den Validierungsdaten. Der Testdatensatz wird vollständig separiert und dient später als Grundlage für die XAI-Evaluation, da Erklärungen zwingend an ungesehenen Daten getestet werden müssen.


```python
X = df.drop(columns=['label'])
y = df['label']

# 1. Testdatensatz abspalten (ca. 15% -> entspricht Sangwans 1500 Bildern)
# WICHTIG: "Stratified sampling was employed".
# Das erreichen wir durch den Parameter 'stratify=y'.
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,  # Entspricht split_rate = 0.25 ??? RELLAY?
    stratify=y,  # Garantiert gleiche Klassenverteilung
    random_state=42  # Wichtig für Reproduzierbarkeit
)


# 2. Den Rest (85%) in Training und Validierung aufteilen (75-25 Split)
# Sangwan nutzt eine split_rate von 0.25 auf den verbleibenden Daten
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    stratify=y_temp,
    random_state=42
)

X_xai, _, y_xai, _ = train_test_split(
    X_test, 
    y_test, 
    train_size=60,         # Exakt 60 Bilder
    stratify=y_test,       # Behält die HAM10000-Klassenverteilung bei
    random_state=42        # Wichtig: Fixer Seed für reproduzierbare Ergebnisse!
)
```

### Oversampling
Ein charakteristisches Merkmal des HAM10000-Datensatzes ist seine starke Klassenimbalance. Während gutartige melanozytäre Nävi stark überrepräsentiert sind (ca. 67 % der Daten), sind andere Klassen (wie Dermatofibrome) nur in geringer Zahl vorhanden. Um zu verhindern, dass das Modell einen Bias zugunsten der Mehrheitsklasse entwickelt, wird hier ein Oversampling-Verfahren angewendet, das sich an der Methodik von Sangwan (2024) orientiert. (GIT-Quelle: https://github.com/HardikSangwan/thesis_diagnostics_skin/tree/main)


```python
oversample = RandomOverSampler(sampling_strategy={0:1000, 1:1500, 2:3000, 3:350, 4:3300, 5:6705, 6:450})

X_train_oversampled, y_train_oversampled = oversample.fit_resample(X_train , y_train)

y_train_oversampled_text = label_encoder.inverse_transform(y_train_oversampled)
print(f"Verteilung nach Sangwan-Oversampling: {Counter(y_train_oversampled_text)}")
```

    Verteilung nach Sangwan-Oversampling: Counter({'nv': 6705, 'mel': 3300, 'bkl': 3000, 'bcc': 1500, 'akiec': 1000, 'vasc': 450, 'df': 350})
    

### HAM10000Dataset Klasse
Hier definieren wir eine benutzerdefinierte PyTorch-Dataset-Klasse, die das effiziente Laden der Bilder, das Zuordnen der Labels und die spätere Übergabe an den DataLoader übernimmt.


```python
class HAM10000Dataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        # Wir wandeln die Pandas Series in Numpy-Arrays um für schnelleren Zugriff
        self.file_paths = file_paths.values
        self.labels = labels.values
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Bildpfad holen und Bild laden
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # 2. Augmentierung anwenden (falls vorhanden)
        if self.transform:
            image = self.transform(image)

        # 3. Label holen. Es ist BEREITS eine Zahl, also direkt in einen Tensor packen!
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label

```

## 2. Datenaufbereitung
Tiefe neuronale Netze reagieren sensibel auf unvorbereitete Rohdaten. In diesem Abschnitt werden die Bilder standardisiert und durch Data Augmentation (Datenaugmentierung) künstlich variiert, um die Generalisierungsfähigkeit der Modelle zu verbessern.

### Augmentierungs-Pipeline erstellen
Mithilfe der ```torchvision.transforms```-Pipeline werden die Bilder auf eine einheitliche Größe (z. B. 224x224 Pixel) skaliert und normalisiert. Für die Trainingsdaten werden zusätzlich zufällige Transformationen (wie Rotationen und Spiegelungen) angewendet, um Overfitting vorzubeugen und das Modell robuster zu machen.


```python
"""
Erstellt die Augmentierungs-Pipeline für das Training basierend auf Sangwan (2024).
Enthält: Resize, Normalize, Random Shifts, Flips, Zooms, Shears, Brightness.
"""
train_data_image_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    # Sangwan-spezifische Augmentations:
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    transforms.ColorJitter(brightness=0.5, contrast=0.1, hue=0.08),  # Helligkeit/Farbe
    transforms.RandomAffine(degrees=0, shear=15),  # Scherung & Zoom
    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

"""
Transformationen für die Validierung/Tests.
KEINE Augmentierung (kein Flip/Shear), nur Resize und Normalize.
"""
validation_data_image_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

```

### Beispiel-Visualisierung der Augmentierungs-Pipeline


```python
# Wir nehmen einfach den ersten Pfad aus deinem Datensatz (oder du wählst einen spezifischen Index)
sample_image_path = X_train_oversampled['path'].iloc[1] 
original_image = Image.open(sample_image_path).convert('RGB')

# --- 3. Plot mit Original und Variationen erstellen ---
num_variations = 5  # Wie viele augmentierte Bilder gezeigt werden sollen
fig, axes = plt.subplots(1, num_variations + 1, figsize=(18, 4))

# Originalbild anzeigen (nur auf 224x224 skaliert für einen fairen Vergleich)
resize_only = transforms.Resize((224, 224))
axes[0].imshow(resize_only(original_image))
axes[0].set_title("Original (nur Resized)", fontsize=12)
axes[0].axis('off')

# Augmentierte Versionen generieren und anzeigen
for i in range(1, num_variations + 1):
    # Durch den Aufruf von viz_transformer() werden die Zufallsoperationen jedes Mal neu berechnet
    aug_image = train_data_image_transformer(original_image)
    aug_image = aug_image.detach().cpu().permute(1, 2, 0).numpy()
    # Farben für Matplotlib in den Bereich [0, 1] zwingen , da die Bilder durch das normalize aus der 
    # Augmentierungspipline sonst nicht "menschlich" dargstellt werden.
    aug_image = (aug_image - aug_image.min()) / (aug_image.max() - aug_image.min() + 1e-8)

    axes[i].imshow(aug_image)
    axes[i].set_title(f"Augmentierung {i}", fontsize=12)
    axes[i].axis('off')

# Layout anpassen, damit die Bilder schön nebeneinander stehen
plt.tight_layout()
plt.show()
```


    
![png](output_19_0.png)
    


### Erstelle Datasets and loader
Die aufbereiteten Pipelines werden nun an die PyTorch-DataLoader übergeben, welche die Bilder für den Trainingsprozess in ressourcenschonende Batches unterteilen. (evtl. nochmal übberarbeiten wegen der Bachtes-Aussage)


```python
train_dataset = HAM10000Dataset(
    X_train_oversampled['path'],
    y_train_oversampled,
    transform=train_data_image_transformer  # Deine Augmentierung
)

val_dataset = HAM10000Dataset(
    X_val['path'],
    y_val,
    transform=validation_data_image_transformer  # Nur Resize/Normalize
)

test_dataset = HAM10000Dataset(
    X_test['path'],
    y_test,
    transform=validation_data_image_transformer  # Nur Resize/Normalize
)

xai_dataset = HAM10000Dataset(
    X_xai['path'],
    y_xai,
    transform=validation_data_image_transformer  # Nur Resize/Normalize, keine Augmentierung!
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # nur 32 Batchsize... sind das nicht zu wenig... wie funktioniert das?
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
xai_loader = DataLoader(xai_dataset, batch_size=4,shuffle=False)

```

## 3. Modelle erstellen
Um sicherzustellen, dass die XAI-Evaluation architekturübergreifend gültig ist und den Trade-off zwischen maximaler Performance und klinischer Effizienz abbildet, werden in dieser Arbeit zwei grundlegend unterschiedliche State-of-the-Art-Modelle implementiert: ResNet-101 und MobileNetV3. Da der Datensatz für ein Training von Grund auf nicht ausreicht, wird bei beiden Modellen auf Transfer Learning (vortrainierte ImageNet-Gewichte) zurückgegriffen.

### Resnet101 erstellen
Das ResNet-101 (Residual Network) repräsentiert in dieser Arbeit den massiven, parameterreichen Ansatz. Durch die Nutzung von "Shortcut-Connections" löst es das Problem der verschwindenden Gradienten und ermöglicht das Training extrem tiefer Netze. Es ist auf maximale Repräsentationskraft und höchste Klassifikationsgenauigkeit ausgelegt.


```python
# 1. ResNet101 laden (Standard für Transfer Learning in der Literatur)
# Sangwan (2024) nutzt ResNet (ResNet50 und ResNet101)
resNet101Model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

# Anzahl der Eingangs-Features holen (bei ResNet50 sind das 2048)
num_ftrs = resNet101Model.fc.in_features

# Den Classifier ersetzen => sangwan pseudocode
resNet101Model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),  # Reduktion der Dimension
    nn.ReLU(),  # Nicht-Linearität
    nn.Dropout(0.5),  # WICHTIG: Gegen Overfitting (siehe Sangwan)
    nn.Linear(512, 7)  # Output: 7 Klassen für HAM10000
)

 # 2. ALLES einfrieren (Basis-Wissen bewahren)
for param in resNet101Model.parameters():
    param.requires_grad = False

# 3. Den letzten großen Block auftauen (Fine-Tuning)
# Bei ResNet ist das "layer4".
for param in resNet101Model.layer4.parameters():
    param.requires_grad = True

for param in resNet101Model.fc.parameters():
    param.requires_grad = True
```

### MobileNetV3 erstellen
Im direkten Kontrast dazu steht das MobileNetV3. Diese Architekturfamilie wurde speziell für den Einsatz auf ressourcenbeschränkten, mobilen Geräten entwickelt (z. B. für Smartphone-Dermatoskope am Patientenbett). Es nutzt effiziente tiefenweise separierbare Faltungen (Depthwise Separable Convolutions) und Attention-Module (Squeeze-and-Excitation), um mit einem Bruchteil der Parameter auszukommen. (Viel zu kompliziert!!!)


```python
mobileNetV3Model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# 2. Den Classifier inspizieren und anpassen
# Der Classifier ist ein nn.Sequential Block.
# Struktur meist:
# (0): Linear (...)
# (1): Hardswish()
# (2): Dropout(...)
# (3): Linear (in_features=1280, out_features=1000)  
# Wir greifen auf die letzte Schicht zu
last_layer_index = len(mobileNetV3Model.classifier) - 1
in_features = mobileNetV3Model.classifier[last_layer_index].in_features

# OPTIMIERUNG 1: Dropout erhöhen
# Um Overfitting bei den relativ kleinen medizinischen Daten zu vermeiden,
# erhöhen wir den Dropout vor der Klassifikation auf 0.5 (Sangwan).
mobileNetV3Model.classifier[last_layer_index - 1] = nn.Dropout(p=0.5, inplace=True)

# OPTIMIERUNG 2: Letzte Schicht auf 7 Klassen (HAM10000) anpassen
mobileNetV3Model.classifier[last_layer_index] = nn.Linear(in_features, 7)
```

## 4. Modelle trainieren
In diesem Schritt erfolgt das eigentliche Fine-Tuning der beiden Modelle. Die obersten Klassifikationsschichten der vortrainierten Netzwerke wurden auf die 7 Läsionsklassen des HAM10000-Datensatzes angepasst. Optimiert wird mit dem Adam-Optimizer und der Categorical Cross-Entropy als Verlustfunktion.

(Hinweis: Der Code-Block für das Training ist standardmäßig auskommentiert, da das Training auf einer Standard-Hardware mehrere Stunden in Anspruch nehmen kann. Die trainierten Gewichte werden stattdessen im nächsten Schritt geladen)


```python
def fit(model, train_loader, val_loader, title, num_epochs=100, learning_rate=0.001, early_stop_patience=10):
    """
    Implementierung der Trainingsschleife basierend auf Sangwan (2024).
    Nutzt Adam Optimizer und ReduceLROnPlateau.
    """
    # Definition der Loss-Funktion: "Categorical Cross Entropy"
    criterion = nn.CrossEntropyLoss()

    # Definition des Optimizers: "Adam"
    # Wir optimieren nur die Parameter, die requires_grad=True haben (unser neuer Classifier Head).
    # „Für das Training des Klassifikators wurde der Adam-Optimizer gewählt. Im Gegensatz zum klassischen Stochastic Gradient Descent (SGD)
    # nutzt Adam adaptive Lernraten, was in der Literatur (Kingma & Ba, 2014) und in der Referenzstudie von Sangwan (2024) mit einer
    # schnelleren Konvergenz des Modells begründet wird.“
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning Rate Scheduler: "Reduce on plateau was used"
    # Reduziert die Lernrate, wenn der Validation-Loss nicht mehr sinkt.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    epochs_no_improve = 0
    
    print(f"Starte Training auf Gerät: {device} für {title}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Jede Epoche hat eine Trainings- und eine Validierungsphase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Modell in Trainingsmodus setzen
                dataloader = train_loader
            else:
                model.eval()  # Modell in Evaluierungsmodus setzen
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            # Listen zum Sammeln aller Vorhersagen und Labels dieser Epoche
            all_preds = []
            all_labels = []

            # Iteration über die Daten
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Parameter-Gradienten auf Null setzen
                optimizer.zero_grad()
                # Forward Pass
                # Nur im Trainings-Modus Gradienten berechnen (spart Speicher bei Val)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward Pass und Optimierung nur in der Trainingsphase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistiken sammeln
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Daten für Precision sammeln (auf CPU verschieben für scikit-learn)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            # Precision berechnen
            # average='weighted': Berücksichtigt das Ungleichgewicht der Klassen
            # zero_division=0: Verhindert Abstürze, falls eine Klasse gar nicht vorhergesagt wurde
            epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {epoch_precision:.4f} F1-Score: {epoch_f1:.4f}')

            # Deep Copy des Modells, wenn es das beste bisher ist (basierend auf Val-Acc)
            if phase == 'val':
                scheduler.step(epoch_loss)  # Update Learning Rate basierend auf Val-Loss
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_precision = epoch_precision
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0  # Counter zurücksetzen
                    print("Neues bestes Modell gespeichert!")
                else:
                    # Keine Verbesserung
                    epochs_no_improve += 1
                    print(f"Keine Verbesserung seit {epochs_no_improve} Epoche(n).")

        print()
        # --- Early Stopping Abbruchbedingung prüfen ---
        if epochs_no_improve >= early_stop_patience:
            print(f"Early Stopping ausgelöst! Keine Verbesserung der Validation Accuracy über {early_stop_patience} aufeinanderfolgende Epochen.")
            break # Bricht die äußere Epochen-Schleife ab

    print(f'Best Val Acc: {best_acc:.4f} Precision: {best_precision:.4f} F1-Score: {best_f1:.4f}')

    # Das beste Modell laden und zurückgeben
    model.load_state_dict(best_model_wts)
    return model

#trained_renet101_model = fit(resNet101Model, train_loader, val_loader, "ResNet101")
#torch.save(trained_renet101_model.state_dict(), MODEL_RESNET101_FILE_PATH)

#trained_mobilenetv3_model = fit(mobileNetV3Model, train_loader, val_loader, "MobileNetV3")
#torch.save(trained_mobilenetv3_model.state_dict(), MODEL_MOBILE_NET_V3_FILE_PATH)
```

## 5. Vortrainierte Modelle laden
Hier laden wir die Gewichte der fertig trainierten Modelle (.pth oder .pt Dateien). Dies ermöglicht eine exakte Reproduzierbarkeit der späteren Erklärungsgenerierung, ohne das Modell jedes Mal neu trainieren zu müssen.


```python
loaded_data_resnet101 = torch.load(MODEL_RESNET101_FILE_PATH)
resNet101Model.load_state_dict(loaded_data_resnet101)

loaded_data_mobilenetv3 = torch.load(MODEL_MOBILE_NET_V3_FILE_PATH)
mobileNetV3Model.load_state_dict(loaded_data_mobilenetv3)
```

    C:\Users\jalle\AppData\Local\Temp\ipykernel_24680\2924343579.py:1: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      loaded_data_resnet101 = torch.load(MODEL_RESNET101_FILE_PATH)
    C:\Users\jalle\AppData\Local\Temp\ipykernel_24680\2924343579.py:4: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      loaded_data_mobilenetv3 = torch.load(MODEL_MOBILE_NET_V3_FILE_PATH)
    




    <All keys matched successfully>



## 6. Modelle gegen den Testdatensatz validieren
Bevor die Black-Box geöffnet und erklärt wird, muss sichergestellt sein, dass die Modelle diagnostisch auf hohem Niveau agieren. In diesem Schritt wird die Performance (Accuracy, Precision, Recall, F1-Score) beider Modelle auf dem ungesehenen Testdatensatz berechnet.


```python
"""
Berechnet Accuracy, Precision, Recall und F1 auf dem Testset.
"""
resNet101Model.eval()  # Wichtig: Evaluation Modus
resNet101Model.to(device)

mobileNetV3Model.eval()  # Wichtig: Evaluation Modus
mobileNetV3Model.to(device)

def evaluateTestData(model, title):
    all_preds = []
    all_labels = []
    
    print(f"Starte Evaluation für {title} auf dem Testdatensatz...")
    
    with torch.no_grad():  # Keine Gradientenberechnung nötig
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # Vorhersage
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
    
            # Sammeln der Ergebnisse (zurück auf CPU schieben)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # --- Berechnung der Metriken ---
    
    # 1. Gesamte Accuracy (Top-1)
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n=== Gesamtergebnis für {title} ===")
    print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    class_names=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    # 2. Detaillierter Bericht (Precision, Recall, F1 pro Klasse)
    # Dies entspricht Table 3 in Sangwan (2024)
    print("\n=== Detaillierter Klassifikationsbericht ===")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4  # 4 Nachkommastellen für Präzision
    )
    print(report)

    # 3. Confusion Matrix
    print(f"Erstelle Confusion Matrix für {title}...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Visualisierung der Confusion Matrix mit Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - {title}', fontsize=14)
    plt.ylabel('Wahre Klasse', fontsize=12)
    plt.xlabel('Vorhergesagte Klasse', fontsize=12)
    
    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()

evaluateTestData(resNet101Model, "ResNet101")
evaluateTestData(mobileNetV3Model, "MobileNetV3")
```

    Starte Evaluation für ResNet101 auf dem Testdatensatz...
    
    === Gesamtergebnis für ResNet101 ===
    Test Accuracy: 0.8609 (86.09%)
    
    === Detaillierter Klassifikationsbericht ===
                  precision    recall  f1-score   support
    
           akiec     0.7400    0.7551    0.7475        49
             bcc     0.8594    0.7143    0.7801        77
             bkl     0.7748    0.7091    0.7405       165
              df     0.7500    0.7059    0.7273        17
             mel     0.6601    0.6048    0.6312       167
              nv     0.9078    0.9493    0.9281      1006
            vasc     1.0000    0.7727    0.8718        22
    
        accuracy                         0.8609      1503
       macro avg     0.8132    0.7445    0.7752      1503
    weighted avg     0.8573    0.8609    0.8579      1503
    
    Erstelle Confusion Matrix für ResNet101...
    


    
![png](output_32_1.png)
    


    Starte Evaluation für MobileNetV3 auf dem Testdatensatz...
    
    === Gesamtergebnis für MobileNetV3 ===
    Test Accuracy: 0.8609 (86.09%)
    
    === Detaillierter Klassifikationsbericht ===
                  precision    recall  f1-score   support
    
           akiec     0.7500    0.7959    0.7723        49
             bcc     0.8485    0.7273    0.7832        77
             bkl     0.7308    0.8061    0.7666       165
              df     0.7368    0.8235    0.7778        17
             mel     0.6475    0.5389    0.5882       167
              nv     0.9208    0.9364    0.9285      1006
            vasc     0.9091    0.9091    0.9091        22
    
        accuracy                         0.8609      1503
       macro avg     0.7919    0.7910    0.7894      1503
    weighted avg     0.8581    0.8609    0.8584      1503
    
    Erstelle Confusion Matrix für MobileNetV3...
    


    
![png](output_32_3.png)
    


## 7. XAI
Dies ist der methodische Kern des Notebooks. Die zuvor validierten Black-Box-Modelle werden nun mithilfe zweier konzeptionell gegensätzlicher XAI-Methoden durchleuchtet. Das Ziel ist es, die generierten Heatmaps nicht nur visuell zu betrachten, sondern objektiv und quantitativ zu messen.

### Definiere Quantus-Metriken
Für die objektive Bewertung der XAI-Verfahren nutzen wir das Quantus-Framework. Wir definieren hier Metriken zur Evaluation der Robustheit (Stabilität der Erklärung bei minimalen Bildstörungen) und der Erklärungstreue / Faithfulness (wie präzise die Erklärung den wahren Modellentscheidungsprozess widerspiegelt, z. B. durch iteratives Entfernen wichtiger Pixel).


```python
import quantus

# QUANTUS-BUGFIX
def patched_perturb_func(arr, mask, **kwargs):
    """
    Fängt den Quantus-internen Shape-Bug ab, indem die 1-Kanal-Maske 
    auf die 3 Farbkanäle des Bildes dupliziert wird.
    """
    if arr.shape != mask.shape and mask.shape[1] == 1:
        mask = np.repeat(mask, arr.shape[1], axis=1)
        
    return quantus.functions.perturb_func.baseline_replacement_by_mask(arr, mask, **kwargs)

# FAITHFULNESS
metric_irof = quantus.IROF(
    #perturb_baseline="mean", is default =>  Rieger und Hansen betonen, dass es essenziell ist, die entfernten Bildsegmente durch den Mittelwert des Datensatzes zu ersetzen und nicht durch Rauschen (Uniform Noise) oder schwarze Pixel
#. Rauschen würde das Bild so stark verfälschen, dass es außerhalb der gelernten Datenverteilung ("out-of-distribution") liegt
#. Man würde dann nicht mehr messen, ob das Feature wichtig war, sondern nur, dass das CNN durch künstliches Rauschen verwirrt wird

    perturb_func=patched_perturb_func,
    return_aggregate=False,
    disable_warnings=True
)

# ROBUSTNESS
metric_robustness = quantus.LocalLipschitzEstimate(
    nr_samples=20,  # EMPFEHLUNG 20: Anlehung an Sangwan "Runs=20"
    disable_warnings=True
)

metrics = {
    "Faithfulness": metric_irof,
    "Robustness": metric_robustness
}
```

### Helferfunktionen


```python
# Eine kleine Helferfunktion, um die Ergebnisse der Metriken darzustellen.
def score_display_helper(scores):
    scores_array = np.array(scores).flatten()
    
    # Metriken berechnen
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)

    formatted_scores = [f"{s:.4f}" for s in scores_array]
    scores_str = ", ".join(formatted_scores) 

    print(f"    Statistik: Mean: {mean_score:.4f} | Std: {std_score:.4f} | Min: {min_score:.4f} | Max: {max_score:.4f}")
    print(f"    Einzel-Scores: [{scores_str}]")
```


```python
# Box-Plot Helferfunkton
def plot_xai_comparison(results_gradcam, results_lime, metric_name):
    # Ergebnisse zusammenführen
    data = [results_gradcam, results_lime]
    labels = ['Grad-CAM', 'LIME']
    
    plt.figure(figsize=(8, 6))
    
    # Den Box-and-Whisker-Plot erstellen
    bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)
    
    # Ein bisschen Farbe für die Thesis
    colors = ['#005EA6', '#FFB6C1'] # blau für Grad-CAM, Hellrosa für LIME
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title(f'Vergleich: {metric_name}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
```

### Definiere Lime
Als Repräsentant der perturbationsbasierten Ansätze implementieren wir hier LIME (Local Interpretable Model-agnostic Explanations). LIME behandelt das CNN als Black-Box, unterteilt das Bild in Superpixel und maskiert diese systematisch, um zu messen, wie sich die Modellvorhersage verändert. Die Hyperparameter (Anzahl der Segmente und Perturbationen) sind kritisch für die Qualität der Erklärung.


```python
def get_lime_explainer(model):
    # LIME benötigt ein lokales, interpretierbares Modell (Standard: Lineare Regression)
    explainer = Lime(model, interpretable_model=SkLearnLinearRegression())

    def wrapper(model, inputs, targets, **kwargs):
        model.eval()
        inputs = torch.as_tensor(inputs, device=device)
        targets = torch.as_tensor(targets, device=device)

        batch_size = inputs.shape[0]
        attrs = []

        # WICHTIG: LIME muss die Bilder einzeln verarbeiten,
        # da die Superpixel (Segmente) spezifisch pro Bild generiert werden.
        for i in range(batch_size):
            single_input = inputs[i:i+1]  # Shape: (1, C, H, W)
            single_target = targets[i:i+1]

            # 1. Bild für die Superpixel-Berechnung vorbereiten
            # Skimage erwartet das Format (H, W, C) als Numpy-Array
            img_np = single_input[0].detach().cpu().numpy().transpose(1, 2, 0)
            
            # Normalisieren (0 bis 1) für eine saubere Segmentierung
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            # 2. Superpixel (Segmente) generieren
            # n_segments bestimmt die Granularität. Sangwan betont, dass Hyperparameter
            # von Erklärungsalgorithmen die Qualität signifikant beeinflussen.
            segments = slic(img_np, n_segments=250, compactness=10, start_label=0)

            # Feature-Maske in das von Captum erwartete Tensor-Format bringen: (1, 1, H, W)
            feature_mask = torch.tensor(segments, device=device).unsqueeze(0).unsqueeze(0)

            # 3. LIME Attribute berechnen
            # n_samples = Anzahl der Perturbationen (Je höher, desto stabiler, aber deutlich langsamer)
            attr = explainer.attribute(
                single_input,
                target=single_target,
                feature_mask=feature_mask,
                n_samples=5000 # (Ribeiro et al., 2016) nutzten für die Bildklassifikation beispielsweise oft 5.000 Samples 
            )

            # Über die Farbkanäle summieren, um eine 2D-Heatmap zu erhalten (analog zu Grad-CAM)
            attr = attr.sum(dim=1)
            attrs.append(attr)

        # Den Batch wieder zusammensetzen
        batch_attr = torch.cat(attrs, dim=0)
        return batch_attr.detach().cpu().numpy()

    return wrapper

def evaluateLIME(model, x_batch, y_batch):
    print("-" * 30)
    print("Evaluate LIME metric...")

    results = {}
    for key in metrics:
        print(f"  -> Running metric: {key}")
        start = time.time()
        scores = metrics[key](
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            device=device,
            explain_func=get_lime_explainer(model),
        )
        end = time.time()
        results[key] = scores;
        print(f"    Dauer: {end - start:.2f}s")
        score_display_helper(scores)
        print()
    return results
```

### Definiere Grad-CAM
Als Repräsentant der gradientenbasierten Ansätze wird Grad-CAM (Gradient-weighted Class Activation Mapping) definiert. Es schaut direkt in die Architektur des CNNs und berechnet die Heatmap aus dem Gradientenfluss der letzten Faltungsschicht in nur einem einzigen Vorwärts- und Rückwärtsdurchlauf.


```python
def get_gradcam_explainer(model, target_layer):
    explainer = LayerGradCam(model, target_layer)

    def wrapper(model, inputs, targets, **kwargs):
        model.eval()

        device = next(model.parameters()).device
        inputs = torch.as_tensor(inputs, device=device)
        targets = torch.as_tensor(targets, device=device)

        # Wichtig: Gradienten nullen, um Akkumulation zu verhindern
        model.zero_grad()

        attr = explainer.attribute(inputs, target=targets)
        
        # Upsampling & Transformation
        attr = F.interpolate(attr, size=inputs.shape[2:], mode='bilinear', align_corners=False)

        return attr.detach().cpu().numpy()

    return wrapper


def evaluateGradCAM(model, x_batch, y_batch, model_target_layer):
    print("-" * 30)
    print("Evaluate Grad-CAM metric...")

    results = {}
    for key in metrics:
        print(f"  -> Running metric: {key}")
        start = time.time()
        scores = metrics[key](
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            device=device,
            explain_func=get_gradcam_explainer(model, model_target_layer),
        )
        end = time.time()
        results[key] = scores;
        print(f"    Dauer: {end - start:.2f}s")
        score_display_helper(scores)
        print()
    return results

```

### Evaluiere XAI mit Quantus-Metriken
In dieser finalen Evaluationsschleife werden Grad-CAM und LIME auf einer gezielten Stichprobe des Testdatensatzes angewendet. Es wird berechnet, welches Grundkonzept – Gradienten oder Perturbation – auf dem jeweiligen Modell (ResNet vs. MobileNet) die treueren und robusteren Erklärungen liefert und wie sich der signifikante Unterschied im Rechenaufwand verhält.


```python
def get_eval_data(dataloader, num_samples):
    """Zieht exakt num_samples Bilder am Stück aus dem Dataloader."""
    x_list, y_list = [], []
    collected = 0
    
    for inputs, labels in dataloader:
        x_list.append(inputs)
        y_list.append(labels)
        collected += len(inputs)
        if collected >= num_samples:
            break
            
    # Tensoren zusammenfügen und exakt auf num_samples abschneiden
    x_tensor = torch.cat(x_list, dim=0)[:num_samples]
    y_tensor = torch.cat(y_list, dim=0)[:num_samples]
    
    return x_tensor.numpy(), y_tensor.numpy()

def evaluateXai(model, target_layer, title, max_gradcam=32, max_lime=10):
    print(f"\n{'='*40}")
    print(f"Starte Evaluation - {title}")
    print(f"{'='*40}")
    
    model = model.to(device)
    
    # 1. Datenbeschaffung
    # Bilder aufteilen (LIME testet exakt dieselben ersten Bilder wie Grad-CAM)
    print(f"Lade {max_gradcam} Bilder aus dem Test-Loader für Grad-CAM...")
    x_gradcam, y_gradcam = get_eval_data(xai_loader, max_gradcam)

    print(f"Lade {max_lime} Bilder aus dem Test-Loader für Lime...")
    x_lime, y_lime = get_eval_data(xai_loader, max_lime)
    
    # 2. Grad-CAM Evaluierung (Alles in einem Rutsch)
    print(f"\n-> Evaluiere Grad-CAM ({len(x_gradcam)} Bilder)...")
    results_gradcam = evaluateGradCAM(model, x_gradcam, y_gradcam, target_layer)
        
    # 3. LIME Evaluierung (Alles in einem Rutsch)
    print(f"\n-> Evaluiere LIME ({len(x_lime)} Bilder)...")
    results_lime = evaluateLIME(model, x_lime, y_lime)
    
    for key in metrics:
        plot_xai_comparison(results_gradcam[key], results_lime[key], key)
    
    print(f"\nEvaluation für {title} abgeschlossen!")

# Aufruf: GradCam und Lime später mit 60 Bildern. Das ist sehr Zeitintensiv (für Lime > 12 Std. mal 2 für Resnet und MobileNet)
evaluateXai(resNet101Model, resNet101Model.layer4[-1], "ResNet101", max_gradcam=60, max_lime=10)
evaluateXai(mobileNetV3Model, mobileNetV3Model.features[-1], "MobileNetV3", max_gradcam=60, max_lime=10)
```

    
    ========================================
    Starte Evaluation - ResNet101
    ========================================
    Lade 60 Bilder aus dem Test-Loader für Grad-CAM...
    Lade 10 Bilder aus dem Test-Loader für Lime...
    
    -> Evaluiere Grad-CAM (60 Bilder)...
    ------------------------------
    Evaluate Grad-CAM metric...
      -> Running metric: Faithfulness
    

    F:\Anaconda\envs\ml_env\Lib\site-packages\quantus\helpers\warn.py:257: UserWarning: The settings for perturbing input e.g., 'perturb_func' didn't cause change in input. Reconsider the parameter settings.
      warnings.warn(
    

        Dauer: 45.82s
        Statistik: Mean: -54404.0977 | Std: 418103.6122 | Min: -3265918.5124 | Max: 96.4910
        Einzel-Scores: [84.7631, 87.8961, 57.0491, 38.5585, 87.0967, 47.3351, 64.8685, 86.2504, 7.3488, 54.1460, 48.1408, 94.8144, 92.1089, 89.6047, 36.1775, 37.5969, -24.5231, 92.0068, 84.5923, 91.6171, 54.4481, -128.7446, 62.3471, 92.4045, 90.8645, 83.4114, 65.1108, 81.7638, 91.6296, 82.7585, 91.1464, 96.4910, 51.2304, -4.4153, 90.3803, 70.4182, 30.8279, 95.0723, 52.6756, -1369.2642, 25.0313, 40.5829, 48.9712, 90.8415, 32.1951, 88.0493, 44.3228, 80.4122, 70.0914, -525.2963, -3265918.5124, 55.5299, 64.7634, 90.7908, 92.4429, 59.5654, 82.2642, 51.8822, 83.8915, 58.3114]
    
      -> Running metric: Robustness
        Dauer: 31.93s
        Statistik: Mean: 0.4133 | Std: 0.1287 | Min: 0.2197 | Max: 0.9952
        Einzel-Scores: [0.3024, 0.2901, 0.3102, 0.5668, 0.4535, 0.4384, 0.3734, 0.2829, 0.4570, 0.2783, 0.3851, 0.5087, 0.2197, 0.2831, 0.4783, 0.3499, 0.7511, 0.3974, 0.2440, 0.4094, 0.4345, 0.5129, 0.3928, 0.3304, 0.4157, 0.3355, 0.3046, 0.6950, 0.4436, 0.3718, 0.4019, 0.5080, 0.3407, 0.4194, 0.3471, 0.4514, 0.4511, 0.3381, 0.4472, 0.4756, 0.5278, 0.4261, 0.4778, 0.2895, 0.4628, 0.9952, 0.3526, 0.3700, 0.3315, 0.2817, 0.6074, 0.3843, 0.4021, 0.3331, 0.2577, 0.5857, 0.3476, 0.4976, 0.3522, 0.3211]
    
    
    -> Evaluiere LIME (10 Bilder)...
    ------------------------------
    Evaluate LIME metric...
      -> Running metric: Faithfulness
    

    F:\Anaconda\envs\ml_env\Lib\site-packages\quantus\helpers\warn.py:257: UserWarning: The settings for perturbing input e.g., 'perturb_func' didn't cause change in input. Reconsider the parameter settings.
      warnings.warn(
    

        Dauer: 543.02s
        Statistik: Mean: 77.3812 | Std: 13.0795 | Min: 49.6401 | Max: 90.8717
        Einzel-Scores: [90.6259, 90.8717, 75.7024, 61.1693, 89.9863, 78.0397, 73.5713, 89.1622, 49.6401, 75.0427]
    
      -> Running metric: Robustness
        Dauer: 11101.93s
        Statistik: Mean: 0.9187 | Std: 0.1404 | Min: 0.6335 | Max: 1.1456
        Einzel-Scores: [0.8294, 0.6335, 1.1456, 1.0557, 0.8589, 0.8216, 0.8596, 1.0328, 0.9452, 1.0044]
    
    


    
![png](output_44_5.png)
    



    
![png](output_44_6.png)
    


    
    Evaluation für ResNet101 abgeschlossen!
    
    ========================================
    Starte Evaluation - MobileNetV3
    ========================================
    Lade 60 Bilder aus dem Test-Loader für Grad-CAM...
    Lade 10 Bilder aus dem Test-Loader für Lime...
    
    -> Evaluiere Grad-CAM (60 Bilder)...
    ------------------------------
    Evaluate Grad-CAM metric...
      -> Running metric: Faithfulness
    

    F:\Anaconda\envs\ml_env\Lib\site-packages\quantus\helpers\warn.py:257: UserWarning: The settings for perturbing input e.g., 'perturb_func' didn't cause change in input. Reconsider the parameter settings.
      warnings.warn(
    

        Dauer: 22.47s
        Statistik: Mean: 51.3493 | Std: 32.1794 | Min: -35.9837 | Max: 97.6987
        Einzel-Scores: [67.6284, 4.5181, 31.6453, 4.7831, 63.1065, 18.3806, 25.7831, 82.3433, 23.3823, 32.1785, 17.8243, 94.9891, 92.5804, 47.4636, 18.7255, 35.3163, 81.1230, 93.4186, 85.8677, 61.1624, 7.9764, 25.8328, 38.0820, 16.2917, 85.7440, 85.9906, 6.0825, 75.0587, 28.1208, 4.9261, 15.8995, 97.6987, 53.2942, 14.3095, 94.0264, 79.4283, 39.9741, 93.3917, 94.9423, 36.9069, 39.5320, 95.7582, 46.6077, 85.3951, 70.1496, 57.2992, 83.3937, 81.5449, 67.9211, -35.9837, 86.7150, 23.4412, 35.4940, 89.3522, 74.8694, 7.2738, 38.6147, 41.4633, 56.4482, 59.4697]
    
      -> Running metric: Robustness
        Dauer: 29.04s
        Statistik: Mean: 0.3059 | Std: 0.1438 | Min: 0.1020 | Max: 0.6304
        Einzel-Scores: [0.1937, 0.4959, 0.6304, 0.3149, 0.4989, 0.5078, 0.3436, 0.1834, 0.4634, 0.4648, 0.1276, 0.2389, 0.3513, 0.1608, 0.6086, 0.5229, 0.3387, 0.1604, 0.4232, 0.3431, 0.2934, 0.1494, 0.3280, 0.3040, 0.1801, 0.1680, 0.3287, 0.2945, 0.4538, 0.5814, 0.2955, 0.2794, 0.1282, 0.1785, 0.2335, 0.1464, 0.4651, 0.1314, 0.1331, 0.3310, 0.3645, 0.2599, 0.4510, 0.1020, 0.3114, 0.6130, 0.2529, 0.1100, 0.1851, 0.1466, 0.3359, 0.3744, 0.3696, 0.1484, 0.1645, 0.2805, 0.1690, 0.3238, 0.1291, 0.4872]
    
    
    -> Evaluiere LIME (10 Bilder)...
    ------------------------------
    Evaluate LIME metric...
      -> Running metric: Faithfulness
    

    F:\Anaconda\envs\ml_env\Lib\site-packages\quantus\helpers\warn.py:257: UserWarning: The settings for perturbing input e.g., 'perturb_func' didn't cause change in input. Reconsider the parameter settings.
      warnings.warn(
    

        Dauer: 287.93s
        Statistik: Mean: 61.8265 | Std: 23.7969 | Min: 25.1498 | Max: 91.1801
        Einzel-Scores: [88.7460, 25.1498, 26.6936, 62.7499, 84.7520, 58.7787, 44.6557, 91.1801, 50.5214, 85.0379]
    
      -> Running metric: Robustness
        Dauer: 6166.21s
        Statistik: Mean: 0.8721 | Std: 0.2071 | Min: 0.6830 | Max: 1.3595
        Einzel-Scores: [0.9564, 0.7455, 0.6959, 0.8887, 0.7480, 0.7225, 0.6830, 1.3595, 0.8061, 1.1149]
    
    


    
![png](output_44_12.png)
    



    
![png](output_44_13.png)
    


    
    Evaluation für MobileNetV3 abgeschlossen!
    

## 8. Visualisierung eines Beispielbildes
Um die berechneten quantitativen Metriken in einen praktischen Kontext zu setzen, werden in diesem Schritt die originalen Eingabebilder gemeinsam mit den generierten Heatmaps von LIME und Grad-CAM nebeneinander geplottet. Dies veranschaulicht visuell den Unterschied zwischen der feingranularen gradientenbasierten Segmentierung und der gröberen superpixel-basierten Maskierung der Modelle.


```python
def visualize_explanations(model, images, labels, target_layer, title, img_idx=0):

   # 1. Daten vorbereiten (Direkt 1 Bild auswählen)
    input_img = images[img_idx: img_idx+1].to(device)
    target = labels[img_idx: img_idx+1].to(device)
    
    # --- DIE TENSOR-GYMNASTIK ---
    # Bild für den Plot vorbereiten: Von GPU auf CPU, zu Numpy, Format drehen (H,W,C)
    img_for_plot = input_img[0].detach().cpu().permute(1, 2, 0).numpy()
    # Farben für Matplotlib in den Bereich [0, 1] zwingen
    img_for_plot = (img_for_plot - img_for_plot.min()) / (img_for_plot.max() - img_for_plot.min() + 1e-8)
    # ----------------------------
    
    # 2. Erklärungen generieren
    # --- Grad-CAM ---
    gradcam_func = get_gradcam_explainer(model, target_layer)
    attr_gradcam = gradcam_func(model, input_img, target)[0] 
    attr_gradcam = attr_gradcam.transpose(1, 2, 0)
    
    # --- LIME ---
    lime_func = get_lime_explainer(model)
    attr_lime = lime_func(model, input_img, target)[0] 
    attr_lime = np.expand_dims(attr_lime, axis=-1)
    
    # 3. Visualisierung
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"XAI Evaluation für ein Testbild: {title}", fontsize=16, fontweight='bold', y=1.05)
    
    # Originalbild plotten
    label_remapped = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
    axs[0].imshow(img_for_plot)
    axs[0].set_title(f"Original (Klasse: {label_remapped[target.item()]})")
    axs[0].axis('off')
    
    # Grad-CAM plotten
    viz.visualize_image_attr(
        attr_gradcam, img_for_plot, method="blended_heat_map", sign="positive",
        show_colorbar=False, use_pyplot=False, title="Grad-CAM", plt_fig_axis=(fig, axs[1])
    )
    
    # LIME plotten
    viz.visualize_image_attr(
        attr_lime, img_for_plot, method="blended_heat_map", sign="positive",
        show_colorbar=False, use_pyplot=False, title="LIME", plt_fig_axis=(fig, axs[2])
    )
    
    plt.tight_layout()
    plt.show()


data_iter = iter(xai_loader)
images, labels = next(data_iter)

# Um sicher zu gehen, dass die Modelle auf dem korrekten device sind.
resNet101Model.to(device)
mobileNetV3Model.to(device)

visualize_explanations(resNet101Model, images, labels, resNet101Model.layer4[-1], "ResNet101", 3)
visualize_explanations(mobileNetV3Model, images, labels, mobileNetV3Model.features[-1],"MobileNetV3", 3)


```


```python

```
