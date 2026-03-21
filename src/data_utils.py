"""
data_utils.py — Descarga, preprocesamiento y carga de datos para Aircraft Skin Defect Classifier.

Funcionalidades:
- Descarga desde Roboflow API
- Recorte de bounding boxes → parches de clasificación
- Unificación de clases entre datasets
- Splits estratificados train/val/test
- Dataset y DataLoader de PyTorch
- Augmentation para entrenamiento
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CLASS_NAMES = ["crack", "dent", "missing_head", "paint_off", "scratch"]
NUM_CLASSES = len(CLASS_NAMES)

# Mapeo de nombres de clase originales (Roboflow) → clase unificada
CLASS_MAP = {
    "crack": "crack",
    "Crack": "crack",
    "dent": "dent",
    "Dent": "dent",
    "scratch": "scratch",
    "Scratch": "scratch",
    "missing-head": "missing_head",
    "Missing-head": "missing_head",
    "missing_head": "missing_head",
    "Missing-Head": "missing_head",
    "paint-off": "paint_off",
    "Paint-off": "paint_off",
    "paint-peel-off": "paint_off",
    "Paint-Peel-Off": "paint_off",
    "paint_off": "paint_off",
    "Paint-Off": "paint_off",
}

# Estadísticas ImageNet (usadas para modelos preentrenados)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Directorio base del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"


# ---------------------------------------------------------------------------
# Descarga de datasets
# ---------------------------------------------------------------------------
def download_datasets(api_key: str, download_dir: Path | None = None):
    """Descarga los dos datasets principales desde Roboflow."""
    from roboflow import Roboflow

    if download_dir is None:
        download_dir = RAW_DIR

    rf = Roboflow(api_key=api_key)

    # Dataset 1: aircraft-skin-defects-merged-final
    print("Descargando dataset 1: aircraft-skin-defects-merged-final ...")
    project1 = rf.workspace("dibya-dillip").project("aircraft-skin-defects-merged-final")
    ds1 = project1.version(4).download("coco", location=str(download_dir / "dataset1"))

    # Dataset 2: aircraft-skin-defects (AI Assistant)
    print("Descargando dataset 2: aircraft-skin-defects (AI Assistant) ...")
    project2 = rf.workspace(
        "ai-assistant-for-general-visual-inspection-of-airplanes"
    ).project("aircraft-skin-defects")
    ds2 = project2.version(3).download("coco", location=str(download_dir / "dataset2"))

    print(f"Datasets descargados en {download_dir}")
    return ds1, ds2


# ---------------------------------------------------------------------------
# Recorte de bounding boxes (COCO format)
# ---------------------------------------------------------------------------
def crop_bboxes_coco(
    coco_dir: Path,
    output_dir: Path,
    min_size: int = 20,
    padding: int = 5,
):
    """
    Lee anotaciones COCO JSON, recorta los bounding boxes de las imágenes
    y guarda parches organizados por clase.

    Args:
        coco_dir: Directorio con 'train/', 'valid/', 'test/' y sus _annotations.coco.json
        output_dir: Directorio destino (parches por clase)
        min_size: Tamaño mínimo del parche (se descartan menores)
        padding: Píxeles extra alrededor del bbox
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = Counter()

    for split_name in ["train", "valid", "test"]:
        split_dir = coco_dir / split_name
        ann_file = split_dir / "_annotations.coco.json"

        if not ann_file.exists():
            print(f"  [SKIP] No encontrado: {ann_file}")
            continue

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # Mapeo id → nombre de categoría
        cat_map = {c["id"]: c["name"] for c in coco["categories"]}
        # Mapeo id → filename
        img_map = {im["id"]: im for im in coco["images"]}

        for ann in coco["annotations"]:
            cat_name_raw = cat_map.get(ann["category_id"], "unknown")
            cat_name = CLASS_MAP.get(cat_name_raw)
            if cat_name is None:
                continue

            img_info = img_map[ann["image_id"]]
            img_path = split_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            x, y, w, h = ann["bbox"]
            if w < min_size or h < min_size:
                continue

            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size

            # Aplicar padding con clamp
            x1 = max(0, int(x) - padding)
            y1 = max(0, int(y) - padding)
            x2 = min(iw, int(x + w) + padding)
            y2 = min(ih, int(y + h) + padding)

            crop = img.crop((x1, y1, x2, y2))

            # Guardar parche
            cls_dir = output_dir / cat_name
            cls_dir.mkdir(exist_ok=True)
            idx = count[cat_name]
            crop.save(cls_dir / f"{cat_name}_{idx:05d}.jpg", quality=95)
            count[cat_name] += 1

    print(f"  Parches generados: {dict(count)}")
    return count


def prepare_datasets(raw_dir: Path | None = None, processed_dir: Path | None = None):
    """Procesa todos los datasets raw → parches clasificados."""
    if raw_dir is None:
        raw_dir = RAW_DIR
    if processed_dir is None:
        processed_dir = PROCESSED_DIR

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True)

    total = Counter()
    for ds_name in sorted(raw_dir.iterdir()):
        if ds_name.is_dir():
            print(f"Procesando {ds_name.name} ...")
            counts = crop_bboxes_coco(ds_name, processed_dir)
            total.update(counts)

    print(f"\nTotal de parches: {sum(total.values())}")
    for cls, n in sorted(total.items()):
        print(f"  {cls}: {n}")
    return total


# ---------------------------------------------------------------------------
# Splits estratificados
# ---------------------------------------------------------------------------
def create_splits(
    processed_dir: Path | None = None,
    splits_dir: Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Crea CSVs con columnas [path, label, label_idx] para train/val/test.
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    if splits_dir is None:
        splits_dir = SPLITS_DIR

    splits_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = [], []
    for cls_dir in sorted(processed_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        if cls_name not in CLASS_NAMES:
            continue
        for img_file in cls_dir.glob("*.jpg"):
            paths.append(str(img_file))
            labels.append(cls_name)

    df = pd.DataFrame({"path": paths, "label": labels})
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(CLASS_NAMES)})

    # Split estratificado
    test_ratio = 1.0 - train_ratio - val_ratio
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), stratify=df["label"], random_state=seed
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), stratify=temp_df["label"], random_state=seed
    )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"Splits creados: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(
    img_size: int = 224,
    augment: bool = False,
    normalize: str = "imagenet",
):
    """
    Retorna transforms de torchvision.

    Args:
        img_size: Tamaño cuadrado final
        augment: Si True, aplica augmentation (para train)
        normalize: 'imagenet' para modelos preentrenados, 'compute' para calcular desde datos
    """
    if normalize == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_grayscale_flat_transforms(img_size: int = 128):
    """Transforms para MLP: escala de grises → aplanar."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Lambda(lambda x: x.view(-1)),  # Aplanar
    ])


# ---------------------------------------------------------------------------
# Dataset de PyTorch
# ---------------------------------------------------------------------------
class DefectDataset(Dataset):
    """Dataset de parches de defectos para clasificación."""

    def __init__(self, csv_path: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        label = int(row["label_idx"])

        if self.transform:
            img = self.transform(img)

        return img, label


def get_dataloaders(
    splits_dir: Path | None = None,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = True,
    normalize: str = "imagenet",
    flatten_grayscale: bool = False,
):
    """
    Crea DataLoaders para train, val y test.

    Args:
        flatten_grayscale: Si True, usa transforms para MLP (grayscale + flatten)
    """
    if splits_dir is None:
        splits_dir = SPLITS_DIR

    if flatten_grayscale:
        train_tf = get_grayscale_flat_transforms(img_size)
        eval_tf = get_grayscale_flat_transforms(img_size)
    else:
        train_tf = get_transforms(img_size, augment=augment_train, normalize=normalize)
        eval_tf = get_transforms(img_size, augment=False, normalize=normalize)

    train_ds = DefectDataset(splits_dir / "train.csv", transform=train_tf)
    val_ds = DefectDataset(splits_dir / "val.csv", transform=eval_tf)
    test_ds = DefectDataset(splits_dir / "test.csv", transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Utilidades de análisis (EDA)
# ---------------------------------------------------------------------------
def compute_dataset_stats(csv_path: str | Path, img_size: int = 224):
    """Calcula media y desviación estándar por canal del dataset."""
    df = pd.read_csv(csv_path)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    pixel_sum = torch.zeros(3)
    pixel_sq_sum = torch.zeros(3)
    n_pixels = 0

    for _, row in df.iterrows():
        img = Image.open(row["path"]).convert("RGB")
        tensor = tf(img)
        pixel_sum += tensor.sum(dim=[1, 2])
        pixel_sq_sum += (tensor ** 2).sum(dim=[1, 2])
        n_pixels += tensor.shape[1] * tensor.shape[2]

    mean = pixel_sum / n_pixels
    std = torch.sqrt(pixel_sq_sum / n_pixels - mean ** 2)
    return mean.tolist(), std.tolist()


def get_class_distribution(csv_path: str | Path) -> dict:
    """Retorna la distribución de clases de un split."""
    df = pd.read_csv(csv_path)
    return df["label"].value_counts().to_dict()


def get_class_weights(csv_path: str | Path) -> torch.Tensor:
    """Calcula pesos inversamente proporcionales a la frecuencia de cada clase."""
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    total = len(df)
    weights = []
    for cls in CLASS_NAMES:
        c = counts.get(cls, 1)
        weights.append(total / (NUM_CLASSES * c))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data utilities for Aircraft Skin Defect Classifier")
    parser.add_argument("--download", action="store_true", help="Download datasets from Roboflow")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--prepare", action="store_true", help="Prepare datasets (crop bboxes)")
    parser.add_argument("--split", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.download or args.all:
        if not args.api_key:
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if not api_key:
                raise ValueError("Provide --api-key or set ROBOFLOW_API_KEY env variable")
        else:
            api_key = args.api_key
        download_datasets(api_key)

    if args.prepare or args.all:
        prepare_datasets()

    if args.split or args.all:
        create_splits()
