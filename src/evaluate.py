"""
evaluate.py — Métricas de evaluación y visualizaciones para todos los modelos.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, class_names, y_proba=None):
    """
    Calcula todas las métricas de evaluación.

    Returns:
        dict con accuracy, f1_macro, f1_per_class, precision, recall, etc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_per_class": {
            cn: f1
            for cn, f1 in zip(
                class_names, f1_score(y_true, y_pred, average=None)
            )
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    # ROC-AUC (si hay probabilidades)
    if y_proba is not None:
        n_classes = len(class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_bin, y_proba[:, 1])
        else:
            metrics["roc_auc_macro"] = roc_auc_score(
                y_bin, y_proba, average="macro", multi_class="ovr"
            )
            metrics["roc_auc_per_class"] = {}
            for i, cn in enumerate(class_names):
                if y_bin[:, i].sum() > 0:
                    metrics["roc_auc_per_class"][cn] = roc_auc_score(
                        y_bin[:, i], y_proba[:, i]
                    )

    return metrics


def get_predictions_with_proba(model, loader, device, is_hf_model=False):
    """Obtiene predicciones y probabilidades del modelo."""
    model.eval()
    all_preds = []
    all_labels = []
    all_proba = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            if is_hf_model:
                outputs = model(pixel_values=images)
                logits = outputs.logits
            else:
                logits = model(images)

            proba = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_proba.append(proba.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_proba),
    )


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """Matriz de confusión con heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absoluta
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
        yticklabels=class_names, ax=axes[0]
    )
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalizada
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
        yticklabels=class_names, ax=axes[1]
    )
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history, title="Training History", save_path=None):
    """Grafica curvas de loss y accuracy durante entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(history["val_acc"], label="Val Acc", marker="s", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(y_true, y_proba, class_names, title="ROC Curves", save_path=None):
    """Grafica curvas ROC por clase."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (cn, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cn} (AUC={roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(dist: dict, title="Class Distribution", save_path=None):
    """Grafica distribución de clases."""
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = list(dist.keys())
    counts = list(dist.values())

    bars = ax.bar(classes, counts, color=sns.color_palette("viridis", len(classes)))
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(count), ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results: dict, metric: str = "f1_macro", save_path=None):
    """
    Compara múltiples modelos en una métrica.
    results: dict de {model_name: metrics_dict}
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(models))
    bars = ax.barh(models, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center")

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_images(dataset, class_names, n_per_class=5, save_path=None):
    """Muestra un grid de imágenes de ejemplo por clase."""
    import pandas as pd

    df = dataset.df if hasattr(dataset, 'df') else None
    if df is None:
        return

    fig, axes = plt.subplots(len(class_names), n_per_class, figsize=(n_per_class * 2.5, len(class_names) * 2.5))

    for i, cn in enumerate(class_names):
        class_df = df[df["label"] == cn]
        samples = class_df.sample(min(n_per_class, len(class_df)), random_state=42)

        for j in range(n_per_class):
            ax = axes[i][j] if len(class_names) > 1 else axes[j]
            if j < len(samples):
                from PIL import Image
                img = Image.open(samples.iloc[j]["path"])
                ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(cn, rotation=0, labelpad=60, fontsize=11, va="center")

    plt.suptitle("Sample Images per Class", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_comparison_table(results: dict, class_names):
    """Imprime una tabla comparativa formateada de todos los modelos."""
    import pandas as pd

    rows = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
            "F1-Macro": f"{metrics.get('f1_macro', 0):.4f}",
            "Precision": f"{metrics.get('precision_macro', 0):.4f}",
            "Recall": f"{metrics.get('recall_macro', 0):.4f}",
        }
        # F1 per class
        f1_pc = metrics.get("f1_per_class", {})
        for cn in class_names:
            row[f"F1-{cn}"] = f"{f1_pc.get(cn, 0):.4f}"

        if "roc_auc_macro" in metrics:
            row["AUC-Macro"] = f"{metrics['roc_auc_macro']:.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
