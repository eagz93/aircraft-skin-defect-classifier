"""
test_app.py — Script de prueba funcional para Aircraft Skin Defect Classifier.

Ejecuta el pipeline de clasificación sobre imágenes reales de cada clase
de defecto y genera evidencia visual (imágenes + reporte).

Uso:
    cd aircraft-skin-defect-classifier
    python test_app.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import ViTForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLASS_NAMES = ["crack", "dent", "missing_head", "paint_off", "scratch"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
MODEL_DIR = Path("results/models/deploy/vit_final")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Test images — one per class from the test split
TEST_IMAGES = {
    "crack":        Path("data/processed/crack/crack_05463.jpg"),
    "dent":         Path("data/processed/dent/dent_03722.jpg"),
    "scratch":      Path("data/processed/scratch/scratch_00658.jpg"),
    "missing_head": Path("data/processed/missing_head/missing_head_02431.jpg"),
    "paint_off":    Path("data/processed/paint_off/paint_off_03626.jpg"),
}

SEVERITY = {
    "crack": "ALTA", "dent": "MEDIA-ALTA", "scratch": "BAJA-MEDIA",
    "missing_head": "ALTA", "paint_off": "BAJA-MEDIA",
}

OUTPUT_DIR = Path("results/app_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model():
    print(f"Cargando modelo desde {MODEL_DIR} ...")
    model = ViTForImageClassification.from_pretrained(
        str(MODEL_DIR), num_labels=NUM_CLASSES
    )
    model = model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Grad-CAM setup
# ---------------------------------------------------------------------------
class HFWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
    def forward(self, x):
        return self.model(pixel_values=x).logits

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(result.size(0), height, width, result.size(2))
    return result.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict(model, cam, img_path):
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img_pil = Image.open(img_path).convert("RGB")
    inp = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    rgb = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    t0 = time.time()
    with torch.no_grad():
        out = model(pixel_values=inp)
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        pred_idx = int(out.logits.argmax(1).item())
    latency = (time.time() - t0) * 1000

    targets = [ClassifierOutputTarget(pred_idx)]
    gc = cam(input_tensor=inp, targets=targets)[0, :]
    cam_img = show_cam_on_image(rgb, gc, use_rgb=True)

    return {
        "pred_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "latency_ms": latency,
        "cam_image": cam_img,
        "original": img_pil,
    }


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  Aircraft Skin Defect Classifier — Test Funcional")
    print("=" * 65)
    print(f"Device: {DEVICE}")

    model = load_model()
    wrapped = HFWrapper(model).to(DEVICE).eval()
    target_layers = [wrapped.model.vit.encoder.layer[-1].layernorm_before]
    cam = GradCAM(model=wrapped, target_layers=target_layers,
                  reshape_transform=vit_reshape_transform)

    results = []
    correct = 0

    for true_label, img_path in TEST_IMAGES.items():
        if not img_path.exists():
            print(f"  [SKIP] {true_label}: imagen no encontrada ({img_path})")
            continue

        res = predict(model, cam, img_path)
        is_correct = res["pred_class"] == true_label
        correct += int(is_correct)
        mark = "OK" if is_correct else "FAIL"

        print(f"\n  [{mark}] True: {true_label:15s} | Pred: {res['pred_class']:15s} "
              f"| Conf: {res['confidence']:.1%} | Latency: {res['latency_ms']:.0f} ms")

        res["true_label"] = true_label
        res["correct"] = is_correct
        results.append(res)

    # ------------------------------------------------------------------
    # Generate combined evidence figure
    # ------------------------------------------------------------------
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 5 * n))
    if n == 1:
        axes = [axes]

    for i, res in enumerate(results):
        mark = "OK" if res["correct"] else "FAIL"

        # Original
        axes[i][0].imshow(res["original"].resize((IMG_SIZE, IMG_SIZE)))
        axes[i][0].set_title(f"Original — True: {res['true_label']}", fontsize=11)
        axes[i][0].axis("off")

        # Grad-CAM
        axes[i][1].imshow(res["cam_image"])
        axes[i][1].set_title(
            f"Grad-CAM — Pred: {res['pred_class']} [{mark}]", fontsize=11
        )
        axes[i][1].axis("off")

        # Probabilities
        colors = [
            "#e74c3c" if cn == res["pred_class"] else "#3498db"
            for cn in CLASS_NAMES
        ]
        bars = axes[i][2].barh(CLASS_NAMES, [res["all_probs"][c] for c in CLASS_NAMES],
                                color=colors)
        axes[i][2].set_xlim(0, 1)
        axes[i][2].set_title("Probabilidades", fontsize=11)
        for bar, cn in zip(bars, CLASS_NAMES):
            axes[i][2].text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{res['all_probs'][cn]:.3f}", va="center", fontsize=9,
            )

    plt.suptitle(
        "Aircraft Skin Defect Classifier — Prueba Funcional Completa",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    evidence_path = OUTPUT_DIR / "test_evidence.png"
    plt.savefig(str(evidence_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Evidencia visual guardada en: {evidence_path}")

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    report = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": DEVICE,
        "model_dir": str(MODEL_DIR),
        "total_images": n,
        "correct": correct,
        "accuracy": correct / max(n, 1),
        "results": [
            {
                "true_label": r["true_label"],
                "predicted": r["pred_class"],
                "confidence": round(r["confidence"], 4),
                "correct": r["correct"],
                "latency_ms": round(r["latency_ms"], 1),
                "severity": SEVERITY.get(r["pred_class"], "N/A"),
                "all_probabilities": {k: round(v, 4) for k, v in r["all_probs"].items()},
            }
            for r in results
        ],
    }
    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Reporte JSON guardado en: {report_path}")
    print(f"\n{'=' * 65}")
    print(f"  RESULTADO: {correct}/{n} correctos ({correct/max(n,1):.0%})")
    print(f"{'=' * 65}")

    return report


if __name__ == "__main__":
    main()
