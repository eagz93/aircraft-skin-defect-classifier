"""
app_v2.py — Aircraft Skin Defect Classifier 2.0

Funcionalidades:
    - Tab 1: Clasificación ViT + LoRA + Grad-CAM + diagnóstico
    - Tab 2: Detección multi-defecto YOLO11 con bounding boxes
    - Tab 3: Pipeline de inspección (YOLO → ViT) con triage por severidad

Despliegue:
    python app_v2.py
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
CLASS_NAMES = ["crack", "dent", "missing_head", "paint_off", "scratch"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results" / "models" / "deploy" / "vit_final"
YOLO_WEIGHTS = BASE_DIR / "results" / "models" / "deploy" / "yolo11_best.pt"
META_FILE = BASE_DIR / "results" / "models" / "deploy" / "deploy_meta.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Severidad por tipo de defecto
SEVERITY_MAP = {
    "crack": {"level": "CRITICAL", "priority": 1, "color": "#e74c3c",
              "action": "Inspección inmediata. Evaluar reparación estructural."},
    "missing_head": {"level": "HIGH", "priority": 2, "color": "#e67e22",
                     "action": "Reemplazo de remache. Inspeccionar adyacentes."},
    "dent": {"level": "MEDIUM-HIGH", "priority": 3, "color": "#f39c12",
             "action": "Medir profundidad/diámetro. Consultar SRM."},
    "scratch": {"level": "MEDIUM", "priority": 4, "color": "#3498db",
                "action": "Verificar profundidad. Re-tratamiento si necesario."},
    "paint_off": {"level": "LOW-MEDIUM", "priority": 5, "color": "#2ecc71",
                  "action": "Limpiar, tratar anticorrosivo, repintar."},
}

DEFECT_DESCRIPTIONS = {
    "crack": (
        "**Grieta (Crack)**\n\n"
        "Fractura visible en la superficie metálica. "
        "Puede comprometer la integridad estructural.\n\n"
        "**Severidad**: CRITICAL\n"
        "**Acción**: Inspección detallada inmediata. "
        "Evaluar longitud, profundidad y propagación."
    ),
    "dent": (
        "**Abolladura (Dent)**\n\n"
        "Deformación cóncava en el fuselaje. "
        "Causada por impacto, granizo o manejo inadecuado.\n\n"
        "**Severidad**: MEDIUM-HIGH\n"
        "**Acción**: Medir profundidad y diámetro. "
        "Comparar con límites del SRM."
    ),
    "scratch": (
        "**Rayón (Scratch)**\n\n"
        "Daño superficial lineal en la piel del avión.\n\n"
        "**Severidad**: MEDIUM\n"
        "**Acción**: Verificar profundidad. Si penetra tratamiento "
        "anticorrosivo, requiere re-tratamiento."
    ),
    "missing_head": (
        "**Cabeza de Remache Faltante (Missing Head)**\n\n"
        "Remache sin cabeza visible. Compromete la unión de paneles.\n\n"
        "**Severidad**: HIGH\n"
        "**Acción**: Reemplazo inmediato. "
        "Inspeccionar remaches adyacentes."
    ),
    "paint_off": (
        "**Desprendimiento de Pintura (Paint Off)**\n\n"
        "Pérdida del recubrimiento protector. "
        "Expone el metal a corrosión.\n\n"
        "**Severidad**: LOW-MEDIUM\n"
        "**Acción**: Limpiar, aplicar tratamiento anticorrosivo y repintar."
    ),
}


# ---------------------------------------------------------------------------
# Cargar modelos
# ---------------------------------------------------------------------------
def load_vit_model():
    """Carga el modelo ViT fine-tuned."""
    if MODEL_DIR.exists():
        model = ViTForImageClassification.from_pretrained(
            str(MODEL_DIR), num_labels=NUM_CLASSES,
        )
    else:
        print("⚠ Modelo ViT no encontrado. Usando modelo base sin fine-tuning.")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
    return model.to(DEVICE).eval()


def load_yolo_model():
    """Carga el modelo YOLO11 fine-tuned."""
    try:
        from ultralytics import YOLO
        if YOLO_WEIGHTS.exists():
            yolo = YOLO(str(YOLO_WEIGHTS))
            print(f"✔ YOLO11 cargado desde: {YOLO_WEIGHTS}")
            return yolo
        else:
            print("⚠ Pesos YOLO11 no encontrados. Descargando modelo base...")
            return YOLO("yolo11n.pt")
    except ImportError:
        print("⚠ ultralytics no instalado. Tab de detección deshabilitado.")
        return None


# Cargar modelos
vit_model = load_vit_model()
yolo_model = load_yolo_model()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Grad-CAM setup
def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class HFModelWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        return self.model(pixel_values=x).logits


wrapped_model = HFModelWrapper(vit_model).to(DEVICE).eval()
target_layers = [wrapped_model.model.vit.encoder.layer[-1].layernorm_before]
cam = GradCAM(
    model=wrapped_model,
    target_layers=target_layers,
    reshape_transform=vit_reshape_transform,
)


# ---------------------------------------------------------------------------
# Tab 1: Clasificación ViT + Grad-CAM
# ---------------------------------------------------------------------------
def classify_defect(image):
    """Clasificación + Grad-CAM + diagnóstico."""
    if image is None:
        return None, {}, "⚠ Por favor sube una imagen."

    img_pil = Image.fromarray(image).convert("RGB")
    input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    rgb_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    with torch.no_grad():
        outputs = vit_model(pixel_values=input_tensor)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_idx = int(outputs.logits.argmax(dim=1).item())

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    label_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    severity = SEVERITY_MAP[pred_class]

    desc = DEFECT_DESCRIPTIONS.get(pred_class, "")
    desc += f"\n\n---\n**Confianza**: {confidence:.1%}"
    desc += f"\n**Prioridad de atención**: {severity['level']}"

    if confidence < 0.5:
        desc += "\n\n⚠ **Confianza baja** — Se recomienda inspección manual."

    return cam_image, label_probs, desc


# ---------------------------------------------------------------------------
# Tab 2: Detección YOLO11
# ---------------------------------------------------------------------------
def detect_defects(image, conf_threshold=0.25):
    """Detección multi-defecto con YOLO11."""
    if image is None:
        return None, "⚠ Por favor sube una imagen."

    if yolo_model is None:
        return None, "⚠ YOLO11 no disponible. Instale: pip install ultralytics"

    results = yolo_model.predict(image, conf=conf_threshold, verbose=False)[0]
    annotated = results.plot()
    annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB

    n_dets = len(results.boxes)
    if n_dets == 0:
        report = "No se detectaron defectos con el umbral de confianza actual."
    else:
        lines = [f"### {n_dets} defecto(s) detectado(s)\n"]
        for i, box in enumerate(results.boxes):
            cls_idx = int(box.cls.item())
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"clase_{cls_idx}"
            conf = float(box.conf.item())
            severity = SEVERITY_MAP.get(cls_name, {})
            sev_level = severity.get("level", "UNKNOWN")
            action = severity.get("action", "")
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            lines.append(
                f"**{i+1}. {cls_name}** — {conf:.1%} confianza\n"
                f"- Severidad: {sev_level}\n"
                f"- Ubicación: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})\n"
                f"- Acción: {action}\n"
            )
        report = "\n".join(lines)

    return annotated_rgb, report


# ---------------------------------------------------------------------------
# Tab 3: Pipeline de Inspección
# ---------------------------------------------------------------------------
def run_inspection_pipeline(image, conf_threshold=0.25):
    """Pipeline: YOLO detección → ViT clasificación por ROI → triage."""
    if image is None:
        return None, None, "⚠ Por favor sube una imagen."

    img_pil = Image.fromarray(image).convert("RGB")

    # Paso 1: ViT clasificación global + Grad-CAM
    input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    rgb_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    with torch.no_grad():
        outputs = vit_model(pixel_values=input_tensor)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_idx = int(outputs.logits.argmax(dim=1).item())

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    severity = SEVERITY_MAP[pred_class]

    # Paso 2: YOLO detección
    yolo_annotated = None
    yolo_dets = []
    if yolo_model is not None:
        results = yolo_model.predict(image, conf=conf_threshold, verbose=False)[0]
        yolo_annotated = results.plot()[:, :, ::-1]

        for box in results.boxes:
            cls_idx = int(box.cls.item())
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"clase_{cls_idx}"
            yolo_dets.append({
                "class": cls_name,
                "confidence": float(box.conf.item()),
                "severity": SEVERITY_MAP.get(cls_name, {}).get("level", "?"),
                "priority": SEVERITY_MAP.get(cls_name, {}).get("priority", 99),
            })
        yolo_dets.sort(key=lambda d: (d["priority"], -d["confidence"]))

    # Generar reporte
    report_lines = [
        "# Reporte de Inspección\n",
        f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---\n",
        "## Clasificación Global (ViT + LoRA)\n",
        f"- **Defecto principal**: {pred_class}\n",
        f"- **Confianza**: {confidence:.1%}\n",
        f"- **Severidad**: {severity['level']}\n",
        f"- **Acción**: {severity['action']}\n",
    ]

    if yolo_dets:
        report_lines.append(f"\n## Detección Multi-Defecto (YOLO11)\n")
        report_lines.append(f"**{len(yolo_dets)} defecto(s) localizados:**\n")
        for i, det in enumerate(yolo_dets):
            report_lines.append(
                f"{i+1}. **{det['class']}** — {det['confidence']:.1%} "
                f"[{det['severity']}]\n"
            )
    elif yolo_model is not None:
        report_lines.append("\n## Detección (YOLO11)\nNo se detectaron defectos con bounding boxes.\n")

    # Evaluación de riesgo
    report_lines.append("\n---\n## Evaluación de Riesgo\n")
    if severity["priority"] <= 2:
        report_lines.append(
            "🔴 **ATENCIÓN INMEDIATA REQUERIDA**\n\n"
            f"Se detectó un defecto de severidad **{severity['level']}**. "
            "Se recomienda detener la operación y realizar inspección manual detallada."
        )
    elif severity["priority"] <= 3:
        report_lines.append(
            "🟡 **INSPECCIÓN PROGRAMADA**\n\n"
            f"Defecto de severidad **{severity['level']}**. "
            "Programar inspección detallada antes del próximo vuelo."
        )
    else:
        report_lines.append(
            "🟢 **MANTENIMIENTO RUTINARIO**\n\n"
            f"Defecto de severidad **{severity['level']}**. "
            "Incluir en próximo ciclo de mantenimiento programado."
        )

    if confidence < 0.5:
        report_lines.append(
            "\n\n⚠ **Nota**: La confianza del clasificador es baja. "
            "Se recomienda revisión manual obligatoria."
        )

    return cam_image, yolo_annotated, "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------
with gr.Blocks(title="Aircraft Defect Classifier 2.0") as demo:
    gr.Markdown(
        """
        # Aircraft Skin Defect Classifier 2.0
        
        Sistema avanzado de IA para **detección, clasificación y triage** de defectos
        en superficies de aeronaves.
        
        | Modelo | Tarea | Métricas |
        |--------|-------|----------|
        | **ViT + LoRA** | Clasificación + Grad-CAM | 95.1% accuracy, 5 clases |
        | **YOLO11n** | Detección multi-defecto | mAP@50 = 0.914, bounding boxes |
        
        ---
        """
    )

    with gr.Tabs():
        # ========================= TAB 1 =========================
        with gr.Tab("Clasificación", id="classify"):
            gr.Markdown("### Clasificación con ViT + Grad-CAM")
            with gr.Row():
                with gr.Column(scale=1):
                    cls_input = gr.Image(label="Imagen de inspección", type="numpy")
                    cls_btn = gr.Button("Clasificar", variant="primary", size="lg")
                with gr.Column(scale=1):
                    cls_cam = gr.Image(label="Grad-CAM — Mapa de atención")
                    cls_label = gr.Label(num_top_classes=5, label="Probabilidades")
            cls_desc = gr.Markdown(label="Diagnóstico")

            cls_btn.click(
                fn=classify_defect,
                inputs=[cls_input],
                outputs=[cls_cam, cls_label, cls_desc],
            )
            cls_input.change(
                fn=classify_defect,
                inputs=[cls_input],
                outputs=[cls_cam, cls_label, cls_desc],
            )

        # ========================= TAB 2 =========================
        with gr.Tab("Detección YOLO11", id="detect"):
            gr.Markdown("### Detección Multi-Defecto con YOLO11")
            with gr.Row():
                with gr.Column(scale=1):
                    det_input = gr.Image(label="Imagen de inspección", type="numpy")
                    det_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Umbral de confianza",
                    )
                    det_btn = gr.Button("Detectar Defectos", variant="primary", size="lg")
                with gr.Column(scale=1):
                    det_output = gr.Image(label="Detecciones con bounding boxes")
            det_report = gr.Markdown(label="Detecciones")

            det_btn.click(
                fn=detect_defects,
                inputs=[det_input, det_conf],
                outputs=[det_output, det_report],
            )

        # ========================= TAB 3 =========================
        with gr.Tab("Pipeline de Inspección", id="pipeline"):
            gr.Markdown(
                "### Pipeline Completo: YOLO11 + ViT + Triage\n"
                "Combina detección y clasificación con evaluación de severidad."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    pipe_input = gr.Image(label="Imagen de inspección", type="numpy")
                    pipe_conf = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Umbral YOLO",
                    )
                    pipe_btn = gr.Button("Ejecutar Inspección", variant="primary", size="lg")
                with gr.Column(scale=1):
                    pipe_cam = gr.Image(label="Grad-CAM (Clasificación)")
                    pipe_det = gr.Image(label="YOLO11 (Detección)")
            pipe_report = gr.Markdown(label="Reporte de Inspección")

            pipe_btn.click(
                fn=run_inspection_pipeline,
                inputs=[pipe_input, pipe_conf],
                outputs=[pipe_cam, pipe_det, pipe_report],
            )

    gr.Markdown(
        """
        ---
        **Proyecto Final — Deep Learning** · Maestría en Ciencia de Datos  
        Dataset: Aircraft Skin Defects (Roboflow, CC BY 4.0)  
        Modelos: ViT-Base + LoRA · YOLO11n fine-tuned  
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        theme=gr.themes.Soft(),
    )
