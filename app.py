"""
app.py — Aplicación Gradio para Aircraft Skin Defect Classifier.

Despliegue:
    - Local:  python app.py
    - HuggingFace Spaces: push este archivo + modelo a tu Space

Funcionalidades:
    - Clasificación de 5 tipos de defectos
    - Mapa de calor Grad-CAM
    - Diagnóstico descriptivo con recomendaciones
"""

import json
from pathlib import Path

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
META_FILE = BASE_DIR / "results" / "models" / "deploy" / "deploy_meta.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFECT_DESCRIPTIONS = {
    "crack": (
        "**Grieta (Crack)**\n\n"
        "Fractura visible en la superficie metálica de la aeronave. "
        "Este tipo de daño puede comprometer la integridad estructural.\n\n"
        "**Severidad**: Alta\n"
        "**Acción recomendada**: Inspección detallada inmediata. "
        "Evaluar longitud, profundidad y propagación. Posible reparación estructural."
    ),
    "dent": (
        "**Abolladura (Dent)**\n\n"
        "Deformación cóncava en la superficie del fuselaje. "
        "Puede ser causada por impacto de objetos, granizo o manejo inadecuado.\n\n"
        "**Severidad**: Media-Alta\n"
        "**Acción recomendada**: Medir profundidad y diámetro. "
        "Comparar con límites de tolerancia del SRM (Structural Repair Manual)."
    ),
    "scratch": (
        "**Rayón (Scratch)**\n\n"
        "Daño superficial lineal en la piel del avión. "
        "Generalmente causado por contacto con herramientas o equipaje.\n\n"
        "**Severidad**: Baja-Media\n"
        "**Acción recomendada**: Verificar profundidad. Si penetra el primer "
        "tratamiento anticorrosivo, requiere re-tratamiento y sellado."
    ),
    "missing_head": (
        "**Cabeza de Remache Faltante (Missing Head)**\n\n"
        "Remache sin cabeza visible, indicando falla en la sujeción. "
        "Compromete la unión de paneles estructurales.\n\n"
        "**Severidad**: Alta\n"
        "**Acción recomendada**: Reemplazo inmediato del remache. "
        "Inspeccionar remaches adyacentes por posible efecto dominó."
    ),
    "paint_off": (
        "**Desprendimiento de Pintura (Paint Off)**\n\n"
        "Pérdida del recubrimiento protector de la superficie. "
        "Expone el metal base a condiciones ambientales y corrosión.\n\n"
        "**Severidad**: Baja-Media\n"
        "**Acción recomendada**: Limpiar área afectada, aplicar tratamiento "
        "anticorrosivo y repintar según especificaciones."
    ),
}


# ---------------------------------------------------------------------------
# Cargar modelo
# ---------------------------------------------------------------------------
def load_model():
    """Carga el modelo ViT fine-tuned para inferencia."""
    if MODEL_DIR.exists():
        model = ViTForImageClassification.from_pretrained(
            str(MODEL_DIR),
            num_labels=NUM_CLASSES,
        )
    else:
        # Fallback: cargar desde HuggingFace base (para demo sin pesos entrenados)
        print("⚠ Modelo entrenado no encontrado. Usando modelo base sin fine-tuning.")
        print(f"  Buscado en: {MODEL_DIR.resolve()}")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )

    model = model.to(DEVICE).eval()
    return model


model = load_model()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Grad-CAM setup
# ---------------------------------------------------------------------------
def vit_reshape_transform(tensor, height=14, width=14):
    """Transforma salida ViT [B, N+1, D] → [B, D, H, W] para Grad-CAM."""
    result = tensor[:, 1:, :]  # Quitar CLS token
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class HFModelWrapper(nn.Module):
    """Wrapper para que Grad-CAM funcione con modelos HuggingFace."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        return self.model(pixel_values=x).logits


wrapped_model = HFModelWrapper(model).to(DEVICE).eval()
target_layers = [wrapped_model.model.vit.encoder.layer[-1].layernorm_before]

cam = GradCAM(
    model=wrapped_model,
    target_layers=target_layers,
    reshape_transform=vit_reshape_transform,
)


# ---------------------------------------------------------------------------
# Función de predicción
# ---------------------------------------------------------------------------
def predict_defect(image):
    """
    Analiza una imagen de superficie de aeronave y retorna:
    1. Mapa Grad-CAM superpuesto
    2. Clasificación con probabilidades
    3. Diagnóstico descriptivo
    """
    if image is None:
        return None, {}, "⚠ Por favor sube una imagen para analizar."

    # Convertir a PIL
    img_pil = Image.fromarray(image).convert("RGB")

    # Preprocesar
    input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    # Imagen RGB [0,1] para superposición
    rgb_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    # Predicción
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).item())

    # Grad-CAM
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Probabilidades como dict
    label_probs = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}

    # Diagnóstico
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx]
    description = DEFECT_DESCRIPTIONS.get(pred_class, "Tipo de defecto no reconocido.")
    description += f"\n\n---\n**Confianza del modelo**: {confidence:.1%}"

    if confidence < 0.5:
        description += (
            "\n\n⚠ **Nota**: La confianza es baja. "
            "Se recomienda inspección manual adicional."
        )

    return cam_image, label_probs, description


# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Aircraft Skin Defect Classifier",
) as demo:
    gr.Markdown(
        """
        # Aircraft Skin Defect Classifier
        
        Sistema de IA para **detección y clasificación de defectos** en superficies 
        de aeronaves mediante inspección visual automatizada.
        
        **Modelo**: Vision Transformer (ViT) fine-tuned con LoRA/PEFT  
        **Clases**: Crack · Dent · Scratch · Missing Head · Paint Off
        
        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Sube imagen de inspección",
                type="numpy",
            )
            submit_btn = gr.Button("Analizar Defecto", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_cam = gr.Image(label="Grad-CAM — Mapa de Atención")
            output_label = gr.Label(num_top_classes=5, label="Clasificación")

    with gr.Row():
        output_desc = gr.Markdown(label="Diagnóstico")

    submit_btn.click(
        fn=predict_defect,
        inputs=[input_image],
        outputs=[output_cam, output_label, output_desc],
    )

    input_image.change(
        fn=predict_defect,
        inputs=[input_image],
        outputs=[output_cam, output_label, output_desc],
    )

    gr.Markdown(
        """
        ---
        **Proyecto Final — Deep Learning** · Maestría en Ciencia de Datos  
        Dataset: Aircraft Skin Defects (Roboflow, CC BY 4.0)  
        Modelo: google/vit-base-patch16-224 + LoRA  
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
