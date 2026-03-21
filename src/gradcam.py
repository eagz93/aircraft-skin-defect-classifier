"""
gradcam.py — Implementación de Grad-CAM para visualización de zonas de atención.

Soporta:
- CNN custom
- ResNet50 (torchvision)
- ViT (Hugging Face) via pytorch-grad-cam
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ---------------------------------------------------------------------------
# Obtener capa objetivo por tipo de modelo
# ---------------------------------------------------------------------------
def get_target_layer(model, model_type: str = "cnn"):
    """
    Devuelve la capa objetivo para Grad-CAM según el tipo de modelo.

    Args:
        model: El modelo de PyTorch
        model_type: 'cnn', 'cnn_deep', 'resnet50', 'vit'
    """
    if model_type in ("cnn", "cnn_deep"):
        # Última capa convolucional del bloque features
        return [model.features[-1].block[0]]

    elif model_type == "resnet50":
        return [model.layer4[-1]]

    elif model_type == "vit":
        # Para ViT de Hugging Face, usar última capa del encoder
        return [model.vit.encoder.layer[-1].layernorm_before]

    else:
        raise ValueError(f"model_type '{model_type}' no soportado")


# ---------------------------------------------------------------------------
# Reshape transform para ViT
# ---------------------------------------------------------------------------
def vit_reshape_transform(tensor, height=14, width=14):
    """
    Transforma la salida del ViT para Grad-CAM.
    ViT produce tokens [B, N+1, D], necesitamos [B, D, H, W].
    """
    result = tensor[:, 1:, :]  # Quitar CLS token
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


# ---------------------------------------------------------------------------
# Grad-CAM principal
# ---------------------------------------------------------------------------
def generate_gradcam(
    model,
    image: Image.Image,
    model_type: str = "cnn",
    target_class: int | None = None,
    img_size: int = 224,
    device: str = "cpu",
):
    """
    Genera un mapa Grad-CAM para una imagen dada.

    Args:
        model: Modelo entrenado
        image: PIL Image
        model_type: 'cnn', 'cnn_deep', 'resnet50', 'vit'
        target_class: Clase objetivo (None = clase predicha)
        img_size: Tamaño de entrada del modelo
        device: Dispositivo

    Returns:
        cam_image: numpy array con el heatmap superpuesto
        prediction: clase predicha
        probabilities: probabilidades por clase
    """
    model.eval()
    model = model.to(device)

    # Preprocesar imagen
    mean = [0.485, 0.456, 0.406]
    std_ = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std_),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Imagen RGB normalizada [0,1] para superposición
    rgb_img = np.array(image.resize((img_size, img_size))).astype(np.float32) / 255.0

    # Predicción
    is_hf = model_type == "vit"
    with torch.no_grad():
        if is_hf:
            outputs = model(pixel_values=input_tensor)
            logits = outputs.logits
        else:
            logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = int(logits.argmax(dim=1).item())

    # Target para Grad-CAM
    if target_class is None:
        target_class = prediction
    targets = [ClassifierOutputTarget(target_class)]

    # Capa objetivo y reshape
    target_layers = get_target_layer(model, model_type)

    reshape_fn = None
    if model_type == "vit":
        reshape_fn = vit_reshape_transform

    # Wrapper para modelos HF
    class HFModelWrapper(torch.nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.model = hf_model

        def forward(self, x):
            return self.model(pixel_values=x).logits

    if is_hf:
        wrapped = HFModelWrapper(model).to(device)
        target_layers = [wrapped.model.vit.encoder.layer[-1].layernorm_before]
        cam = GradCAM(
            model=wrapped,
            target_layers=target_layers,
            reshape_transform=reshape_fn,
        )
    else:
        cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image, prediction, probabilities


def plot_gradcam(
    image: Image.Image,
    cam_image: np.ndarray,
    prediction: int,
    probabilities: np.ndarray,
    class_names: list,
    title: str = "Grad-CAM Visualization",
    save_path=None,
):
    """Plotea la imagen original, el Grad-CAM y las probabilidades."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Imagen original
    axes[0].imshow(image.resize((224, 224)))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Grad-CAM
    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM — Pred: {class_names[prediction]}")
    axes[1].axis("off")

    # Probabilidades
    colors = ["#e74c3c" if i == prediction else "#3498db" for i in range(len(class_names))]
    bars = axes[2].barh(class_names, probabilities, color=colors)
    axes[2].set_xlabel("Probability")
    axes[2].set_title("Class Probabilities")
    axes[2].set_xlim(0, 1)
    for bar, prob in zip(bars, probabilities):
        axes[2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.3f}", va="center")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
