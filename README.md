# Aircraft Skin Defect Classifier

Sistema progresivo de Deep Learning para la detección y clasificación de defectos en superficies de aeronaves, desarrollado como proyecto final del curso de Deep Learning — Maestría en Ciencia de Datos.

## Problema

La inspección visual de superficies de aeronaves es un proceso crítico en la industria aeronáutica. Este proyecto automatiza la clasificación de **5 tipos de defectos**:

| Clase | Descripción |
|-------|-------------|
| **Crack** | Grietas en la superficie metálica |
| **Dent** | Abolladuras por impacto |
| **Scratch** | Rayones superficiales |
| **Missing Head** | Cabezas de remaches faltantes |
| **Paint Off** | Desprendimiento de pintura |

## Arquitectura Progresiva

El sistema se construye en 5 etapas, cada una añadiendo complejidad:

```
Etapa 1: MLP Base          → Línea base con red densa
Etapa 2: CNN Profunda       → Extracción de features espaciales
Etapa 3: Vision Transformer → Transfer learning con ViT + ResNet50
Etapa 4: VAE / GAN          → Generación de datos sintéticos
Etapa 5: LoRA + Deploy      → Fine-tuning eficiente + Gradio demo
```

## Resultados

| Modelo | Accuracy | F1-Macro | Parámetros |
|--------|----------|----------|------------|
| MLP Base | — | — | — |
| CNN Custom | — | — | — |
| ResNet50 (fine-tuned) | — | — | — |
| ViT-Base (fine-tuned) | — | — | — |
| ViT + LoRA | — | — | — |
| ViT + Synthetic Data | — | — | — |

*(Se completará conforme avance el proyecto)*

## Estructura del Proyecto

```
aircraft-skin-defect-classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md                       # Instrucciones de descarga
├── notebooks/
│   ├── 01_EDA_MLP_Base.ipynb           # EDA + modelo MLP base
│   ├── 02_CNN_Profunda.ipynb           # CNN desde cero
│   ├── 03_Pretrained_Transformers.ipynb # ViT + ResNet50
│   ├── 04_Componente_Generativo.ipynb  # VAE / GAN
│   └── 05_FineTuning_Deploy.ipynb      # LoRA + cuantización + pruning
├── src/
│   ├── data_utils.py                   # Carga, preprocesamiento, augmentation
│   ├── models.py                       # MLP, CNN, wrappers
│   ├── train.py                        # Loop de entrenamiento
│   ├── evaluate.py                     # Métricas y visualizaciones
│   └── gradcam.py                      # Grad-CAM
├── app.py                              # Demo Gradio
├── configs/
│   └── default.yaml                    # Hiperparámetros
└── results/
    └── figures/                        # Gráficas generadas
```

## Instalación

```bash
git clone https://github.com/TU_USUARIO/aircraft-skin-defect-classifier.git
cd aircraft-skin-defect-classifier
pip install -r requirements.txt
```

## Datos

Los datos provienen de Roboflow Universe (licencia CC BY 4.0):

1. **aircraft-skin-defects-merged-final** (~2k imgs) — [Enlace](https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-merged-final)
2. **aircraft-skin-defects** (~4.6k imgs) — [Enlace](https://universe.roboflow.com/ai-assistant-for-general-visual-inspection-of-airplanes/aircraft-skin-defects)

Para descargar y preparar los datos:
```bash
# Opción 1: Ejecutar el notebook 01
# Opción 2: Script directo
python src/data_utils.py --download --api-key TU_ROBOFLOW_API_KEY
```

Ver [data/README.md](data/README.md) para más detalles.

## Uso

### Entrenamiento
Los notebooks en `notebooks/` contienen todo el pipeline de entrenamiento progresivo.
Ejecutar en orden: 01 → 02 → 03 → 04 → 05.

### Demo
```bash
python app.py
# Abre http://localhost:7860 en tu navegador
```

El demo permite:
- Subir una imagen de superficie de aeronave
- Ver la clasificación con probabilidades
- Visualizar un mapa de calor Grad-CAM resaltando la zona del defecto

## Demo en Línea

[HuggingFace Spaces](https://huggingface.co/spaces/TU_USUARIO/aircraft-skin-defect-classifier) *(se publicará en Semana 5)*

## Stack Tecnológico

- **PyTorch** + **Hugging Face Transformers** + **timm**
- **peft** (LoRA) + **bitsandbytes** (cuantización)
- **pytorch-grad-cam** para visualización
- **Gradio** para despliegue
- **Roboflow** para gestión de datos

## Licencia

MIT — ver [LICENSE](LICENSE)

## Citación de Datos

```bibtex
@misc{aircraft-skin-defects-merged-final_dataset,
    title = {aircraft-skin-defects-merged-final Dataset},
    author = {Dibya Dillip},
    howpublished = {\url{https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-merged-final}},
    journal = {Roboflow Universe},
    publisher = {Roboflow},
    year = {2023}
}
```
