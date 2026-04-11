# Guía de Uso — Aircraft Skin Defect Classifier (App Gradio)

## Descripción

Aplicación web de IA para la **detección y clasificación automatizada de defectos** en superficies de aeronaves. El sistema analiza fotografías de inspección y produce:

1. **Clasificación del defecto** con probabilidades por clase
2. **Mapa de calor Grad-CAM** que resalta la zona del defecto detectado
3. **Diagnóstico descriptivo** con severidad y acción recomendada

## Requisitos

- Python 3.10+
- GPU NVIDIA (opcional, funciona en CPU)
- Modelo entrenado en `results/models/deploy/vit_final/`

### Instalación de dependencias

```bash
cd aircraft-skin-defect-classifier
pip install -r requirements.txt
```

Dependencias principales: `torch`, `transformers`, `gradio`, `grad-cam`, `Pillow`.

## Ejecución

### Local

```bash
cd aircraft-skin-defect-classifier
python app.py
```

La aplicación se abre en **http://localhost:7860**.

### HuggingFace Spaces

Para desplegar en la nube, subir `app.py` + `results/models/deploy/` a un Space de HuggingFace con SDK Gradio.

## Interfaz

La interfaz tiene tres zonas principales:

```
┌──────────────────────────────────────────────────────┐
│  Aircraft Skin Defect Classifier                     │
├────────────────────┬─────────────────────────────────┤
│                    │  Grad-CAM — Mapa de Atención    │
│  Sube imagen de    │  (mapa de calor superpuesto)    │
│  inspección        │                                 │
│  [Arrastrar/Subir] ├─────────────────────────────────┤
│                    │  Clasificación del Defecto       │
│  [Analizar Defecto]│  crack     ████████████ 99.3%   │
│                    │  dent      █            0.03%   │
│                    │  ...                             │
├────────────────────┴─────────────────────────────────┤
│  Diagnóstico                                         │
│  Grieta (Crack): Fractura visible en la superficie   │
│  metálica... Severidad: Alta                         │
│  Acción: Inspección detallada inmediata.             │
└──────────────────────────────────────────────────────┘
```

## Cómo usar

### Paso 1: Subir imagen
- Arrastra una imagen de la superficie de una aeronave al panel izquierdo
- Formatos aceptados: JPG, PNG, JPEG, BMP, WEBP
- La imagen puede ser de cualquier resolución (se redimensiona a 224×224 internamente)

### Paso 2: Analizar
- Haz clic en **"Analizar Defecto"** (o la predicción se ejecuta automáticamente al subir)
- Tiempo de inferencia: ~30-40 ms en GPU, ~200-400 ms en CPU

### Paso 3: Interpretar resultados

| Salida | Descripción |
|--------|-------------|
| **Grad-CAM** | Mapa de calor superpuesto sobre la imagen. Las zonas rojas/amarillas indican dónde el modelo detecta el defecto |
| **Clasificación** | Barra de probabilidades para cada una de las 5 clases |
| **Diagnóstico** | Descripción del defecto, nivel de severidad y acción recomendada |

## Clases de defectos

| Clase | Nombre | Severidad | Descripción |
|-------|--------|-----------|-------------|
| `crack` | Grieta | **Alta** | Fractura en la superficie metálica. Compromete integridad estructural |
| `dent` | Abolladura | **Media-Alta** | Deformación cóncava por impacto. Evaluar profundidad según SRM |
| `missing_head` | Remache faltante | **Alta** | Cabeza de remache ausente. Falla en sujeción de paneles |
| `scratch` | Rayón | **Baja-Media** | Daño superficial lineal. Revisar si penetra capa anticorrosiva |
| `paint_off` | Pintura desprendida | **Baja-Media** | Pérdida de recubrimiento. Riesgo de corrosión |

## Modelo

- **Arquitectura**: Vision Transformer (ViT-Base, patch 16×16, 224px)
- **Fine-tuning**: LoRA/PEFT (r=8, α=16) sobre capas de atención (query, value)
- **Parámetros entrenables**: 298K (0.35% del total de 86M)
- **Accuracy en test**: 94.81%
- **F1-Macro en test**: 95.11%

## Resultados de prueba funcional

Prueba ejecutada el 2026-04-09 con una imagen por clase del set de test:

| Imagen | Clase Real | Predicción | Confianza | Latencia |
|--------|-----------|------------|-----------|----------|
| crack_05463.jpg | crack | crack | 99.3% | 245 ms |
| dent_03722.jpg | dent | dent | 96.0% | 35 ms |
| scratch_00658.jpg | scratch | scratch | 53.6% | 32 ms |
| missing_head_02431.jpg | missing_head | missing_head | 100.0% | 31 ms |
| paint_off_03626.jpg | paint_off | paint_off | 100.0% | 30 ms |

**Resultado: 5/5 correctos (100%)**

> Nota: La clase `scratch` muestra menor confianza (53.6%) porque comparte características visuales con `paint_off`. 
> Esto es esperado y el modelo incluye advertencias automáticas cuando la confianza es < 50%.

## Evidencia visual

La evidencia generada se encuentra en:
- **Imagen compuesta**: `results/app_test/test_evidence.png`
- **Reporte JSON**: `results/app_test/test_report.json`

## Prueba automatizada

Para ejecutar la prueba funcional completa:

```bash
python test_app.py
```

Genera automáticamente la figura de evidencia y el reporte JSON.

## Arquitectura del sistema

```
Imagen de entrada (JPG/PNG)
        │
        ▼
┌──────────────────┐
│  Preprocessing   │  Resize 224×224 → ToTensor → Normalize (ImageNet)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ViT + LoRA      │  Vision Transformer con adaptadores de bajo rango
│  (86M params,    │  en capas de atención query/value
│   298K trainable)│
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│ Logits │ │ Grad-CAM │  Mapa de atención sobre última capa del encoder
│ → Softmax│ │          │
└────┬───┘ └─────┬────┘
     │           │
     ▼           ▼
┌────────────────────┐
│  Post-procesamiento│
│  - Top-5 probs     │
│  - Heatmap overlay │
│  - Diagnóstico     │
└────────────────────┘
```

## Limitaciones

- El modelo clasifica **un tipo de defecto por imagen** (no detección multi-defecto)
- Entrenado con imágenes de close-up/recortes de bounding box, no con imágenes panorámicas completas del fuselaje
- La clase `scratch` tiene menor volumen de datos de entrenamiento (1,051 muestras vs 7,000+ de otras clases)
- Requiere el modelo entrenado en `results/models/deploy/vit_final/`; sin él, carga un ViT base sin fine-tuning

## Troubleshooting

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: gradio` | `pip install gradio` |
| `ModuleNotFoundError: pytorch_grad_cam` | `pip install grad-cam` |
| Modelo no encontrado | Verificar que `results/models/deploy/vit_final/` contenga `config.json` y `model.safetensors` |
| Puerto 7860 en uso | Cambiar `server_port` en `app.py` o matar el proceso anterior |
| Predicción incorrecta | Verificar que la imagen sea de superficie de aeronave (close-up del defecto) |
