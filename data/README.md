# Datos — Aircraft Skin Defect Classifier

## Fuentes de Datos

### Dataset Principal
- **aircraft-skin-defects-merged-final** — Dibya Dillip
  - URL: https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-merged-final
  - ~2,000 imágenes | 5 clases | Object Detection | CC BY 4.0

### Dataset Complementario
- **aircraft-skin-defects** — AI Assistant for General Visual Inspection
  - URL: https://universe.roboflow.com/ai-assistant-for-general-visual-inspection-of-airplanes/aircraft-skin-defects
  - ~4,600 imágenes | 5 clases | Object Detection | CC BY 4.0

### Clases Unificadas
| Clase | Descripción |
|-------|-------------|
| `crack` | Grietas en la superficie |
| `dent` | Abolladuras |
| `scratch` | Rayones superficiales |
| `missing_head` | Cabezas de remaches faltantes |
| `paint_off` | Desprendimiento de pintura |

## Descarga

Ejecutar el notebook `01_EDA_MLP_Base.ipynb` que contiene el código de descarga
automática vía Roboflow API, o bien ejecutar:

```bash
python src/data_utils.py --download --api-key TU_API_KEY
```

## Estructura tras descarga y preprocesamiento

```
data/
├── raw/                    # Datos descargados sin modificar
│   ├── dataset1/           # aircraft-skin-defects-merged-final
│   └── dataset2/           # aircraft-skin-defects (AI Assistant)
├── processed/              # Parches recortados por clase
│   ├── crack/
│   ├── dent/
│   ├── scratch/
│   ├── missing_head/
│   └── paint_off/
└── splits/                 # CSVs con rutas y etiquetas
    ├── train.csv
    ├── val.csv
    └── test.csv
```

**NOTA**: Las imágenes NO se suben al repositorio. Solo se incluyen scripts de descarga.
Agrega `data/raw/` y `data/processed/` a `.gitignore`.
