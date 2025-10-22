# 🥜 Sistema de Detección de Cuerpos Extraños en Maní

Pipeline completo para detectar objetos pequeños (piedras, palitos, metal, plástico) en imágenes de alta resolución de maní usando YOLOv8 y SAHI (Slicing Aided Hyper Inference).

## 📋 Características

- ✅ **Procesamiento de imágenes de alta resolución** (3840x2160px)
- ✅ **Detección de objetos pequeños** (200x200px) con alta precisión
- ✅ **Estrategia de tiling/cropping** para mantener resolución
- ✅ **Post-procesamiento avanzado** con NMS optimizado
- ✅ **Comparación de métodos** (estándar vs SAHI)
- ✅ **Exportación a múltiples formatos** (ONNX, TensorRT, etc.)

## 🚀 Instalación Rápida

### Opción 1: Script automático
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### Opción 2: Con pip
```bash
pip install -r requirements.txt
```

### Opción 3: Manual
```bash
# PyTorch con CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Dependencias principales
pip install ultralytics sahi opencv-python matplotlib tqdm
```

## 📁 Estructura del Proyecto

```
proyecto/
├── datasets/
│   ├── original/          # Dataset original (3840x2160)
│   │   ├── images/
│   │   │   ├── train/     # Imágenes de entrenamiento
│   │   │   └── val/       # Imágenes de validación
│   │   └── labels/
│   │       ├── train/     # Anotaciones YOLO
│   │       └── val/
│   └── tiled/            # Dataset procesado (640x640)
├── models/               # Modelos entrenados
├── output/               # Resultados de inferencia
├── runs/                 # Logs de entrenamiento
└── scripts/
    ├── dataset_preparation.py  # Preparación con tiling
    ├── train_model.py         # Entrenamiento
    └── inference_sahi.py      # Inferencia
```

## 🔧 Pipeline Completo

### 1️⃣ Preparación del Dataset

Divide las imágenes grandes en tiles de 640x640 con overlap:

```bash
python dataset_preparation.py \
    --source datasets/original \
    --output datasets/tiled \
    --tile-size 640 \
    --overlap 0.2 \
    --classes piedra palito metal plastico
```

**Parámetros importantes:**
- `--tile-size`: Tamaño de cada tile (default: 640)
- `--overlap`: Superposición entre tiles (default: 0.2 = 20%)
- `--keep-empty`: Mantener tiles sin objetos (útil para reducir falsos positivos)

**Resultado esperado:**
- Una imagen de 3840x2160 genera ~42 tiles con 20% de overlap
- Las anotaciones se ajustan automáticamente para cada tile

### 2️⃣ Entrenamiento del Modelo

Entrena YOLOv8 optimizado para objetos pequeños:

```bash
python train_model.py \
    --data datasets/tiled/data.yaml \
    --model x \
    --epochs 200 \
    --img-size 640 \
    --batch-size 16 \
    --device 0
```

**Configuraciones recomendadas por GPU:**

| GPU | VRAM | Modelo | Batch Size | Img Size |
|-----|------|--------|------------|----------|
| RTX 3060 | 12GB | YOLOv8x | 8 | 640 |
| RTX 3080 | 10GB | YOLOv8l | 12 | 640 |
| RTX 4090 | 24GB | YOLOv8x | 16 | 1280 |
| T4 | 16GB | YOLOv8x | 12 | 640 |

**Hiperparámetros optimizados (ya incluidos):**
- Augmentation específica para objetos pequeños
- Copy-paste augmentation (0.3)
- Mosaic reducido (0.5)
- Mayor peso a pérdida de bbox (7.5)

### 3️⃣ Inferencia con SAHI

Procesa imágenes de alta resolución con slicing:

```bash
# Imagen individual
python inference_sahi.py \
    --model runs/detect/*/weights/best.pt \
    --image test_image.jpg \
    --conf 0.25 \
    --visualize

# Lote de imágenes
python inference_sahi.py \
    --model runs/detect/*/weights/best.pt \
    --folder test_images/ \
    --conf 0.25

# Comparar métodos
python inference_sahi.py \
    --model runs/detect/*/weights/best.pt \
    --image test_image.jpg \
    --compare
```

## 📊 Ejemplos de Uso

### Ejemplo 1: Pipeline Completo

```python
# 1. Preparar dataset
from dataset_preparation import DatasetTiler

tiler = DatasetTiler(
    source_dir="datasets/original",
    output_dir="datasets/tiled",
    tile_size=640,
    overlap=0.2
)
tiler.process_dataset(keep_empty_tiles=False)
tiler.create_data_yaml(['piedra', 'palito', 'metal', 'plastico'])

# 2. Entrenar modelo
from train_model import YOLOTrainer

trainer = YOLOTrainer(
    data_yaml="datasets/tiled/data.yaml",
    model_size='x',
    device='0'
)
trainer.train_model(epochs=200, img_size=640)

# 3. Inferencia
from inference_sahi import PeanutInspector

inspector = PeanutInspector(
    model_path="runs/detect/*/weights/best.pt",
    confidence_threshold=0.25
)
detections, result = inspector.detect_with_sahi(
    "test_image.jpg",
    slice_size=640,
    overlap_ratio=0.2
)
```

### Ejemplo 2: Inferencia en Tiempo Real

```python
import cv2
from inference_sahi import PeanutInspector

inspector = PeanutInspector("models/best.pt")

# Procesar video o stream
cap = cv2.VideoCapture("video.mp4")  # o 0 para webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detectar (puedes procesar cada N frames para mayor velocidad)
    detections, _ = inspector.detect_with_sahi(frame)
    
    # Dibujar resultados
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}",
                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Detección', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## ⚡ Optimización de Rendimiento

### Para Mayor Velocidad:
1. **Reducir overlap**: Usar 0.1 en lugar de 0.2
2. **Modelo más pequeño**: YOLOv8m o YOLOv8l
3. **Aumentar conf threshold**: 0.35 en lugar de 0.25
4. **Usar TensorRT**: Exportar modelo para GPU NVIDIA
5. **Procesamiento por lotes**: Procesar múltiples tiles simultáneamente

### Para Mayor Precisión:
1. **Aumentar overlap**: Usar 0.3 o 0.4
2. **Modelo más grande**: YOLOv8x
3. **Reducir conf threshold**: 0.15
4. **Aumentar épocas**: 300-500 épocas
5. **Data augmentation**: Más copy-paste y mixup

## 🔍 Comparación de Métodos

| Método | Ventajas | Desventajas | Uso Recomendado |
|--------|----------|-------------|-----------------|
| **YOLO Estándar** | Rápido (0.05s) | Pierde objetos pequeños | Objetos >50px |
| **SAHI (Nuestro)** | Alta precisión en objetos pequeños | Más lento (2-3s) | Objetos <50px |
| **Imagen completa 1280** | Balance velocidad/precisión | Requiere más VRAM | GPUs potentes |

## 📈 Métricas Esperadas

Con el pipeline optimizado deberías obtener:

- **mAP50**: 0.85-0.92
- **mAP50-95**: 0.65-0.75
- **Precisión**: 0.88-0.94
- **Recall**: 0.82-0.90
- **FPS**: 5-15 (dependiendo del hardware)

## 🐛 Solución de Problemas

### "CUDA out of memory"
```bash
# Reducir batch size
python train_model.py --batch-size 4

# O usar CPU
python train_model.py --device cpu
```

### "No se detectan objetos"
```python
# Reducir threshold de confianza
inspector = PeanutInspector(model_path, confidence_threshold=0.15)

# Aumentar overlap
detect_with_sahi(image, overlap_ratio=0.3)
```

### "Muchos falsos positivos"
```python
# Aumentar threshold
inspector = PeanutInspector(model_path, confidence_threshold=0.4)

# Mejorar NMS
detect_with_sahi(image, postprocess='GREEDYNMM')
```

## 📚 Referencias

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [SAHI Documentation](https://github.com/obss/sahi)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## 📝 Notas Importantes

1. **Calidad de Anotaciones**: La precisión depende mucho de la calidad de las anotaciones. Asegúrate de que todos los objetos estén correctamente etiquetados.

2. **Balance de Clases**: Si tienes desbalance (ej: muchas piedras, pocos plásticos), considera usar weighted loss o data augmentation específica.

3. **Validación Cruzada**: Usa diferentes splits de validación para asegurar que el modelo generaliza bien.

4. **Monitoreo**: Usa herramientas como Weights & Biases o TensorBoard para monitorear el entrenamiento.

## 🤝 Contribuciones

Si encuentras mejoras o tienes sugerencias, ¡son bienvenidas!

## 📄 Licencia

MIT License

---

**Desarrollado para detección de alta precisión en control de calidad de alimentos** 🥜✨
