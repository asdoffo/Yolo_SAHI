#!/bin/bash

# Script de instalación de dependencias para detección de objetos pequeños con YOLO
# Compatible con Ubuntu/Debian y sistemas basados en Linux

echo "================================================"
echo "🚀 Instalación de dependencias para YOLO + SAHI"
echo "================================================"

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# Instalar PyTorch (con CUDA 11.8)
echo "🔥 Instalando PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar Ultralytics YOLOv8
echo "🎯 Instalando YOLOv8..."
pip install ultralytics

# Instalar SAHI (Slicing Aided Hyper Inference)
echo "🔍 Instalando SAHI..."
pip install sahi

# Instalar dependencias adicionales
echo "📚 Instalando dependencias adicionales..."
pip install opencv-python
pip install opencv-contrib-python
pip install Pillow
pip install matplotlib
pip install seaborn
pip install tqdm
pip install PyYAML
pip install pandas
pip install scipy

# Instalar herramientas de anotación opcionales
echo "🏷️ Instalando herramientas opcionales..."
pip install labelImg  # Para anotación manual
pip install roboflow  # Para gestión de datasets

# Verificar instalación de CUDA (opcional)
echo ""
echo "🖥️ Verificando disponibilidad de GPU..."
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
    python -c "import torch; print('CUDA Version:', torch.version.cuda)"
else
    echo "⚠️ GPU no detectada. El entrenamiento será más lento en CPU."
fi

# Crear estructura de carpetas
echo ""
echo "📁 Creando estructura de carpetas..."
mkdir -p datasets/original/images/train
mkdir -p datasets/original/images/val
mkdir -p datasets/original/labels/train
mkdir -p datasets/original/labels/val
mkdir -p datasets/tiled
mkdir -p models
mkdir -p output
mkdir -p runs

echo ""
echo "✅ Instalación completada!"
echo ""
echo "📝 Estructura de carpetas creada:"
echo "   datasets/"
echo "   ├── original/     # Dataset original (3840x2160)"
echo "   │   ├── images/"
echo "   │   │   ├── train/"
echo "   │   │   └── val/"
echo "   │   └── labels/"
echo "   │       ├── train/"
echo "   │       └── val/"
echo "   ├── tiled/        # Dataset procesado (640x640)"
echo "   models/           # Modelos entrenados"
echo "   output/           # Resultados de inferencia"
echo "   runs/             # Logs de entrenamiento"
echo ""
echo "🚀 ¡Listo para comenzar!"
echo "   1. Coloca tus imágenes y anotaciones en datasets/original/"
echo "   2. Ejecuta: python dataset_preparation.py"
echo "   3. Entrena: python train_model.py"
echo "   4. Infiere: python inference_sahi.py"
