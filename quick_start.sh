#!/bin/bash

echo "============================================"
echo "🚀 INICIO RÁPIDO - DETECCIÓN EN MANÍ"
echo "============================================"
echo ""

# Verificar Python
echo "📌 Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no encontrado. Por favor instala Python 3.8+"
    exit 1
fi
echo "✅ Python3 encontrado: $(python3 --version)"
echo ""

# Crear entorno virtual (opcional pero recomendado)
echo "📌 Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Entorno virtual creado"
else
    echo "✅ Entorno virtual ya existe"
fi

# Activar entorno
echo "📌 Activando entorno virtual..."
source venv/bin/activate
echo "✅ Entorno activado"
echo ""

# Instalar dependencias
echo "📌 Instalando dependencias..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dependencias instaladas"
echo ""

# Crear estructura de carpetas
echo "📌 Creando estructura de carpetas..."
mkdir -p datasets/original/images/train
mkdir -p datasets/original/images/val
mkdir -p datasets/original/labels/train
mkdir -p datasets/original/labels/val
mkdir -p datasets/tiled
mkdir -p models
mkdir -p output
echo "✅ Estructura creada"
echo ""

echo "============================================"
echo "✨ SISTEMA LISTO PARA USAR"
echo "============================================"
echo ""
echo "🎯 COMANDOS RÁPIDOS:"
echo ""
echo "1. TEST RÁPIDO (con datos sintéticos):"
echo "   python pipeline_example.py --quick"
echo "   python pipeline_example.py --full"
echo ""
echo "2. CON TUS PROPIOS DATOS:"
echo "   a) Coloca imágenes en: datasets/original/images/train/"
echo "   b) Coloca anotaciones en: datasets/original/labels/train/"
echo "   c) Ejecuta:"
echo "      python dataset_preparation.py --source datasets/original --output datasets/tiled"
echo "      python train_model.py --data datasets/tiled/data.yaml --epochs 50"
echo "      python inference_sahi.py --model models/best.pt --image tu_imagen.jpg"
echo ""
echo "3. DEMO INTERACTIVA:"
echo "   python pipeline_example.py --demo"
echo ""
echo "📚 Para más información, consulta README.md"
echo ""
