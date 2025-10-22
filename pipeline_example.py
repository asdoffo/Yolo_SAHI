"""
Script de ejemplo para ejecutar el pipeline completo
Útil para testing rápido y demostración
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

# Importar los módulos del proyecto
from dataset_preparation import DatasetTiler
from train_model import YOLOTrainer
from inference_sahi import PeanutInspector


def run_complete_pipeline():
    """
    Ejecuta el pipeline completo de principio a fin
    """
    print("\n" + "="*60)
    print("🚀 EJECUTANDO PIPELINE COMPLETO DE DETECCIÓN")
    print("="*60)
    
    # 1. Crear dataset de ejemplo
    print("\n📁 PASO 1: Preparación del Dataset")
    print("-"*40)
    
    # Verificar si existe dataset real
    if not Path('datasets/original/images/train').exists():
        print("⚠️ No se encontró dataset")
    else:
        source_dir = 'datasets/original'
        print("✅ Usando dataset real encontrado")
    
    # 2. Aplicar tiling al dataset
    print("\n🔲 PASO 2: Aplicando Tiling (640x640)")
    print("-"*40)
    
    tiler = DatasetTiler(
        source_dir=source_dir,
        output_dir='datasets/tiled',
        tile_size=640,
        overlap=0.2
    )
    
    tiler.process_dataset(keep_empty_tiles=False)
    
    # Detectar clases automáticamente o usar las del YAML original
    tiler.create_data_yaml(source_yaml=f'{source_dir}/data.yaml' if Path(f'{source_dir}/data.yaml').exists() else None)
    tiler.print_statistics()
    
   
    

def interactive_demo():
    """
    Demo interactiva para probar el sistema
    """
    print("\n" + "="*60)
    print("🎮 DEMO INTERACTIVA")
    print("="*60)
    
    # Verificar si existe modelo
    if not Path('models/best.pt').exists():
        print("❌ No se encontró modelo entrenado.")
        print("   Ejecuta primero: python pipeline_example.py --train")
        return
    
    # Cargar inspector
    inspector = PeanutInspector(
        model_path='models/best.pt',
        confidence_threshold=0.25
    )
    
    while True:
        print("\n📷 Opciones:")
        print("1. Procesar imagen")
        print("2. Procesar carpeta")
        print("3. Capturar de cámara")
        print("4. Salir")
        
        choice = input("\nSelecciona opción (1-4): ")
        
        if choice == '1':
            image_path = input("Ruta de la imagen: ")
            if Path(image_path).exists():
                detections, _ = inspector.detect_with_sahi(
                    image_path,
                    visualize=True
                )
                print(f"✅ Detectados {len(detections)} objetos")
            else:
                print("❌ Imagen no encontrada")
        
        elif choice == '2':
            folder_path = input("Ruta de la carpeta: ")
            if Path(folder_path).exists():
                results = inspector.process_batch(folder_path)
                print(f"✅ Procesadas {len(results)} imágenes")
            else:
                print("❌ Carpeta no encontrada")
        
        elif choice == '3':
            print("📸 Presiona 'q' para salir, 'ESPACIO' para capturar")
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow('Cámara - Presiona ESPACIO para capturar', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Guardar y procesar
                    cv2.imwrite('capture.jpg', frame)
                    detections, _ = inspector.detect_with_sahi('capture.jpg')
                    print(f"✅ Detectados {len(detections)} objetos")
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '4':
            print("👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline de ejemplo para detección de cuerpos extraños')
    parser.add_argument('--full', action='store_true', 
                       help='Ejecutar pipeline completo')
       
    args = parser.parse_args()
    
    if args.full:
        run_complete_pipeline()
    