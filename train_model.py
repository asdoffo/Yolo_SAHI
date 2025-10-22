"""
Script de entrenamiento de YOLOv8 optimizado para detección de objetos pequeños
Configurado específicamente para detectar cuerpos extraños en maní
"""

import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

class YOLOTrainer:
    def __init__(self, data_yaml, model_size='x', device='0'):
        """
        Inicializa el entrenador de YOLO
        
        Args:
            data_yaml: Path al archivo data.yaml
            model_size: Tamaño del modelo (n, s, m, l, x)
            device: Dispositivo a usar ('0' para GPU 0, 'cpu' para CPU)
        """
        self.data_yaml = Path(data_yaml)
        self.model_size = model_size
        self.device = device
        self.project_name = f"yolo_peanuts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Verificar disponibilidad de GPU
        if device != 'cpu' and not torch.cuda.is_available():
            print("⚠️ GPU no disponible, usando CPU")
            self.device = 'cpu'
        else:
            print(f"✅ Usando dispositivo: {self.device}")
            if self.device != 'cpu':
                print(f"   GPU: {torch.cuda.get_device_name(int(self.device))}")
                print(f"   VRAM: {torch.cuda.get_device_properties(int(self.device)).total_memory / 1e9:.2f} GB")
    
    def get_optimized_hyperparameters(self):
        """
        Retorna hiperparámetros optimizados para objetos pequeños
        """
        return {
            # Optimizaciones para objetos pequeños
            'box': 7.5,  # Mayor peso a la pérdida de bbox
            'cls': 1.5,  # Peso de pérdida de clasificación
            'dfl': 1.5,  # Distribution Focal Loss
            
            # Learning rate
            'lr0': 0.001,  # Learning rate inicial más conservador
            'lrf': 0.01,   # Learning rate final
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Augmentation optimizada para objetos pequeños
            'hsv_h': 0.015,  # Variación de tono
            'hsv_s': 0.4,    # Variación de saturación
            'hsv_v': 0.4,    # Variación de brillo
            'degrees': 10.0,  # Rotación aleatoria
            'translate': 0.1, # Traslación
            'scale': 0.3,    # Escala reducida para no perder objetos pequeños
            'shear': 2.0,    # Cizallamiento
            'perspective': 0.0001,  # Perspectiva
            'flipud': 0.5,   # Flip vertical
            'fliplr': 0.5,   # Flip horizontal
            'mosaic': 0.5,   # Reducido para objetos pequeños
            'mixup': 0.1,    # Mixup para robustez
            'copy_paste': 0.3,  # Copy-paste augmentation (muy útil para objetos pequeños)
            
            # Optimizaciones de entrenamiento
            'close_mosaic': 15,  # Desactivar mosaic en las últimas épocas
            'amp': True,     # Automatic Mixed Precision para mayor velocidad
        }
    
    def train_model(self, epochs=200, img_size=640, batch_size=None, resume=False):
        """
        Entrena el modelo YOLO
        
        Args:
            epochs: Número de épocas
            img_size: Tamaño de imagen para entrenamiento
            batch_size: Tamaño del batch (None para auto)
            resume: Si continuar entrenamiento previo
        """
        # Cargar modelo preentrenado
        if resume and Path(f"runs/detect/{self.project_name}/weights/last.pt").exists():
            model_path = f"runs/detect/{self.project_name}/weights/last.pt"
            print(f"📂 Resumiendo entrenamiento desde: {model_path}")
            model = YOLO(model_path)
        else:
            model_path = f'yolov8{self.model_size}.pt'
            print(f"📦 Cargando modelo base: {model_path}")
            model = YOLO(model_path)
        
        # Obtener hiperparámetros optimizados
        hyperparams = self.get_optimized_hyperparameters()
        
        # Auto-batch si no se especifica
        if batch_size is None:
            if self.device != 'cpu':
                # Estimar batch size basado en VRAM disponible
                vram_gb = torch.cuda.get_device_properties(int(self.device)).total_memory / 1e9
                if img_size == 640:
                    batch_size = min(16, int(vram_gb * 2))  # Aproximación
                elif img_size == 1280:
                    batch_size = min(8, int(vram_gb * 0.8))
                else:
                    batch_size = 8
            else:
                batch_size = 4
        
        print(f"\n🚀 Iniciando entrenamiento:")
        print(f"   - Épocas: {epochs}")
        print(f"   - Tamaño de imagen: {img_size}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Proyecto: {self.project_name}")
        
        # Entrenar modelo
        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            project='runs/detect',
            name=self.project_name,
            exist_ok=resume,
            resume=resume,
            patience=30,  # Early stopping
            save=True,
            save_period=10,  # Guardar cada 10 épocas
            cache=False,  # Cache de imágenes (usar True si tienes RAM suficiente)
            workers=8,
            pretrained=True,
            optimizer='AdamW',  # Mejor para fine-tuning
            verbose=True,
            seed=42,
            deterministic=False,
            single_cls=False,
            rect=False,
            cos_lr=True,  # Cosine learning rate scheduler
            fraction=1.0,
            profile=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.1,  # Dropout para regularización
            val=True,
            plots=True,
            **hyperparams  # Aplicar hiperparámetros optimizados
        )
        
        print(f"\n✅ Entrenamiento completado")
        print(f"📊 Mejores pesos guardados en: runs/detect/{self.project_name}/weights/best.pt")
        
        return results
    
    def evaluate_model(self, model_path=None):
        """
        Evalúa el modelo entrenado
        
        Args:
            model_path: Path al modelo (None para usar el último entrenado)
        """
        if model_path is None:
            model_path = f"runs/detect/{self.project_name}/weights/best.pt"
        
        print(f"\n📊 Evaluando modelo: {model_path}")
        
        model = YOLO(model_path)
        
        # Validar en el conjunto de validación
        metrics = model.val(
            data=str(self.data_yaml),
            imgsz=640,
            batch=8,
            device=self.device,
            plots=True,
            verbose=True
        )
        
        # Mostrar métricas
        print("\n📈 Métricas de Evaluación:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        # Guardar métricas en JSON
        metrics_dict = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'classes': metrics.box.ap_class_index.tolist() if hasattr(metrics.box, 'ap_class_index') else []
        }
        
        metrics_path = Path(f"runs/detect/{self.project_name}/metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        return metrics
    
    def export_model(self, format='onnx', optimize=True):
        """
        Exporta el modelo a diferentes formatos
        
        Args:
            format: Formato de exportación (onnx, tensorrt, coreml, etc.)
            optimize: Si optimizar el modelo
        """
        model_path = f"runs/detect/{self.project_name}/weights/best.pt"
        model = YOLO(model_path)
        
        print(f"\n📦 Exportando modelo a formato {format}...")
        
        # Configuración de exportación para objetos pequeños
        export_args = {
            'format': format,
            'imgsz': 640,
            'half': False,  # FP16 (usar True si tu hardware lo soporta)
            'dynamic': True,  # Entrada dinámica
            'simplify': optimize,
            'opset': 17 if format == 'onnx' else None,
            'workspace': 4,  # GB para TensorRT
            'nms': True,  # Incluir NMS en el modelo
        }
        
        # Exportar
        export_path = model.export(**export_args)
        print(f"✅ Modelo exportado a: {export_path}")
        
        return export_path


def main():
    parser = argparse.ArgumentParser(description='Entrenar YOLOv8 para detección de objetos pequeños')
    parser.add_argument('--data', type=str, required=True,
                       help='Path al archivo data.yaml')
    parser.add_argument('--model', type=str, default='x',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Tamaño del modelo YOLO (default: x)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Número de épocas (default: 200)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Tamaño de imagen para entrenamiento (default: 640)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: auto)')
    parser.add_argument('--device', type=str, default='0',
                       help='Dispositivo GPU/CPU (default: 0)')
    parser.add_argument('--resume', action='store_true',
                       help='Resumir entrenamiento previo')
    parser.add_argument('--evaluate', action='store_true',
                       help='Solo evaluar modelo existente')
    parser.add_argument('--export', type=str, default=None,
                       choices=['onnx', 'tensorrt', 'coreml', 'tflite'],
                       help='Exportar modelo al formato especificado')
    
    args = parser.parse_args()
    
    # Crear entrenador
    trainer = YOLOTrainer(
        data_yaml=args.data,
        model_size=args.model,
        device=args.device
    )
    
    if args.evaluate:
        # Solo evaluar
        trainer.evaluate_model()
    else:
        # Entrenar
        trainer.train_model(
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            resume=args.resume
        )
        
        # Evaluar después del entrenamiento
        trainer.evaluate_model()
    
    # Exportar si se solicita
    if args.export:
        trainer.export_model(format=args.export)


if __name__ == "__main__":
    main()
