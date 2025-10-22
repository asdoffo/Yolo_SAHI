"""
Script de inferencia optimizado para detección en imágenes de alta resolución
Utiliza SAHI (Slicing Aided Hyper Inference) para detectar objetos pequeños
Añadida GUI interactiva usando tkinter.
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
# from sahi.utils.cv import read_image # No usada directamente
# from sahi.utils.file import download_from_url # No usada directamente
import json
from datetime import datetime
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import yaml 
import os  

# ----------------- MÓDULOS PARA LA GUI -----------------
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
# -------------------------------------------------------

# La clase PeanutInspector se mantiene casi idéntica
class PeanutInspector:

    def _load_class_names_from_yaml(self):
        """
        Busca y lee el archivo data.yaml para obtener la lista de nombres de clases.
        Si no lo encuentra, pregunta al usuario por la ruta.
        """
        model_dir = self.model_path.parent
        # Intentos automáticos
        yaml_path_candidates = [
            model_dir / 'data.yaml',        # Mismo directorio
            model_dir.parent / 'data.yaml'  # Directorio padre (común en runs/detect/name/weights)
        ]
        
        found_path = None
        for path in yaml_path_candidates:
            if path.exists():
                found_path = path
                break
        
        # Si no se encuentra automáticamente, pedir al usuario
        if not found_path:
            print("⚠️ Advertencia: data.yaml no encontrado automáticamente. Solicitando ruta al usuario...")
            
            # Usar la función de diálogo de archivo definida previamente
            manual_path = filedialog.askopenfilename(
                title="Seleccionar archivo data.yaml",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
            )
            
            if manual_path:
                found_path = Path(manual_path)
            else:
                print("❌ Búsqueda de data.yaml cancelada por el usuario.")
                return None # Cancelado por el usuario

        yaml_path = found_path
        
        # Procesamiento del archivo YAML
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                class_names = data.get('names')
                
                if class_names is None:
                    raise KeyError("El archivo YAML no contiene la clave 'names'.")
                
                print(f"✅ Clases cargadas desde '{yaml_path.name}' ({len(class_names)} clases): {class_names}")
                return class_names
                
        except Exception as e:
            messagebox.showerror("Error de YAML", f"Error al leer o parsear data.yaml en {yaml_path}:\n{e}")
            print(f"❌ Error al leer o parsear data.yaml en {yaml_path}: {e}")
            return None


    def __init__(self, model_path, confidence_threshold=0.25, iou_threshold=0.45):
        """
        Inicializa el inspector de maní
        
        Args:
            model_path: Path al modelo YOLO entrenado
            confidence_threshold: Umbral de confianza para detecciones
            iou_threshold: Umbral IoU para NMS
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Verificar disponibilidad de GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Usando dispositivo: {self.device}")
        
        # 1. Cargar Nombres de Clases Dinámicamente
        loaded_names = self._load_class_names_from_yaml()
        
        if loaded_names:
            self.class_names = loaded_names
        else:
            # Fallback a las clases por defecto si el archivo no se pudo leer
            self.class_names = ['piedra', 'palito', 'metal', 'plastico']
            print(f"⚠️ Usando clases por defecto: {self.class_names}")

        # Definir colores (puede requerir ajuste manual o un mapeo dinámico si hay muchas clases)
        default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        self.class_colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(self.class_names)
        }

        # Cargar modelo para SAHI
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            device=self.device
        )
        
        # También cargar modelo YOLO directamente para comparación
        self.yolo_model = YOLO(str(model_path))
        
                
        # Estadísticas
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'detections_per_class': {name: 0 for name in self.class_names},
            'processing_times': []
        }
    
    # MÉTODOS detect_with_sahi, detect_standard, visualize_detections,
    # process_batch, compare_methods, y print_statistics SE MANTIENEN SIN CAMBIOS.
    # Los incluyo con la etiqueta 'pass' para ahorrar espacio, pero debes mantener el código original.
    
    def detect_with_sahi(self, image_path, slice_size=640, overlap_ratio=0.2, visualize=False, postprocess='GREEDYNMM'):
        """ Realiza detección usando SAHI """
        # [Mantener el código original de detect_with_sahi aquí]
        print(f"\n🔍 Procesando: {image_path}")
        start_time = time.time()
        
        # Obtener predicción con slicing
        result = get_sliced_prediction(
            str(image_path),
            self.detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type=postprocess,
            postprocess_match_threshold=self.iou_threshold,
            verbose=1 if visualize else 0
        )
        
        # Procesar resultados
        detections = []
        for prediction in result.object_prediction_list:
            detection = {
                'class': self.class_names[int(prediction.category.id)],
                'confidence': prediction.score.value,
                'bbox': prediction.bbox.to_voc_bbox(),  # [x_min, y_min, x_max, y_max]
                'area': prediction.bbox.area
            }
            detections.append(detection)
            
            # Actualizar estadísticas
            self.stats['detections_per_class'][detection['class']] += 1
        
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        self.stats['total_detections'] += len(detections)
        self.stats['total_images'] += 1
        
        print(f"✅ Detectados {len(detections)} objetos en {processing_time:.2f}s")
        
        # Visualizar si se solicita
        if visualize:
            self.visualize_detections(image_path, detections, result)
        
        return detections, result

    def detect_standard(self, image_path, visualize=False):
        """ Realiza detección estándar con YOLO """
        # [Mantener el código original de detect_standard aquí]
        print(f"\n🔍 Detección estándar: {image_path}")
        
        # Inferencia directa
        results = self.yolo_model(str(image_path), 
                                  conf=self.confidence_threshold,
                                  iou=self.iou_threshold)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detection = {
                    'class': self.class_names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        if visualize:
            # Uso de cv2.imshow es problemático con tkinter, mejor usar Matplotlib o el plot de Ultralytics
            annotated = results[0].plot()
            cv2.imshow('Detección Estándar', cv2.resize(annotated, (1280, 720)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections

    def visualize_detections(self, image_path, detections, sahi_result=None):
        """ Visualiza las detecciones en la imagen """
        # [Mantener el código original de visualize_detections aquí]
        # Leer imagen
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crear figura
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Imagen original con detecciones
        axes[0].imshow(image_rgb)
        axes[0].set_title(f'Detecciones: {len(detections)} objetos')
        axes[0].axis('off')
        
        # Dibujar bounding boxes
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            width = x_max - x_min
            height = y_max - y_min
            
            # Color según la clase
            color = np.array(self.class_colors.get(det['class'], (255, 255, 255))) / 255
            
            # Dibujar rectángulo
            rect = Rectangle((x_min, y_min), width, height,
                            linewidth=2, edgecolor=color, facecolor='none')
            axes[0].add_patch(rect)
            
            # Añadir etiqueta
            label = f"{det['class']} {det['confidence']:.2f}"
            axes[0].text(x_min, y_min - 5, label,
                            color='white', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # Visualización con heat map de SAHI
        if sahi_result is not None:
            axes[1].imshow(image_rgb)
            axes[1].set_title('Mapa de calor de detecciones')
            axes[1].axis('off')
            
            # Crear heatmap
            heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            for det in detections:
                x_min, y_min, x_max, y_max = [int(x) for x in det['bbox']]
                heatmap[y_min:y_max, x_min:x_max] += det['confidence']
            
            # Normalizar y aplicar colormap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
                axes[1].imshow(heatmap, cmap='hot', alpha=0.5)
        else:
            # Mostrar imagen con slices
            axes[1].imshow(image_rgb)
            axes[1].set_title('Grid de procesamiento (640x640)')
            axes[1].axis('off')
            
            # Dibujar grid
            h, w = image.shape[:2]
            slice_size = 640
            overlap = 0.2
            stride = int(slice_size * (1 - overlap))
            
            for y in range(0, h - slice_size + 1, stride):
                axes[1].axhline(y=y, color='cyan', linewidth=0.5, alpha=0.3)
            for x in range(0, w - slice_size + 1, stride):
                axes[1].axvline(x=x, color='cyan', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # También guardar la imagen anotada
        output_path = Path('output') / f"detected_{Path(image_path).name}"
        output_path.parent.mkdir(exist_ok=True)
        
        # Dibujar en imagen OpenCV
        for det in detections:
            x_min, y_min, x_max, y_max = [int(x) for x in det['bbox']]
            color = self.class_colors.get(det['class'], (255, 255, 255))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(str(output_path), image)
        print(f"💾 Imagen guardada en: {output_path}")

    def process_batch(self, image_folder, output_json=True, visualize=False, slice_size=640, overlap=0.2):
        """ Procesa un lote de imágenes """
        # [Mantener el código original de process_batch aquí, pero actualizar la llamada a detect_with_sahi]
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))
        
        print(f"\n📁 Procesando {len(image_files)} imágenes...")
        
        all_results = {}
        
        for img_path in tqdm(image_files, desc="Procesando"):
            detections, _ = self.detect_with_sahi(
                img_path, 
                slice_size=slice_size, # 👈 Añadido parámetro
                overlap_ratio=overlap, # 👈 Añadido parámetro
                visualize=visualize
            )
            
            all_results[str(img_path)] = {
                'filename': img_path.name,
                'detections': detections,
                'total_objects': len(detections),
                'timestamp': datetime.now().isoformat()
            }
        
        # Guardar resultados
        if output_json:
            output_path = Path('output') / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"📊 Resultados guardados en: {output_path}")
        
        # Mostrar estadísticas
        self.print_statistics()
        
        return all_results
    
    def compare_methods(self, image_path, slice_size=640, overlap=0.2):
        """ Compara detección estándar vs SAHI """
        # [Mantener el código original de compare_methods aquí, pero actualizar la llamada a detect_with_sahi]
        print("\n" + "="*50)
        print("📊 COMPARACIÓN DE MÉTODOS")
        print("="*50)
        
        # Detección estándar
        print("\n1️⃣ Detección Estándar (imagen completa redimensionada):")
        start = time.time()
        detections_standard = self.detect_standard(image_path)
        time_standard = time.time() - start
        print(f"   - Objetos detectados: {len(detections_standard)}")
        print(f"   - Tiempo: {time_standard:.2f}s")
        
        # Detección con SAHI
        print("\n2️⃣ Detección con SAHI (slicing):")
        start = time.time()
        detections_sahi, _ = self.detect_with_sahi(
            image_path,
            slice_size=slice_size,    # 👈 Añadido parámetro
            overlap_ratio=overlap     # 👈 Añadido parámetro
        )
        time_sahi = time.time() - start
        print(f"   - Objetos detectados: {len(detections_sahi)}")
        print(f"   - Tiempo: {time_sahi:.2f}s")
        
        # Comparación
        print("\n📈 Resultados:")
        improvement = (len(detections_sahi) - len(detections_standard)) / max(len(detections_standard), 1) * 100
        print(f"   - Mejora en detecciones: {improvement:.1f}%")
        print(f"   - Factor de tiempo: {time_sahi/time_standard:.1f}x")
        
        return detections_standard, detections_sahi

    def print_statistics(self):
        """ Imprime estadísticas del procesamiento """
        # [Mantener el código original de print_statistics aquí]
        if self.stats['total_images'] == 0:
            return
        
        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS FINALES")
        print("="*50)
        print(f"📷 Imágenes procesadas: {self.stats['total_images']}")
        print(f"🎯 Total de detecciones: {self.stats['total_detecciones']}")
        print(f"⏱️ Tiempo promedio: {np.mean(self.stats['processing_times']):.2f}s")
        
        print("\n📈 Distribución por clase:")
        for class_name, count in self.stats['detections_per_class'].items():
            percentage = (count / max(self.stats['total_detections'], 1)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        avg_per_image = self.stats['total_detecciones'] / self.stats['total_images']
        print(f"\n📊 Promedio por imagen: {avg_per_image:.1f} objetos")
        print("="*50)


# -----------------------------------------------------------
# FUNCIONES AUXILIARES PARA LA GUI
# -----------------------------------------------------------

def choose_file(title, filetypes):
    """Abre un diálogo para seleccionar un archivo."""
    root = tk.Tk()
    root.withdraw() # Oculta la ventana principal de Tkinter
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return filepath

def choose_directory(title):
    """Abre un diálogo para seleccionar un directorio."""
    root = tk.Tk()
    root.withdraw()
    dirpath = filedialog.askdirectory(title=title)
    return dirpath

def get_input(title, prompt, default_value, value_type=float):
    """Pide un valor numérico al usuario con un valor por defecto."""
    root = tk.Tk()
    root.withdraw()
    while True:
        user_input = simpledialog.askstring(title, prompt, initialvalue=str(default_value))
        if user_input is None:
            return None # El usuario canceló
        try:
            return value_type(user_input)
        except ValueError:
            messagebox.showerror("Error de Entrada", f"Por favor, ingrese un valor numérico válido para {title}.")

def ask_yes_no(title, prompt):
    """Pide una confirmación de Sí/No."""
    root = tk.Tk()
    root.withdraw()
    return messagebox.askyesno(title, prompt)

# -----------------------------------------------------------
# FUNCIÓN PRINCIPAL INTERACTIVA
# -----------------------------------------------------------

def run_interactive_inference():
    """
    Función principal que ejecuta el proceso de inferencia de forma interactiva
    """
    print("🚀 Iniciando Interfaz Interactiva de Inferencia SAHI...")
    
    # 1. Parámetros del Modelo y Generales
    # ------------------------------------------------
    
    # Modelo
    model_path = choose_file("Seleccionar Archivo del Modelo (ej. best.pt)", [("Model Files", "*.pt")])
    if not model_path:
        print("❌ Operación cancelada. Modelo no seleccionado.")
        return
        
    # Parámetros básicos
    conf = get_input("Umbral de Confianza", "Ingresa el umbral de confianza (ej: 0.25):", 0.25, float)
    if conf is None: return
    iou = get_input("Umbral IoU", "Ingresa el umbral IoU para NMS (ej: 0.45):", 0.45, float)
    if iou is None: return
    
    # SAHI Parámetros
    slice_size = get_input("Tamaño de Slice", "Tamaño del slice para SAHI (ej: 640):", 640, int)
    if slice_size is None: return
    overlap = get_input("Overlap", "Ratio de overlap entre slices (ej: 0.2):", 0.2, float)
    if overlap is None: return

    # Inicializar Inspector
    try:
        inspector = PeanutInspector(
            model_path=model_path,
            confidence_threshold=conf,
            iou_threshold=iou
        )
    except Exception as e:
        messagebox.showerror("Error de Inicialización", f"No se pudo cargar el modelo: {e}")
        print(f"❌ Error al cargar el modelo: {e}")
        return

    # 2. Modo de Operación
    # ------------------------------------------------
    
    root = tk.Tk()
    root.withdraw()
    mode = simpledialog.askstring("Modo de Operación", "Ingresa el modo: [I]magen individual, [C]arpeta o Co[m]parar métodos (I/C/M)", initialvalue="I").upper()
    
    visualize = ask_yes_no("Visualización", "¿Desea visualizar los resultados con Matplotlib (se abrirá una ventana)?")

    if mode == 'I':
        # IMAGEN INDIVIDUAL
        image_path = choose_file("Seleccionar Imagen Individual", [("Image Files", "*.jpg;*.png")])
        if not image_path: return
        
        print("\n=== MODO: IMAGEN INDIVIDUAL ===")
        inspector.detect_with_sahi(
            image_path,
            slice_size=slice_size,
            overlap_ratio=overlap,
            visualize=visualize
        )
    
    elif mode == 'C':
        # PROCESAR CARPETA
        folder_path = choose_directory("Seleccionar Carpeta con Imágenes")
        if not folder_path: return
        
        print("\n=== MODO: PROCESAR CARPETA ===")
        inspector.process_batch(
            folder_path,
            output_json=True,
            visualize=visualize,
            slice_size=slice_size,
            overlap=overlap
        )
        
    elif mode == 'M':
        # COMPARAR MÉTODOS
        image_path = choose_file("Seleccionar Imagen para Comparación", [("Image Files", "*.jpg;*.png")])
        if not image_path: return
        
        print("\n=== MODO: COMPARAR MÉTODOS (Estándar vs SAHI) ===")
        inspector.compare_methods(
            image_path,
            slice_size=slice_size,
            overlap=overlap
        )
        
    else:
        print("❌ Modo no reconocido. Operación terminada.")

# Reemplaza el `main()` original
if __name__ == "__main__":
    # La aplicación principal de Tkinter debe cerrarse manualmente después de que 
    # todos los diálogos se hayan completado para evitar errores.
    try:
        run_interactive_inference()
        print("\nProceso de inferencia interactiva finalizado.")
    except Exception as e:
        print(f"Un error inesperado ocurrió: {e}")
        sys.exit(1)