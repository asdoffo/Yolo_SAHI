import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
import argparse
import random # 👈 Importación añadida para la división

class DatasetTiler:
    def __init__(self, source_dir, output_dir, tile_size=640, overlap=0.2, splits=(0.7, 0.15, 0.15)):
        """
        Inicializa el preparador de dataset con tiling
        
        Args:
            source_dir: Directorio con estructura dataset/images y dataset/labels
            output_dir: Directorio de salida para el dataset procesado
            tile_size: Tamaño de cada tile (640x640)
            overlap: Porcentaje de solapamiento entre tiles
            splits: Tupla con la proporción (train, test, val)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.splits = splits # 👈 Almacenar las proporciones
        self.split_names = ['train', 'test', 'val'] # 👈 Lista de nombres de splits
        
        # Crear estructura de directorios
        self.setup_directories()
        
        # Estadísticas
        self.stats = {
            'total_images': 0,
            'total_tiles': 0,
            'total_annotations': 0,
            'tiles_with_objects': 0,
            'empty_tiles': 0,
            'objects_per_class': {}
        }
        
        # Mapear imágenes originales a su split (ej. 'img1.jpg' -> 'train')
        self.image_to_split = self.split_source_images()

    def setup_directories(self):
        """Crea la estructura de directorios para el dataset procesado, incluyendo 'test'"""
        # ⚠️ Modificación: Incluir 'test' en la creación de directorios
        for split in self.split_names:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def split_source_images(self):
        """
        Identifica las imágenes originales y las asigna a 'train', 'test' o 'val' 
        según la proporción 70/15/15.
        
        Asume que las imágenes originales están en source_dir/images (sin subcarpetas de split).
        Si ya tiene subcarpetas (ej. source/images/train), solo toma las de 'train'.
        """
        print("🔍 Buscando imágenes de origen para la división...")
        
        # Buscar imágenes en una posible subcarpeta 'train' si la estructura original ya existía
        source_images_path = self.source_dir / 'images' / 'train'
        if not source_images_path.exists():
             # Si no hay 'train', buscar directamente en source_dir/images
             source_images_path = self.source_dir / 'images' 
             
        if not source_images_path.exists():
            print(f"⚠️ Directorio de imágenes no encontrado: {source_images_path}")
            return {}

        image_files = list(source_images_path.glob('*.jpg')) + list(source_images_path.glob('*.png'))
        
        if not image_files:
            print(f"⚠️ No se encontraron imágenes en {source_images_path}")
            return {}
            
        random.shuffle(image_files) # Mezclar para asegurar una división aleatoria
        
        total_images = len(image_files)
        
        # Calcular los límites de los splits
        train_end = int(total_images * self.splits[0])
        test_end = train_end + int(total_images * self.splits[1])
        
        # Asignar a splits
        split_map = {}
        for i, img_path in enumerate(image_files):
            if i < train_end:
                split_map[img_path.name] = 'train'
            elif i < test_end:
                split_map[img_path.name] = 'test'
            else:
                split_map[img_path.name] = 'val'
                
        print(f"✅ División de imágenes de origen ({total_images} total):")
        print(f"   - Train: {train_end} ({self.splits[0]*100:.0f}%)")
        print(f"   - Test: {test_end - train_end} ({self.splits[1]*100:.0f}%)")
        print(f"   - Val: {total_images - test_end} ({self.splits[2]*100:.0f}%)")
        
        return split_map

    # El resto de las funciones (create_tiles_from_image, read_yolo_annotations, adjust_annotations_for_tile) 
    # se mantienen sin cambios ya que no dependen de los nombres de los splits.
    # ...
    
    def create_tiles_from_image(self, image_path, include_edges=True):
        """
        Divide una imagen en tiles con overlap
        ... (sin cambios) ...
        """
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error al leer imagen: {image_path}")
            return []
        
        h, w = img.shape[:2]
        tiles = []
        
        # Tiles principales con stride
        for y in range(0, h - self.tile_size + 1, self.stride):
            for x in range(0, w - self.tile_size + 1, self.stride):
                tiles.append({
                    'image': img[y:y+self.tile_size, x:x+self.tile_size],
                    'x_offset': x,
                    'y_offset': y,
                    'coords': (x, y, x+self.tile_size, y+self.tile_size)
                })
        
        if include_edges:
            # Tiles del borde derecho
            if w % self.stride != 0 and w > self.tile_size:
                for y in range(0, h - self.tile_size + 1, self.stride):
                    tiles.append({
                        'image': img[y:y+self.tile_size, w-self.tile_size:w],
                        'x_offset': w - self.tile_size,
                        'y_offset': y,
                        'coords': (w-self.tile_size, y, w, y+self.tile_size)
                    })
            
            # Tiles del borde inferior
            if h % self.stride != 0 and h > self.tile_size:
                for x in range(0, w - self.tile_size + 1, self.stride):
                    tiles.append({
                        'image': img[h-self.tile_size:h, x:x+self.tile_size],
                        'x_offset': x,
                        'y_offset': h - self.tile_size,
                        'coords': (x, h-self.tile_size, x+self.tile_size, h)
                    })
            
            # Esquina inferior derecha
            if w % self.stride != 0 and h % self.stride != 0 and w > self.tile_size and h > self.tile_size:
                tiles.append({
                    'image': img[h-self.tile_size:h, w-self.tile_size:w],
                    'x_offset': w - self.tile_size,
                    'y_offset': h - self.tile_size,
                    'coords': (w-self.tile_size, h-self.tile_size, w, h)
                })
        
        return tiles

    def read_yolo_annotations(self, label_path):
        """Lee anotaciones en formato YOLO"""
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        annotations.append([float(x) for x in parts])
        return annotations

    def adjust_annotations_for_tile(self, annotations, tile_info, img_width, img_height):
        """
        Ajusta las anotaciones YOLO para un tile específico
        ... (sin cambios) ...
        """
        adjusted = []
        x_offset = tile_info['x_offset']
        y_offset = tile_info['y_offset']
        
        for ann in annotations:
            cls, x_center, y_center, width, height = ann
            
            # Convertir a coordenadas absolutas
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height
            
            # Calcular bordes del bbox
            x_min = x_center_abs - width_abs / 2
            x_max = x_center_abs + width_abs / 2
            y_min = y_center_abs - height_abs / 2
            y_max = y_center_abs + height_abs / 2
            
            # Verificar si el bbox intersecta con el tile
            tile_x_min = x_offset
            tile_x_max = x_offset + self.tile_size
            tile_y_min = y_offset
            tile_y_max = y_offset + self.tile_size
            
            # Calcular intersección
            inter_x_min = max(x_min, tile_x_min)
            inter_x_max = min(x_max, tile_x_max)
            inter_y_min = max(y_min, tile_y_min)
            inter_y_max = min(y_max, tile_y_max)
            
            # Si hay intersección significativa (al menos 20% del objeto original)
            if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                original_area = width_abs * height_abs
                
                if inter_area / original_area >= 0.2:  # Umbral mínimo de 20%
                    # Recortar el bbox a los límites del tile
                    clipped_x_min = max(x_min, tile_x_min) - x_offset
                    clipped_x_max = min(x_max, tile_x_max) - x_offset
                    clipped_y_min = max(y_min, tile_y_min) - y_offset
                    clipped_y_max = min(y_max, tile_y_max) - y_offset
                    
                    # Calcular nuevo centro y dimensiones
                    new_x_center = (clipped_x_min + clipped_x_max) / 2
                    new_y_center = (clipped_y_min + clipped_y_max) / 2
                    new_width = clipped_x_max - clipped_x_min
                    new_height = clipped_y_max - clipped_y_min
                    
                    # Normalizar para el tile
                    new_x_norm = new_x_center / self.tile_size
                    new_y_norm = new_y_center / self.tile_size
                    new_w_norm = new_width / self.tile_size
                    new_h_norm = new_height / self.tile_size
                    
                    # Asegurar que los valores estén en [0, 1]
                    new_x_norm = max(0.001, min(0.999, new_x_norm))
                    new_y_norm = max(0.001, min(0.999, new_y_norm))
                    new_w_norm = max(0.001, min(0.999, new_w_norm))
                    new_h_norm = max(0.001, min(0.999, new_h_norm))
                    
                    adjusted.append([int(cls), new_x_norm, new_y_norm, new_w_norm, new_h_norm])
                    
                    # Actualizar estadísticas
                    cls_id = int(cls)
                    if cls_id not in self.stats['objects_per_class']:
                        self.stats['objects_per_class'][cls_id] = 0
                    self.stats['objects_per_class'][cls_id] += 1
        
        return adjusted

    def process_dataset(self, keep_empty_tiles=False, min_object_size=0.01):
        """
        Procesa todo el dataset dividiendo imágenes en tiles basado en la división train/test/val.
        """
        print("🔄 Procesando dataset...")
        
        # ⚠️ Modificación: Usar el mapa de división generado en __init__
        source_image_dir = self.source_dir / 'images' / 'train' # Asumir el directorio de origen más probable
        if not source_image_dir.exists():
            source_image_dir = self.source_dir / 'images'
            
        source_label_dir = self.source_dir / 'labels' / 'train' # Asumir el directorio de origen más probable
        if not source_label_dir.exists():
            source_label_dir = self.source_dir / 'labels'
        
        # Iterar sobre las imágenes que fueron divididas
        images_to_process = list(self.image_to_split.keys())
        
        print(f"\n📁 Procesando: {len(images_to_process)} imágenes originales")
        
        for img_name in tqdm(images_to_process, desc="Procesando imágenes de origen"):
            self.stats['total_images'] += 1
            
            # Obtener el split de destino
            split = self.image_to_split.get(img_name)
            if not split:
                print(f"⚠️ Imagen {img_name} no asignada a split. Omitiendo.")
                continue

            img_path = source_image_dir / img_name
            label_path = source_label_dir / f"{Path(img_name).stem}.txt"
            
            # Obtener path de anotaciones
            annotations = self.read_yolo_annotations(label_path)
            
            # Leer imagen para obtener dimensiones
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Crear tiles
            tiles = self.create_tiles_from_image(img_path)
            
            for tile_idx, tile_info in enumerate(tiles):
                self.stats['total_tiles'] += 1
                
                # Ajustar anotaciones para este tile
                tile_annotations = self.adjust_annotations_for_tile(
                    annotations, tile_info, img_w, img_h
                )
                
                # Filtrar objetos muy pequeños
                tile_annotations = [
                    ann for ann in tile_annotations 
                    if ann[3] * ann[4] >= min_object_size
                ]
                
                # Decidir si guardar el tile
                if len(tile_annotations) > 0 or keep_empty_tiles:
                    # Generar nombre único para el tile
                    tile_name = f"{Path(img_name).stem}_tile_{tile_idx:03d}"
                    
                    # Guardar imagen del tile en su split de destino
                    tile_img_path = self.output_dir / 'images' / split / f"{tile_name}.jpg"
                    cv2.imwrite(str(tile_img_path), tile_info['image'])
                    
                    # Guardar anotaciones
                    tile_label_path = self.output_dir / 'labels' / split / f"{tile_name}.txt"
                    with open(tile_label_path, 'w') as f:
                        for ann in tile_annotations:
                            f.write(' '.join(map(str, ann)) + '\n')
                    
                    # Actualizar estadísticas
                    if len(tile_annotations) > 0:
                        self.stats['tiles_with_objects'] += 1
                        self.stats['total_annotations'] += len(tile_annotations)
                    else:
                        self.stats['empty_tiles'] += 1

    def create_data_yaml(self, class_names=None, source_yaml=None):
        """
        Crea el archivo data.yaml para YOLO.
        """
        # ... (Lógica para determinar clases, sin cambios significativos) ...
        # Intentar leer el YAML original si existe
        if source_yaml and Path(source_yaml).exists():
            print(f"📖 Leyendo configuración desde: {source_yaml}")
            with open(source_yaml, 'r') as f:
                original_config = yaml.safe_load(f)
                class_names = original_config.get('names', [])
                nc = original_config.get('nc', len(class_names))
                print(f"   Clases encontradas: {class_names}")
        
        # Si no se especifican clases, intentar detectarlas automáticamente
        elif class_names is None:
            # Buscar YAML en el directorio fuente
            possible_yamls = list(self.source_dir.glob('*.yaml')) + \
                           list(self.source_dir.glob('*.yml')) + \
                           list(self.source_dir.glob('**/data.yaml'))
            
            if possible_yamls:
                print(f"📖 YAML encontrado: {possible_yamls[0]}")
                with open(possible_yamls[0], 'r') as f:
                    original_config = yaml.safe_load(f)
                    class_names = original_config.get('names', [])
                    nc = original_config.get('nc', len(class_names))
                    print(f"   Clases detectadas: {class_names}")
            else:
                # Detectar clases desde las anotaciones (buscar en 'train', 'test', 'val' si ya se procesó)
                print("🔍 Detectando clases desde las anotaciones...")
                detected_classes = set()
                # ⚠️ Modificación: Incluir 'test' en la búsqueda de clases
                for split in self.split_names:
                    labels_dir = self.source_dir / 'labels' / split
                    if labels_dir.exists():
                        for label_file in labels_dir.glob('*.txt'):
                            with open(label_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        detected_classes.add(int(parts[0]))
                
                if detected_classes:
                    num_classes = max(detected_classes) + 1
                    class_names = [f'class_{i}' for i in range(num_classes)]
                    print(f"⚠️ Clases detectadas automáticamente: {num_classes} clases")
                    print("   Usando nombres genéricos. Considera especificar los nombres reales.")
                else:
                    print("⚠️ No se pudieron detectar clases. Usando valores por defecto.")
                    class_names = ['object']
        
        # Crear configuración
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            # ⚠️ Modificación: Añadir 'test'
            'test': 'images/test', 
            'val': 'images/val',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Archivo data.yaml creado en: {yaml_path}")
        print(f"   - Número de clases: {len(class_names)}")
        print(f"   - Clases: {class_names}")
    
    def print_statistics(self):
        """Imprime estadísticas del procesamiento"""
        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS DEL DATASET")
        print("="*50)
        print(f"📷 Imágenes originales procesadas: {self.stats['total_images']}")
        print(f"🔲 Total de tiles generados: {self.stats['total_tiles']}")
        print(f"✅ Tiles con objetos: {self.stats['tiles_with_objects']}")
        print(f"⬜ Tiles vacíos: {self.stats['empty_tiles']}")
        print(f"📦 Total de anotaciones: {self.stats['total_annotations']}")
        
        if self.stats['tiles_with_objects'] > 0:
            avg_objects = self.stats['total_annotations'] / self.stats['tiles_with_objects']
            print(f"📊 Promedio de objetos por tile (con objetos): {avg_objects:.2f}")
        
        if self.stats['objects_per_class']:
            print("\n📈 Distribución por clase:")
            for cls_id, count in sorted(self.stats['objects_per_class'].items()):
                print(f"   Clase {cls_id}: {count} objetos")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Preparar dataset con tiling para YOLO')
    parser.add_argument('--source', type=str, required=True, 
                       help='Directorio fuente con estructura YOLO (ej: dataset_original/)')
    parser.add_argument('--output', type=str, required=True,
                       help='Directorio de salida para dataset procesado (ej: tiled_dataset/)')
    parser.add_argument('--tile-size', type=int, default=640,
                       help='Tamaño de los tiles (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.2,
                       help='Overlap entre tiles (default: 0.2)')
    parser.add_argument('--keep-empty', action='store_true',
                       help='Mantener tiles sin objetos')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                       help='Nombres de las clases (opcional, se detecta automáticamente)')
    parser.add_argument('--source-yaml', type=str, default=None,
                       help='Path al archivo YAML original (opcional)')
    
    args = parser.parse_args()
    
    # Proporción: 70% train, 15% test, 15% val
    splits = (0.70, 0.15, 0.15) 
    
    # Procesar dataset
    # ⚠️ Modificación: Pasar las proporciones al inicializador
    tiler = DatasetTiler(
        source_dir=args.source,
        output_dir=args.output,
        tile_size=args.tile-size,
        overlap=args.overlap,
        splits=splits
    )
    
    # Procesar imágenes y anotaciones
    tiler.process_dataset(keep_empty_tiles=args.keep_empty)
    
    # Crear archivo de configuración
    tiler.create_data_yaml(
        class_names=args.classes,
        source_yaml=args.source_yaml
    )
    
    # Mostrar estadísticas
    tiler.print_statistics()


if __name__ == "__main__":
    main()