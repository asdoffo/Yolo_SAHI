import cv2
import os
import numpy as np
import random
import albumentations as A
import yaml 
from datetime import datetime
from tkinter import Tk, filedialog, simpledialog

# =========================================================================
# === 0. CLASES Y CONFIGURACIÓN ESTATICA ===
# =========================================================================

CLASS_NAMES = {
    0: 'Metal', 1: 'Cascabullo', 2: 'Cebollin', 3: 'Hueso', 4: 'Maiz', 
    5: 'Palo', 6: 'Piedras', 7: 'Plastico', 8: 'Soja', 9: 'Vidrio', 10: 'Cascote'
}

ORIGINAL_IMG_WIDTH = 3840
ORIGINAL_IMG_HEIGHT = 2160

CROP_SIZE = 640
RANDOM_SHIFT = 200 
CROPS_PER_ANNOTATION = 2 

# --- PIPELINE DE AUMENTACIÓN CON SOPORTE PARA BOUNDING BOXES ---
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5), 
    A.VerticalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3), 
    A.ColorJitter(
        brightness=(0.8, 1.2), contrast=(0.8, 1.2), 
        saturation=(0.8, 1.2), hue=(-0.05, 0.05), p=0.5
    ),
    A.ToGray(p=0.1),
], bbox_params=A.BboxParams(
    format='yolo',  # Usamos formato YOLO (x_center, y_center, width, height) normalizado
    label_fields=['class_labels'],  # Campo que contiene las clases
    min_area=0,
    min_visibility=0.3  # Mantener solo bboxes con al menos 30% visible después de la transformación
))


# =========================================================================
# === FUNCIONES DE UTILIDAD ===
# =========================================================================

def select_directory(title):
    """Abre un diálogo para seleccionar una carpeta."""
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path

def yolo_to_pixels(img_width, img_height, x_center_norm, y_center_norm, width_norm, height_norm):
    """Convierte las coordenadas normalizadas de YOLO a coordenadas de píxeles."""
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    box_width = width_norm * img_width
    box_height = height_norm * img_height
    return x_center, y_center, box_width, box_height

def pixels_to_yolo(crop_size, center_x, center_y, box_width, box_height):
    """Convierte coordenadas de píxeles a formato YOLO normalizado."""
    x_center_norm = center_x / crop_size
    y_center_norm = center_y / crop_size
    width_norm = box_width / crop_size
    height_norm = box_height / crop_size
    return x_center_norm, y_center_norm, width_norm, height_norm

def get_safe_crop_region(img, target_center_x, target_center_y, CROP_SIZE):
    """
    Ajusta el centro del recorte para evitar salirse de la imagen.
    Si el recorte se sale de los límites, lo desplaza para que quede completamente dentro.
    """
    img_height, img_width, _ = img.shape
    HALF_CROP = CROP_SIZE // 2
    
    # Ajustar el centro del recorte si se sale de los límites
    adjusted_center_x = target_center_x
    adjusted_center_y = target_center_y
    
    # Ajustar en X
    if target_center_x - HALF_CROP < 0:
        adjusted_center_x = HALF_CROP
    elif target_center_x + HALF_CROP > img_width:
        adjusted_center_x = img_width - HALF_CROP
    
    # Ajustar en Y
    if target_center_y - HALF_CROP < 0:
        adjusted_center_y = HALF_CROP
    elif target_center_y + HALF_CROP > img_height:
        adjusted_center_y = img_height - HALF_CROP
    
    # Calcular las coordenadas del recorte
    crop_x_start = adjusted_center_x - HALF_CROP
    crop_x_end = adjusted_center_x + HALF_CROP
    crop_y_start = adjusted_center_y - HALF_CROP
    crop_y_end = adjusted_center_y + HALF_CROP
    
    # Asegurar que las coordenadas estén dentro de los límites
    crop_x_start = max(0, crop_x_start)
    crop_x_end = min(img_width, crop_x_end)
    crop_y_start = max(0, crop_y_start)
    crop_y_end = min(img_height, crop_y_end)
    
    # Realizar el recorte
    cropped_img = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    # Si por alguna razón el recorte no es exactamente CROP_SIZE x CROP_SIZE, 
    # rellenar con blanco
    if cropped_img.shape[0] != CROP_SIZE or cropped_img.shape[1] != CROP_SIZE:
        final_crop = np.ones((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8) * 255
        h, w = cropped_img.shape[:2]
        final_crop[:h, :w] = cropped_img
    else:
        final_crop = cropped_img
    
    return final_crop, crop_x_start, crop_y_start, crop_x_end, crop_y_end

def is_bbox_in_crop(bbox_center_x, bbox_center_y, bbox_width, bbox_height, 
                    crop_x_start, crop_y_start, crop_x_end, crop_y_end):
    """
    Verifica si un bounding box está dentro del área del recorte.
    Retorna True si el centro del bbox está dentro del recorte.
    """
    if (crop_x_start <= bbox_center_x <= crop_x_end and 
        crop_y_start <= bbox_center_y <= crop_y_end):
        return True
    return False

def clip_bbox_to_crop(bbox_center_x, bbox_center_y, bbox_width, bbox_height,
                      crop_x_start, crop_y_start, crop_x_end, crop_y_end):
    """
    Ajusta un bounding box para que se mantenga dentro de los límites del recorte.
    Retorna las nuevas coordenadas relativas al recorte.
    """
    # Convertir centro a esquinas
    x1 = bbox_center_x - bbox_width / 2
    y1 = bbox_center_y - bbox_height / 2
    x2 = bbox_center_x + bbox_width / 2
    y2 = bbox_center_y + bbox_height / 2
    
    # Hacer clip a los límites del recorte
    x1_clipped = max(crop_x_start, min(crop_x_end, x1))
    y1_clipped = max(crop_y_start, min(crop_y_end, y1))
    x2_clipped = max(crop_x_start, min(crop_x_end, x2))
    y2_clipped = max(crop_y_start, min(crop_y_end, y2))
    
    # Convertir de vuelta a centro y dimensiones (relativas al recorte)
    new_width = x2_clipped - x1_clipped
    new_height = y2_clipped - y1_clipped
    new_center_x = (x1_clipped + x2_clipped) / 2 - crop_x_start
    new_center_y = (y1_clipped + y2_clipped) / 2 - crop_y_start
    
    # Validar que el bbox tenga dimensiones válidas
    if new_width <= 0 or new_height <= 0:
        return None
    
    return new_center_x, new_center_y, new_width, new_height

def load_all_annotations(annotation_path, img_width, img_height):
    """
    Carga todas las anotaciones de un archivo YOLO y las convierte a píxeles.
    """
    annotations = []
    
    if not os.path.exists(annotation_path):
        return annotations
    
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        try:
            parts = list(map(float, line.split()))
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = yolo_to_pixels(
                img_width, img_height, parts[1], parts[2], parts[3], parts[4]
            )
            
            annotations.append({
                'class_id': class_id,
                'center_x': center_x,
                'center_y': center_y,
                'box_width': box_width,
                'box_height': box_height
            })
        except Exception as e:
            print(f"Error parseando línea: {line.strip()} - {e}")
            continue
    
    return annotations

def apply_augmentation_with_bboxes(image, bboxes_yolo_format, class_labels, subset_name):
    """
    Aplica aumentación de datos a la imagen Y a los bounding boxes simultáneamente.
    
    Args:
        image: numpy array de la imagen
        bboxes_yolo_format: lista de bboxes en formato YOLO normalizado [[x_center, y_center, width, height], ...]
        class_labels: lista de class_ids correspondientes a cada bbox
        subset_name: 'train', 'val' o 'test'
    
    Returns:
        transformed_image: imagen transformada
        transformed_bboxes: bboxes transformados en formato YOLO
        transformed_labels: class_ids correspondientes
    """
    # Solo aplicar aumentación a train y val
    if subset_name not in ['train', 'val']:
        return image, bboxes_yolo_format, class_labels
    
    # Si no hay bboxes, solo transformar la imagen
    if len(bboxes_yolo_format) == 0:
        transformed = augmentation_pipeline(image=image, bboxes=[], class_labels=[])
        return transformed['image'], [], []
    
    try:
        # Aplicar transformación
        transformed = augmentation_pipeline(
            image=image,
            bboxes=bboxes_yolo_format,
            class_labels=class_labels
        )
        
        return transformed['image'], transformed['bboxes'], transformed['class_labels']
    
    except Exception as e:
        print(f"⚠️ Error en aumentación: {e}. Devolviendo imagen original.")
        return image, bboxes_yolo_format, class_labels

def generate_data_yaml(output_base_dir, data_root_name, CLASS_NAMES, mode, positive_data_path):
    """Genera el archivo data.yaml."""
    
    names_list = [v for k, v in sorted(CLASS_NAMES.items())]
    data_folder_name = os.path.basename(positive_data_path)

    if mode == 'deteccion':
        data_config = {
            'path': f'./{data_root_name}', 
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(names_list), 
            'names': names_list
        }
    else:
        data_config = {
            'info': 'Dataset de Clasificación',
            'classes': names_list,
            'splits': ['train', 'val', 'test']
        }
        
    yaml_filepath = os.path.join(output_base_dir, 'data.yaml')
    
    try:
        with open(yaml_filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
        print(f"✅ Archivo 'data.yaml' generado para '{mode}' en: {yaml_filepath}")
    except Exception as e:
        print(f"❌ Error al generar data.yaml: {e}")


# =========================================================================
# === FUNCIONES PRINCIPALES DE GENERACIÓN ===
# =========================================================================

def generate_crops(data_config, subset_name, mode):
    """
    Genera recortes positivos e incluye TODAS las etiquetas de objetos dentro del recorte.
    Aplica aumentación correctamente tanto a imágenes como a bounding boxes.
    """
    
    IMAGE_DIR = os.path.join(data_config['positive_data_path'], 'images', subset_name)
    YOLO_ANNO_DIR = os.path.join(data_config['positive_data_path'], 'labels', subset_name)
    
    if mode == 'deteccion':
        output_img_dir = os.path.join(data_config['output_root'], 'images', subset_name)
        output_lbl_dir = os.path.join(data_config['output_root'], 'labels', subset_name)
        os.makedirs(output_lbl_dir, exist_ok=True)
    else:
        output_img_dir = os.path.join(data_config['output_root'], subset_name, 'positivos')
    
    os.makedirs(output_img_dir, exist_ok=True)
    
    crop_count = 0
    
    for image_file in os.listdir(IMAGE_DIR):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(IMAGE_DIR, image_file)
        annotation_path = os.path.join(YOLO_ANNO_DIR, base_name + '.txt')

        img = cv2.imread(image_path)
        if img is None: 
            continue
        
        img_height, img_width, _ = img.shape
        
        if not os.path.exists(annotation_path): 
            continue
        
        # CARGAR TODAS LAS ANOTACIONES DE LA IMAGEN
        all_annotations = load_all_annotations(annotation_path, img_width, img_height)
        
        if not all_annotations:
            continue
        
        # Iterar sobre cada anotación para generar recortes centrados en ella
        for ann_index, main_annotation in enumerate(all_annotations):
            center_x = main_annotation['center_x']
            center_y = main_annotation['center_y']
            class_id = main_annotation['class_id']
            
            for i in range(data_config['crops_per_annotation']):
                # Generar múltiples recortes con desplazamiento aleatorio
                shift_x = random.randint(-data_config['random_shift'], data_config['random_shift'])
                shift_y = random.randint(-data_config['random_shift'], data_config['random_shift'])
                
                target_center_x = int(center_x + shift_x)
                target_center_y = int(center_y + shift_y)
                
                # Obtener recorte seguro (SIN padding con bordes replicados)
                cropped_img, crop_x_start, crop_y_start, crop_x_end, crop_y_end = get_safe_crop_region(
                    img, target_center_x, target_center_y, data_config['crop_size']
                )
                
                if mode == 'deteccion':
                    # BUSCAR TODOS LOS OBJETOS QUE CAEN DENTRO DE ESTE RECORTE
                    bboxes_in_crop = []  # Formato YOLO normalizado para albumentations
                    class_labels_in_crop = []
                    
                    for annotation in all_annotations:
                        bbox_center_x = annotation['center_x']
                        bbox_center_y = annotation['center_y']
                        bbox_width = annotation['box_width']
                        bbox_height = annotation['box_height']
                        bbox_class_id = annotation['class_id']
                        
                        # Verificar si este objeto está en el recorte
                        if is_bbox_in_crop(bbox_center_x, bbox_center_y, bbox_width, bbox_height,
                                          crop_x_start, crop_y_start, crop_x_end, crop_y_end):
                            
                            # Ajustar el bbox a los límites del recorte
                            clipped = clip_bbox_to_crop(
                                bbox_center_x, bbox_center_y, bbox_width, bbox_height,
                                crop_x_start, crop_y_start, crop_x_end, crop_y_end
                            )
                            
                            if clipped is not None:
                                new_center_x, new_center_y, new_width, new_height = clipped
                                
                                # Convertir a formato YOLO normalizado
                                x_norm, y_norm, w_norm, h_norm = pixels_to_yolo(
                                    data_config['crop_size'],
                                    new_center_x, new_center_y,
                                    new_width, new_height
                                )
                                
                                # Validar que las coordenadas estén en el rango [0, 1]
                                if (0 <= x_norm <= 1 and 0 <= y_norm <= 1 and 
                                    0 < w_norm <= 1 and 0 < h_norm <= 1):
                                    # Guardar en formato YOLO para albumentations
                                    bboxes_in_crop.append([x_norm, y_norm, w_norm, h_norm])
                                    class_labels_in_crop.append(bbox_class_id)
                    
                    # APLICAR AUMENTACIÓN con los bounding boxes
                    augmented_img, augmented_bboxes, augmented_labels = apply_augmentation_with_bboxes(
                        cropped_img, bboxes_in_crop, class_labels_in_crop, subset_name
                    )
                    
                    # Guardar la imagen
                    output_filename = f"{base_name}_ann{ann_index}_crop{i}_clase{class_id}.jpg"
                    output_path = os.path.join(output_img_dir, output_filename)
                    cv2.imwrite(output_path, augmented_img)
                    
                    # Escribir TODAS las etiquetas transformadas en el archivo
                    if len(augmented_bboxes) > 0:
                        label_filename = output_filename.replace('.jpg', '.txt')
                        label_path = os.path.join(output_lbl_dir, label_filename)
                        
                        with open(label_path, 'w') as lf:
                            for bbox, label in zip(augmented_bboxes, augmented_labels):
                                x_c, y_c, w, h = bbox
                                # Asegurar que class_id sea entero, no float
                                lf.write(f"{int(label)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                        
                        crop_count += 1
                    else:
                        # Si no hay etiquetas válidas, eliminar la imagen
                        os.remove(output_path)
                
                else:  # Clasificación
                    # Para clasificación, solo aplicar aumentación a la imagen
                    augmented_img, _, _ = apply_augmentation_with_bboxes(
                        cropped_img, [], [], subset_name
                    )
                    
                    output_filename = f"{base_name}_ann{ann_index}_crop{i}_clase{class_id}.jpg"
                    output_path = os.path.join(output_img_dir, output_filename)
                    cv2.imwrite(output_path, augmented_img)
                    crop_count += 1

    print(f"Recortes POSITIVOS generados para {subset_name}: {crop_count}")
    return crop_count

def generate_negative_crops(data_config, subset_name):
    """Genera recortes negativos (sin elementos a detectar)."""
    
    IMAGE_DIR = os.path.join(data_config['negative_data_path'], subset_name)
    OUTPUT_DIR = os.path.join(data_config['output_root'], subset_name, 'negativos')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    crop_count = 0

    print(f"\n--- Procesando Negativos: {subset_name} ---")

    for image_file in os.listdir(IMAGE_DIR):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        image_path = os.path.join(IMAGE_DIR, image_file)
        img = cv2.imread(image_path)
        if img is None: 
            continue
        
        img_h, img_w, _ = img.shape
        step = data_config['crop_size']
        
        for y in range(0, img_h - step + 1, step):
            for x in range(0, img_w - step + 1, step):
                cropped_img = img[y:y + step, x:x + step]
                
                # Aplicar aumentación (sin bboxes)
                augmented_img, _, _ = apply_augmentation_with_bboxes(
                    cropped_img, [], [], subset_name
                )
                
                output_filename = f"neg_{os.path.splitext(image_file)[0]}_x{x}_y{y}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                cv2.imwrite(output_path, augmented_img)
                crop_count += 1
                
    print(f"Recortes NEGATIVOS generados para {subset_name}: {crop_count}")
    return crop_count


def main():
    print("=========================================================")
    print("=== GENERADOR DE DATASETS (YOLO/CLASIFICACIÓN) V3.1 =====")
    print("=========================================================")
    
    output_base_dir = select_directory("1. Selecciona el DIRECTORIO BASE para guardar el nuevo Dataset")
    if not output_base_dir:
        print("Operación cancelada. No se seleccionó la carpeta de salida.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_root_name = f"DS_YOLO_{timestamp}"
    output_root = os.path.join(output_base_dir, data_root_name)
    os.makedirs(output_root, exist_ok=True)
    
    pos_data_path = select_directory("2. Selecciona la CARPETA RAIZ de tus imágenes ETIQUETADAS")
    if not pos_data_path:
        print("Operación cancelada. No se seleccionó la carpeta de datos etiquetados.")
        return

    neg_data_path = select_directory("3. Selecciona la CARPETA RAIZ de tus imágenes SIN ETIQUETAS (Si no aplica, omite)")
    if not neg_data_path:
        print("Advertencia: No se seleccionó la carpeta de datos sin etiquetas.")
    
    root = Tk()
    root.withdraw()
    mode_input = simpledialog.askstring("Modo de Operación", "Selecciona el MODO (Deteccion o Clasificacion):")
    root.destroy()
    
    mode = 'deteccion'
    if mode_input and mode_input.lower().startswith('c'):
         mode = 'clasificacion'

    print(f"\n--- Resumen de Configuración ---")
    print(f"Modo seleccionado: {mode.upper()}")
    print(f"Ruta de salida: {output_root}")
    print(f"Imágenes Positivas: {pos_data_path}")
    print(f"Imágenes Negativas: {neg_data_path if neg_data_path else 'N/A'}")
    print(f"Tamaño de recorte: {CROP_SIZE}x{CROP_SIZE}")
    print(f"----------------------------------")

    config = {
        'img_width': ORIGINAL_IMG_WIDTH,
        'img_height': ORIGINAL_IMG_HEIGHT,
        'crop_size': CROP_SIZE,
        'random_shift': RANDOM_SHIFT,
        'crops_per_annotation': CROPS_PER_ANNOTATION,
        'positive_data_path': pos_data_path,
        'negative_data_path': neg_data_path,
        'output_root': output_root
    }

    total_crops = 0
    splits = ['train', 'val', 'test']
    
    print("\n[FASE 1/2] Generando recortes POSITIVOS...")
    for subset in splits:
        total_crops += generate_crops(config, subset, mode)
    
    if mode == 'clasificacion' and neg_data_path:
        print("\n[FASE 2/2] Generando recortes NEGATIVOS...")
        for subset in splits:
            total_crops += generate_negative_crops(config, subset)
    
    generate_data_yaml(output_base_dir, data_root_name, CLASS_NAMES, mode, pos_data_path)

    print("\n--- PROCESO COMPLETADO ---")
    print(f"Total de recortes generados: {total_crops}")
    print(f"Dataset guardado en: {output_root}")

if __name__ == "__main__":
    main()