import os
import shutil
import random
import yaml # Necesitas instalar la librería PyYAML: pip install pyyaml

def split_dataset(root_dir, class_names, train_percent=0.8, val_percent=0.1, test_percent=0.1):
    """
    Divide un conjunto de datos (imágenes y etiquetas) y genera el archivo data.yaml.

    Args:
        root_dir (str): Directorio raíz que contiene 'images/train' y 'labels/train'.
        class_names (dict): Diccionario de nombres de clases {índice: nombre}.
        train_percent (float): Porcentaje para el conjunto de entrenamiento.
        val_percent (float): Porcentaje para el conjunto de validación.
        test_percent (float): Porcentaje para el conjunto de prueba.
    """
    # 1. Validación inicial de directorios
    original_images_dir = os.path.join(root_dir, 'images', 'train')
    original_labels_dir = os.path.join(root_dir, 'labels', 'train')

    if not os.path.exists(original_images_dir) or not os.path.exists(original_labels_dir):
        print(f"Error: No se encontraron los directorios de datos esperados dentro de {root_dir}.")
        return

    # 2. Validación y preparación de porcentajes
    total_percent = train_percent + val_percent + test_percent
    if not (0.99 <= total_percent <= 1.01):
        print(f"Error: La suma de porcentajes debe ser 1 (100%). Suma actual: {total_percent}")
        return

    # 3. Preparar directorios de destino (val y test)
    for split_name in ['val', 'test']:
        os.makedirs(os.path.join(root_dir, 'images', split_name), exist_ok=True)
        os.makedirs(os.path.join(root_dir, 'labels', split_name), exist_ok=True)

    # 4. Obtener la lista maestra de archivos de imágenes y aleatorizar
    all_images = [f for f in os.listdir(original_images_dir) if os.path.isfile(os.path.join(original_images_dir, f))]
    random.shuffle(all_images)
    total_files = len(all_images)

    print(f"Archivos de imagen encontrados: {total_files}")
    
    # 5. Calcular los límites de los splits
    val_count = int(total_files * val_percent)
    test_count = int(total_files * test_percent)
    train_count = total_files - val_count - test_count # El resto para entrenamiento

    print(f"División: Train={train_count}, Val={val_count}, Test={test_count}")

    # 6. Realizar la separación y mover los archivos

    val_files = all_images[:val_count]
    test_files = all_images[val_count : val_count + test_count]

    def move_file_pair(filename, destination_split):
        base_name, ext = os.path.splitext(filename)
        label_filename = base_name + '.txt' # Asumiendo formato de etiqueta .txt (ej. YOLO)

        # Rutas de origen
        src_img = os.path.join(original_images_dir, filename)
        src_label = os.path.join(original_labels_dir, label_filename)
        
        # Rutas de destino
        dst_img_folder = os.path.join(root_dir, 'images', destination_split)
        dst_label_folder = os.path.join(root_dir, 'labels', destination_split)
        
        shutil.move(src_img, os.path.join(dst_img_folder, filename))
        
        if os.path.exists(src_label):
            shutil.move(src_label, os.path.join(dst_label_folder, label_filename))
        else:
            print(f"Advertencia: No se encontró etiqueta '{label_filename}' para '{filename}'.")


    # Mover archivos de Val
    print(f"\nMoviendo {len(val_files)} archivos a 'val'...")
    for f in val_files:
        move_file_pair(f, 'val')

    # Mover archivos de Test
    print(f"Moviendo {len(test_files)} archivos a 'test'...")
    for f in test_files:
        move_file_pair(f, 'test')
    
    print("\n¡División de conjuntos completada con éxito!")
    
    # 7. Generar el archivo data.yaml
    generate_data_yaml(root_dir, class_names)


def generate_data_yaml(root_dir, class_names):
    """Crea el archivo data.yaml con la configuración correcta."""

    # Normalizar los nombres de las clases (ej. Ceboll\xEDn -> Cebollín)
    # y convertir el diccionario de clases {indice: nombre} a la lista [nombre1, nombre2, ...]
    names_list = [v for k, v in sorted(class_names.items())]

    data_config = {
        # La ruta base donde se encuentran las carpetas 'images' y 'labels'
        'path': f'../{os.path.basename(root_dir)}', 
        # Rutas relativas a la carpeta 'path'
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        
        # Número de clases
        'nc': len(names_list), 
        
        # Lista de nombres de clases
        'names': names_list
    }
    
    yaml_filepath = os.path.join(os.path.dirname(root_dir), 'data.yaml')
    
    try:
        with open(yaml_filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
        print(f"\n✅ Archivo 'data.yaml' generado con éxito en: {yaml_filepath}")
    except Exception as e:
        print(f"❌ Error al generar data.yaml: {e}")

# =========================================================================
# === CÓMO USAR EL SCRIPT ===
# =========================================================================

# 1. Define la carpeta raíz exportada de CVAT
root_directory_name = 'D:\Proyectos\IA\AGD\PMD\Yolo_SAHI\datasets\original\Images_ASD_2025_10_21' 

# 2. Define las proporciones de división (deben sumar 1.0)
TRAIN_RATIO = 0.80 
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# 3. Define las clases *exactamente* como estaban, pero en formato de diccionario Python:
# Las codificaciones como \xED se han normalizado a caracteres normales ('í', 'á')
CLASS_NAMES = {
  0: 'Metal',
  1: 'Cascabullo',
  2: 'Cebollin',
  3: 'Hueso',
  4: 'Maiz',
  5: 'Palo',
  6: 'Piedras',
  7: 'Plastico',
  8: 'Soja',
  9: 'Vidrio',
  10: 'Cascote'
}

# Ejecutar la función
split_dataset(root_directory_name, CLASS_NAMES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)