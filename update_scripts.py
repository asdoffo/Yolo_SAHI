"""
Script de actualización para corregir la detección automática de clases
Aplica los cambios necesarios a dataset_preparation.py y pipeline_example.py
"""

import yaml
from pathlib import Path

def update_dataset_preparation():
    """Actualiza dataset_preparation.py con la nueva funcionalidad"""
    
    # Código actualizado para el método create_data_yaml
    new_method = '''    def create_data_yaml(self, class_names=None, source_yaml=None):
        """
        Crea el archivo data.yaml para YOLO
        
        Args:
            class_names: Lista de nombres de clases (opcional)
            source_yaml: Path al YAML original para copiar las clases (opcional)
        """
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
            possible_yamls = list(self.source_dir.glob('*.yaml')) + \\
                           list(self.source_dir.glob('*.yml')) + \\
                           list(self.source_dir.glob('**/data.yaml'))
            
            if possible_yamls:
                print(f"📖 YAML encontrado: {possible_yamls[0]}")
                with open(possible_yamls[0], 'r') as f:
                    original_config = yaml.safe_load(f)
                    class_names = original_config.get('names', [])
                    nc = original_config.get('nc', len(class_names))
                    print(f"   Clases detectadas: {class_names}")
            else:
                # Detectar clases desde las anotaciones
                print("🔍 Detectando clases desde las anotaciones...")
                detected_classes = set()
                for split in ['train', 'val']:
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
'''
    
    print("📝 Actualizando dataset_preparation.py...")
    
    # Leer el archivo existente
    with open('dataset_preparation.py', 'r') as f:
        content = f.read()
    
    # Buscar y reemplazar el método create_data_yaml
    import re
    pattern = r'def create_data_yaml\(self.*?\n(?:.*?\n)*?.*?print\(f"✅ Archivo data\.yaml creado'
    
    if 'def create_data_yaml(self, class_names):' in content:
        # Versión antigua encontrada
        start = content.find('def create_data_yaml(self, class_names):')
        # Encontrar el final del método (siguiente def o final de clase)
        end = content.find('\n    def ', start + 1)
        if end == -1:
            end = content.find('\n\ndef ', start + 1)
        
        if start != -1 and end != -1:
            content = content[:start] + new_method.strip() + '\n\n    ' + content[end+1:]
        
        # También actualizar el main()
        content = content.replace(
            'tiler.create_data_yaml(args.classes)',
            'tiler.create_data_yaml(\n        class_names=args.classes,\n        source_yaml=args.source_yaml\n    )'
        )
        
        # Actualizar los argumentos del parser
        content = content.replace(
            "default=['piedra', 'palito', 'metal', 'plastico'],",
            "default=None,"
        )
        
        # Añadir el nuevo argumento source-yaml si no existe
        if '--source-yaml' not in content:
            parser_end = content.find('args = parser.parse_args()')
            if parser_end != -1:
                new_arg = """    parser.add_argument('--source-yaml', type=str, default=None,
                       help='Path al archivo YAML original (opcional)')
    
    """
                content = content[:parser_end] + new_arg + content[parser_end:]
        
        # Guardar el archivo actualizado
        with open('dataset_preparation.py', 'w') as f:
            f.write(content)
        
        print("✅ dataset_preparation.py actualizado correctamente")
        return True
    else:
        print("⚠️ El archivo ya parece estar actualizado o tiene un formato diferente")
        return False


def update_pipeline_example():
    """Actualiza pipeline_example.py para usar la detección automática"""
    
    print("📝 Actualizando pipeline_example.py...")
    
    with open('pipeline_example.py', 'r') as f:
        content = f.read()
    
    # Buscar y reemplazar la línea problemática
    old_line = "tiler.create_data_yaml(['piedra', 'palito', 'metal', 'plastico'])"
    new_line = "# Detectar clases automáticamente o usar las del YAML original\n    tiler.create_data_yaml(source_yaml=f'{source_dir}/data.yaml' if Path(f'{source_dir}/data.yaml').exists() else None)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Guardar el archivo actualizado
        with open('pipeline_example.py', 'w') as f:
            f.write(content)
        
        print("✅ pipeline_example.py actualizado correctamente")
        return True
    else:
        print("⚠️ El archivo ya parece estar actualizado")
        return False


def check_existing_yaml(dataset_path='datasets/original'):
    """Verifica si existe un archivo YAML en el dataset"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"⚠️ El directorio {dataset_path} no existe aún")
        return None
    
    # Buscar archivos YAML
    yaml_files = list(dataset_path.glob('*.yaml')) + list(dataset_path.glob('*.yml'))
    
    if yaml_files:
        print(f"\n📋 Archivo YAML encontrado: {yaml_files[0]}")
        
        with open(yaml_files[0], 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"   Clases detectadas: {config.get('names', [])}")
        print(f"   Número de clases: {config.get('nc', 0)}")
        
        return yaml_files[0]
    else:
        print(f"\n⚠️ No se encontró archivo YAML en {dataset_path}")
        print("   El sistema intentará detectar las clases automáticamente desde las anotaciones")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 ACTUALIZACIÓN DE SCRIPTS")
    print("=" * 60)
    print("\nEste script actualiza los archivos para detectar clases automáticamente")
    print("desde tu archivo YAML existente o desde las anotaciones.\n")
    
    # Actualizar archivos
    updated_1 = update_dataset_preparation()
    updated_2 = update_pipeline_example()
    
    if updated_1 or updated_2:
        print("\n✨ Actualización completada!")
    else:
        print("\n✅ Los archivos ya estaban actualizados")
    
    # Verificar si hay un YAML existente
    yaml_file = check_existing_yaml()
    
    print("\n" + "=" * 60)
    print("📌 USO ACTUALIZADO:")
    print("=" * 60)
    
    if yaml_file:
        print(f"\n# Opción 1: Usar tu YAML existente")
        print(f"python dataset_preparation.py --source datasets/original --output datasets/tiled --source-yaml {yaml_file}")
    
    print(f"\n# Opción 2: Detección automática (desde anotaciones o YAML)")
    print(f"python dataset_preparation.py --source datasets/original --output datasets/tiled")
    
    print(f"\n# Opción 3: Especificar clases manualmente")
    print(f"python dataset_preparation.py --source datasets/original --output datasets/tiled --classes clase1 clase2 clase3")
    
    print("\nNota: El sistema ahora:")
    print("  1. Primero busca un YAML existente en el directorio fuente")
    print("  2. Si no lo encuentra, detecta las clases desde las anotaciones")
    print("  3. Si especificas --source-yaml, usa ese archivo específico")
    print("  4. Si especificas --classes, usa esa lista de clases")
