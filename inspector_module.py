# inspector_module.py (Simplificado para Jetson/Streamlit)
import cv2
import numpy as np
from pathlib import Path
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import time
from collections import Counter
import random # Solo para la simulación de bboxes si no se dibuja en cv2

# Definición de las 11 clases del data.yaml
CLASSES = {
    0: "Metal",
    1: "Cascabullo",
    2: "Cebollin",
    3: "Hueso",
    4: "Maiz",
    5: "Palo",
    6: "Piedras",
    7: "Plastico",
    8: "Soja",
    9: "Vidrio",
    10: "Cascote"
}

# Definimos las clases que consideramos "Cuerpos Extraños" (todas en este caso, excepto quizás un Maní 'OK' si existiera)
FOREIGN_CLASSES = list(CLASSES.values())


class StreamlitInspector:
    def __init__(self, model_path, confidence_threshold=0.25, iou_threshold=0.45):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_names = list(CLASSES.values())
        
        # Usamos CPU/GPU (cuda:0) o TensorRT si el .onnx fue optimizado
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # NOTA: En la Jetson, se recomienda usar ONNX Runtime o TensorRT Engine.
        # Asumiremos que el AutoDetectionModel de SAHI maneja la carga del ONNX.
        # (Necesitarás instalar 'sahi', 'ultralytics', 'onnxruntime' en tu Docker/Jetson)
        
        print(f"⚙️ Inicializando modelo SAHI en dispositivo: {self.device}")
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=str(model_path), # Esto debería ser best.onnx
            confidence_threshold=confidence_threshold,
            device=self.device
        )

    def detect_frame(self, frame_np, slice_size=640, overlap_ratio=0.2, postprocess='GREEDYNMM'):
        """ Realiza detección en un frame de numpy usando SAHI y dibuja los resultados. """
        if self.detection_model is None:
            # Manejo de error si el modelo no se cargó
            frame_annotated = frame_np.copy()
            cv2.putText(frame_annotated, "ERROR: MODELO NO CARGADO", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            return frame_annotated, [], Counter(), 0.0
        start_time = time.time()
        
        # SAHI requiere la ruta o una imagen cargada, pero get_sliced_prediction acepta np.ndarray
        result = get_sliced_prediction(
            frame_np,
            self.detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type=postprocess,
            postprocess_match_threshold=self.iou_threshold,
            verbose=0
        )
        
        detections = []
        detection_counts = Counter()

        # Dibuja los resultados directamente en una copia del frame para Streamlit
        frame_annotated = frame_np.copy()
        
        # Mapeo de color simple (puedes mejorarlo)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)] 

        for prediction in result.object_prediction_list:
            class_id = int(prediction.category.id)
            class_name = CLASSES.get(class_id, "Desconocido")
            confidence = prediction.score.value
            bbox = prediction.bbox.to_voc_bbox()  # [x_min, y_min, x_max, y_max]
            
            detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
            detection_counts[class_name] += 1

            x_min, y_min, x_max, y_max = [int(x) for x in bbox]
            # Obtenemos el color BGR
            color_bgr = self.class_colors.get(class_id, (255, 255, 255)) 
            
            # Rectángulo
            cv2.rectangle(frame_annotated, (x_min, y_min), (x_max, y_max), color_bgr, 2)
            
            # Etiqueta
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame_annotated, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        processing_time = time.time() - start_time
        
        return frame_annotated, detections, detection_counts, processing_time