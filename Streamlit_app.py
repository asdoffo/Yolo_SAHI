import streamlit as st
import time
import pandas as pd
import numpy as np
from collections import deque
import random
import cv2 # Se incluiría para leer la cámara USB real en la Jetson, aquí se usa para simular un frame

# --- CONFIGURACIÓN GENERAL DE LA PÁGINA ---
st.set_page_config(
    page_title="Detección de Cuerpos Extraños en Maní (Supervisor)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTES Y SIMULACIÓN DE CLASES ---
# Simulación de las clases basado en tu data.yaml
CLASSES = {0: "Maní (Normal)", 1: "Cuerpo Extraño (Piedra)", 2: "Cuerpo Extraño (Cáscara)", 3: "Maní Dañado"}
FOREIGN_CLASSES = [1, 2] # IDs de las clases que son "cuerpos extraños"
INFERENCE_LATENCY_SECONDS = 2.0 # Tu tiempo de inferencia reportado

# Buffer para almacenar las últimas inferencias (para estadísticas)
MAX_BUFFER_SIZE = 50
inference_history = deque(maxlen=MAX_BUFFER_SIZE)

# Función para simular la carga y ejecución del modelo ONNX/SAHI
@st.cache_resource
def load_and_warmup_model(onnx_path="best.onnx"):
    """
    Carga el modelo ONNX y realiza un 'warmup'.
    En la implementación real, aquí se cargaría el motor ONNX (por ejemplo, con ONNX Runtime
    o un motor optimizado para Jetson como TensorRT si se usa un .engine) y la configuración SAHI.
    """
    st.write(f"⚙️ Cargando modelo {onnx_path} y preparando entorno...")
    time.sleep(1) # Simula la carga
    st.success("✅ Modelo y configuración SAHI cargados correctamente.")
    return "Modelo ONNX/SAHI (Simulado)"

# Función que simula la inferencia real
def perform_sahi_inference(model, frame):
    """
    Simula el proceso de inferencia real:
    1. Divide la imagen con SAHI.
    2. Ejecuta el modelo YoloV8 ONNX en los recortes.
    3. Combina los resultados.

    Retorna: (frame_con_detecciones, resultados_deteccion_dict)
    """
    start_time = time.time()
    time.sleep(INFERENCE_LATENCY_SECONDS) # Simula la latencia de 2s

    # Simulación de resultados de detección:
    # Genera una lista de detecciones simuladas
    num_detections = random.randint(5, 15)
    detections = []
    for _ in range(num_detections):
        # Simula: [xmin, ymin, xmax, ymax, confianza, clase_id]
        class_id = random.choices(list(CLASSES.keys()), weights=[0.85, 0.05, 0.05, 0.05])[0]
        detections.append({
            'class_id': class_id,
            'class_name': CLASSES[class_id],
            'confidence': random.uniform(0.7, 0.99)
        })

    end_time = time.time()
    latency = end_time - start_time

    # Simulación de dibujar bounding boxes en el frame
    # (En la implementación real, esto usaría cv2.rectangle o similar)
    # Aquí simplemente se retorna el frame de simulación (que no se modifica realmente)

    return frame, detections, latency


# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.title("🛠️ Configuración del Sistema")

# 1. Parámetro de Configuración: Intervalo de Inferencia
st.sidebar.markdown("---")
st.sidebar.header("Control de Inferencias")
# El mínimo de inferencia debe ser el tiempo de procesamiento (2s)
min_interval = INFERENCE_LATENCY_SECONDS
default_interval = max(min_interval, 3.0)

inference_interval = st.sidebar.slider(
    'Intervalo entre Inferencias (segundos)',
    min_value=min_interval,
    max_value=10.0,
    value=default_interval,
    step=0.5,
    help=f"Tiempo que el sistema espera antes de tomar el siguiente frame para procesar. Debe ser >= {min_interval}s."
)
st.sidebar.info(f"El modelo demora **{INFERENCE_LATENCY_SECONDS}s** en procesar una imagen.")

# 2. Simulación de la Cámara USB (Placeholder/Control)
st.sidebar.markdown("---")
st.sidebar.header("Fuente de Video")
if st.sidebar.checkbox("Simular Cámara USB Activa", value=True):
    # En la implementación real, aquí se inicializaría cv2.VideoCapture(0)
    st.sidebar.success("Cámara USB (Simulada) ON")
else:
    st.sidebar.warning("Cámara USB (Simulada) OFF. El procesamiento está pausado.")

# Inicializar/Cargar el Modelo
st.sidebar.markdown("---")
st.sidebar.header("Estado del Modelo")
model_placeholder = st.sidebar.empty() # Placeholder para mostrar el estado de la carga
with model_placeholder:
    model = load_and_warmup_model() # Carga el modelo (solo se ejecuta una vez)

# --- CUERPO PRINCIPAL DE LA APLICACIÓN (VISUALIZACIÓN Y ESTADÍSTICAS) ---
st.title("Sistema de Detección de Cuerpos Extraños en Maní 🥜 (JETSON NANO)")
st.markdown("Acceso para Supervisores. Muestra detecciones en tiempo real y métricas.")

# Contenedor principal para visualización
col_video, col_stats = st.columns([2, 1])

# --- Columna de Video/Detección ---
with col_video:
    st.header("Visualización en Tiempo Real")
    video_placeholder = st.empty() # Placeholder para el frame del video
    status_placeholder = st.empty() # Placeholder para el estado de la inferencia

# --- Columna de Estadísticas ---
with col_stats:
    st.header("Estadísticas de Lote")
    # Indicadores Clave
    kpi1, kpi2 = st.columns(2)
    kpi_total_detections = kpi1.empty()
    kpi_foreign_rate = kpi2.empty()

    # Historial / Tabla de Lote
    st.subheader("Historial Reciente (Últimas Detecciones)")
    stats_table_placeholder = st.empty()
    chart_placeholder = st.empty()

# --- Bucle de Procesamiento (El corazón de la aplicación) ---
if st.sidebar.checkbox("Simular Cámara USB Activa", value=True):
    # Inicializar el contador de frames y la última inferencia
    frame_counter = 0
    last_inference_time = time.time()

    while True:
        # 1. Captura de Frame (Simulada)
        # En la vida real, usaríamos cap.read() si se inicializó cv2.VideoCapture(0)
        # Aquí creamos un frame simulado (fondo gris)
        frame = np.zeros((480, 640, 3), dtype=np.uint8) + 50
        frame_counter += 1

        # 2. Decisión de Inferencia
        current_time = time.time()
        time_since_last_inference = current_time - last_inference_time

        if time_since_last_inference >= inference_interval:
            # --- REALIZAR INFERENCIA ---
            status_placeholder.warning(f"⏳ Procesando Frame {frame_counter}...")
            frame_processed, detections, latency = perform_sahi_inference(model, frame)
            last_inference_time = current_time

            # Añadir resultados al historial
            foreign_count = sum(1 for det in detections if det['class_id'] in FOREIGN_CLASSES)
            total_detections = len(detections)
            inference_history.append({
                'Time': pd.Timestamp.now().strftime('%H:%M:%S'),
                'Total_Detections': total_detections,
                'Foreign_Count': foreign_count,
                'Foreign_Rate': (foreign_count / total_detections) if total_detections else 0,
                'Latency_s': round(latency, 2)
            })

            # 3. Actualizar Visualización
            # Dibujar el contador de cuerpos extraños en la simulación
            if foreign_count > 0:
                 cv2.putText(frame_processed, f"CUERPO EXTRAÑO DETECTADO: {foreign_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                 cv2.putText(frame_processed, "Producción Limpia", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Convertir el frame (OpenCV BGR) a RGB para Streamlit
            frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, caption=f"Frame #{frame_counter} | Detecciones: {total_detections}", use_column_width=True)
            status_placeholder.success(f"✅ Inferencia completada en {latency:.2f}s. Cuerpos Extraños: {foreign_count}.")

            # 4. Actualizar Estadísticas
            df_history = pd.DataFrame(inference_history)

            if not df_history.empty:
                # KPI 1: Detecciones Totales en el Lote
                total_foreign = df_history['Foreign_Count'].sum()
                kpi_total_detections.metric("Cuerpos Extraños (Lote)", f"{total_foreign} unid.")

                # KPI 2: Tasa Promedio de Cuerpos Extraños
                avg_rate = df_history['Foreign_Rate'].mean() * 100
                kpi_foreign_rate.metric("Tasa Promedio (%)", f"{avg_rate:.2f} %")

                # Tabla de Historial
                stats_table_placeholder.dataframe(
                    df_history[['Time', 'Total_Detections', 'Foreign_Count', 'Foreign_Rate', 'Latency_s']]
                    .rename(columns={'Foreign_Count': 'Extraños', 'Total_Detections': 'Total', 'Foreign_Rate': 'Tasa (%)', 'Latency_s': 'Latencia (s)'})
                    .sort_values(by='Time', ascending=False)
                    .head(10),
                    use_container_width=True,
                    hide_index=True
                )

                # Gráfico de Tasa de Extraños
                chart_placeholder.line_chart(df_history['Foreign_Rate'].rename('Tasa de Cuerpos Extraños'))

        else:
            # No es tiempo de inferencia, solo mostrar el frame 'vivo' (sin procesamiento)
            frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame_display, "Esperando proxima inferencia...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video_placeholder.image(frame_display, caption=f"Esperando... Siguiente en {inference_interval - time_since_last_inference:.1f}s", use_column_width=True)

        # Pequeña pausa para evitar que el bucle consuma demasiado CPU cuando no está haciendo inferencia.
        # Es clave para aplicaciones en bucle en Streamlit.
        time.sleep(0.1)

else:
    # Estado cuando la cámara no está activa
    st.warning("El sistema está inactivo. Por favor, active la 'Simular Cámara USB Activa' en la barra lateral para iniciar la detección.")